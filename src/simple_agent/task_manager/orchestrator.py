"""Orchestrator lifecycle."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, cast

from pi.agent import AgentTool, AgentToolResult
from pi.agent.types import AgentMessage
from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage

from simple_agent.index.indexer import AgentIndex
from simple_agent.run_log import runtime_logger
from simple_agent.task_manager.base_lifecycle import (
    BaseTaskLifecycle,
    SessionState,
    TaskLifecycleError,
    USER_TASK_SYSTEM_PROMPT,
    render_prompt_template,
    task_instruction_text,
)
from simple_agent.task_manager.models import UserTask
from simple_agent.task_manager.review import TaskTreeRenderer
from simple_agent.tool.common_tools import create_all_coding_tools

if TYPE_CHECKING:
    from simple_agent.process.agent_process import AgentProcess


USER_TASK_INSTRUCTION_TEMPLATE = """\
<system-instruction>
{% if task_info %}
## Current task process information
{{ task_info }}
{% endif %}

IMPORTANT: Focus on current task: {{ task }}. If the task is complex, decompose it into sub-tasks that explore and search for context using tools. Sub-tasks should gather facts and inspect code — do NOT create sub-tasks whose goal is to generate a text response or write a summary. Use tools directly whenever possible before delegating to a sub-task.

{% if task_instruction %}
{{ task_instruction }}
{% endif %}
</system-instruction>
"""


USER_COMPACTION_INSTRUCTION_TEMPLATE = """\
System metadata phase — auto-triggered, not requested by the user. Do NOT generate
plain text or conversational responses; use only the provided tools.

Compaction instructions:
- Review the task view and record every must-include tool-call log id needed to preserve context.
- Use only compact tools while recording useful tool-call log ids.
- When all useful tool-call logs are recorded, respond without tool calls to finish compaction.

Task view to compact:
{{ task_view }}
"""


USER_INDEX_MEMORY_INSTRUCTION_TEMPLATE = """\
System metadata phase — auto-triggered, not requested by the user. Do NOT generate
plain text or conversational responses; use only the provided tools.

Index memory upsert instructions:
- Review the finished task.
- First call index_tree to inspect the existing index memory and repository tree context for the task scope.

When to update an index entry:
- You explored and reviewed the entry thoroughly in this task, AND the entry is significant and key.
- An existing description is wrong or outdated because the entry has been modified — amend it to match current reality.

When NOT to update an index entry:
- The entry already has a description that is proper, thorough, and comprehensive — no need to append or duplicate.
- The entry was NOT explored or reviewed in this task — do not add a description for it.
- The entry is not very important and its purpose can be easily inferred from its name — skip it.
- Most importantly: do not update entries whose index descriptions are already thorough. Only update the most
  significant entries touched by this task, or correct wrong / legacy descriptions caused by code changes.

- Each upserted description should be short, factual, and say what the entry does.
- If no index entry needs updating, finish immediately without any explanation — just respond without tool calls.
- When useful memory is written, or no significant in-scope memory is needed, respond without tool calls to finish this phase.

Task view:
{{ task_view }}
"""

USER_TASK_COMPACT_SYSTEM_PROMPT = """You are in a system metadata phase — this is auto-triggered by the system,
not requested by the user. Do NOT generate plain text or conversational responses.
Use only the compact tools to record useful tool-call log IDs. When all useful
tool-call logs are recorded, respond without tool calls to finish compaction."""

USER_TASK_INDEX_MEMORY_SYSTEM_PROMPT = """You are in a system metadata phase — this is auto-triggered by the
system, not requested by the user. Do NOT generate plain text or conversational
responses. Use the index tools to inspect and update repository memory. When
enough useful memory is written, respond without tool calls to finish this phase."""

ORCHESTRATOR_SYSTEM_PROMPT = """You are an orchestrator agent. Your job is to help the agent manage tasks.
NEVER generate a text response — use only the provided tools.
Inspect the task progress and task plan, then decide whether to create a
sub-task, update the task plan, or do nothing and let the current task continue."""

ORCHESTRATOR_INSTRUCTION_TEMPLATE = """\
{% if response %}
## Agent Response
{{ response }}
{% endif %}

Based on current task progress:
{{ task_progress }}

And task plan:
{{ task_plan }}

Review the agent's response and task progress. Decide what to do next:
- Use `set_instruction` to give the agent a new task to work on.
- Use `update_task_plan` to mark finished items and add new pending items.

When to update the task plan:
1. Based on the task context, mark already-finished tasks as [x].
2. If the current task is complex, decompose it and add new pending tasks as [ ].
3. Based on task progress, think about the next task to run and reflect it in the plan.
   If the remaining work is simple, keep running the current task without changes.

If no action is needed, respond without tool calls."""


class OrchestratorLifecycle(BaseTaskLifecycle):
    task: UserTask | None
    _agent_index: AgentIndex | None

    def set_data(self, session_state: SessionState) -> None:
        self._session_state = session_state
        task = self._session_state.current_task
        if task is None:
            raise TaskLifecycleError("Session state has no current task")
        if task.kind != "user_task":
            raise TaskLifecycleError("Orchestrator expects a user_task")
        self.task = cast(UserTask, task)
        self._agent_index = AgentIndex(base_dir=self._session_state.workspace_dir)

    def clear_data(self) -> None:
        super().clear_data()
        self._agent_index = None

    def instruction_text(self) -> str:
        tool_call_count = len(self.task.tool_call_log_ids)
        task_info = None
        if tool_call_count > 10:
            task_info = TaskTreeRenderer(format="tree", depth=1).render(self.task)
        return render_prompt_template(
            USER_TASK_INSTRUCTION_TEMPLATE,
            task=self.task.title,
            task_info=task_info,
            task_instruction=task_instruction_text(
                has_common_task=True,
                has_repo_memory_task=False,
            ),
        )

    def finish_task(self, *, result: str | None = None) -> UserTask:
        self.task.status = "done"
        self.task.result = result
        self.task.touch()
        self.finished_task = self.task
        return self.task

    def create_tools(self) -> list[AgentTool]:
        return [
            self.create_finish_common_task_tool(),
            self._agent_index.create_tree_tool(),
            *create_all_coding_tools(self._session_state.workspace_dir),
        ]

    async def run(
        self,
        *,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None = None,
    ) -> SessionState:
        return await self._manage_task(
            agent_process=agent_process,
            cancel_event=cancel_event,
        )

    def _build_set_instruction_tool(self) -> AgentTool:
        """Build a tool that lets the orchestrator set an instruction for the task agent."""
        tool = AgentTool(
            name="set_instruction",
            description=(
                "Set an instruction for the task agent. The agent will see this "
                "as its current task to work on. Call this to guide the agent "
                "on what to do next."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "The instruction for the task agent.",
                    },
                },
                "required": ["instruction"],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            self.task.instruction = params["instruction"]
            return AgentToolResult(content=[TextContent(text="Instruction set.")])

        tool.execute = execute
        return tool

    def _build_update_task_plan_tool(self) -> AgentTool:
        """Build a tool that updates the full task plan on SessionState.

        The agent passes a markdown task list representing the entire task tree
        plan. Completed tasks are `- [x]`, pending tasks are `- [ ]`.
        The orchestrator reads this in _manage_task post-handle to sync the
        actual task tree.
        """
        tool = AgentTool(
            name="update_task_plan",
            description=(
                "Update the full task plan. Pass a markdown task list where "
                "`- [x]` marks completed tasks and `- [ ]` marks pending tasks. "
                "The orchestrator will sync the actual task tree from this plan. "
                "Include ALL tasks (completed and pending) each time you call this."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_plan": {
                        "type": "string",
                        "description": (
                            "The full markdown task plan, e.g.:\n"
                            "## Tasks\n\n"
                            "- [x] Completed task A\n"
                            "- [ ] Pending task B\n"
                            "- [ ] Pending task C\n"
                        ),
                    },
                },
                "required": ["task_plan"],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            self.task.task_plan = params["task_plan"]
            return AgentToolResult(
                content=[TextContent(text="Task plan updated.")]
            )

        tool.execute = execute
        return tool

    async def _manage_task(
        self,
        *,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None = None,
    ) -> SessionState:
        """Inspect context and manage the task plan.

        0. Check status: if done, end the run
        1. Prepare: build prompt, tools, and message buffer
        2. Run: call _run_loop to let the agent review and update the plan
        3. Post-handle: route back to CommonTaskLifecycle
        """
        task = self.task

        # ── 1. Prepare ──────────────────────────────────────────────────
        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT
        task_progress = TaskTreeRenderer(format="tree", depth=1).render(task)
        task_plan = self.task.task_plan or "(no plan yet)"
        instruction_text = render_prompt_template(
            ORCHESTRATOR_INSTRUCTION_TEMPLATE,
            response=self.task.response,
            task_progress=task_progress,
            task_plan=task_plan,
        )
        instruction_message = UserMessage(
            content=[TextContent(text=instruction_text)],
            timestamp=int(time.time() * 1000),
        )
        tools: list[AgentTool] = [
            self._build_set_instruction_tool(),
            self._build_update_task_plan_tool(),
        ]
        buffer: list[AgentMessage] = [
            *self._session_state.message_values(),
            instruction_message,
        ]

        # ── 2. Run ──────────────────────────────────────────────────────
        await self._run_loop(
            system_prompt=system_prompt,
            messages=buffer,
            tools=tools,
            agent_process=agent_process,
            cancel_event=cancel_event,
        )

        # ── 3. Post-handle ──────────────────────────────────────────────
        self._session_state.next_phase = "common_task"
        self.set_current_task(task.id, task)
        return self._session_state

    async def _run_loop(
        self,
        *,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        """Run an LLM/tool loop on a caller-owned message buffer.

        Does NOT modify session_state or the database. *messages* is mutated
        in-place: assistant and tool-result messages are appended to it.
        The caller provides the initial messages and owns post-handling.
        """
        while True:
            assistant_message = await agent_process.call_llm_step(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                cancel_event=cancel_event,
            )
            messages.append(assistant_message)

            tool_results = await agent_process.run_tool_calls_step(
                tools=tools,
                assistant_message=assistant_message,
                cancel_event=cancel_event,
            )
            messages.extend(tool_results)

            if not tool_results:
                break

    def should_compact_after_turn(self) -> bool:
        return len(self.task.tool_call_log_ids) > 10

    async def run_one_turn(
        self,
        *,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None = None,
    ) -> SessionState:
        task = self.task
        tools = self.create_tools()
        user_instruction_message = UserMessage(
            content=[TextContent(text=self.instruction_text())],
            timestamp=int(time.time() * 1000),
        )
        context_messages = list(self._session_state.messages)
        run_messages = [*self._session_state.message_values(), user_instruction_message]
        turn_result = await self.run_agent_turn(
            agent_process=agent_process,
            system_prompt=USER_TASK_SYSTEM_PROMPT,
            messages=run_messages,
            tools=tools,
            cancel_event=cancel_event,
        )
        assistant_message = turn_result.assistant_message
        assistant_entry = turn_result.assistant_entry
        tool_result_entries = turn_result.tool_result_entries
        tool_call_records = turn_result.tool_call_records
        tool_call_log_ids = turn_result.tool_call_log_ids
        has_tool_call = turn_result.has_tool_call
        new_messages = [assistant_entry, *tool_result_entries]
        turn_end_message_id = new_messages[-1].id
        self._session_state.append_messages(new_messages)

        task.tool_call_log_ids.extend(tool_call_log_ids)
        if tool_call_log_ids:
            task.touch()

        if not has_tool_call and task.status != "done":
            self.finish_task()

        def route_after_turn() -> None:
            if self.finished_task is not None:
                self.stamp_finished_task(end_message_id=turn_end_message_id)
                if self.should_compact_after_turn():
                    self.set_current_task(task.id, task)
                    return
                self.set_current_task(None, None)
                return

            if has_tool_call:
                self.set_current_task(task.id, task)
                return

        route_after_turn()

        if self.finished_task is not None:
            self.stamp_finished_task(end_message_id=turn_end_message_id)

        runtime_logger.log_handle_running(
            session_id=self._session_state._require_session_id(),
            messages=context_messages,
            user_instruction_message=user_instruction_message,
            assistant_message_id=assistant_entry.id,
            assistant_message=assistant_message,
            tool_result_entries=tool_result_entries,
        )

        tasks_to_sync = [task]
        with self._session_state.create_database_session() as session:
            self._session_state.append_messages_to_database(
                messages=new_messages,
                session=session,
            )
            self._session_state.append_tool_calls_to_database(
                tool_calls=tool_call_records,
                session=session,
            )
            self._session_state.append_tasks_to_database(
                tasks=tasks_to_sync,
                session=session,
            )
            session.commit()
        self.clear_turn_indicators()
        return self._session_state

    # ------------------------------------------------------------------
    # User-task compaction phase
    # ------------------------------------------------------------------

    def compaction_instruction_text(self) -> str:
        task_view = TaskTreeRenderer(
            format="tree",
            depth=None,
        ).render(self.task)
        return render_prompt_template(
            USER_COMPACTION_INSTRUCTION_TEMPLATE,
            task_view=task_view,
        )

    async def run_compact_one_turn(
        self,
        *,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None = None,
    ) -> SessionState:
        task = self.task
        tools = self.create_compact_one_turn_tools()
        user_instruction_message = UserMessage(
            content=[TextContent(text=self.compaction_instruction_text())],
            timestamp=int(time.time() * 1000),
        )
        run_messages = [*self._session_state.message_values(), user_instruction_message]
        turn_result = await self.run_agent_turn(
            agent_process=agent_process,
            system_prompt=USER_TASK_COMPACT_SYSTEM_PROMPT,
            messages=run_messages,
            tools=tools,
            cancel_event=cancel_event,
        )
        new_messages = [turn_result.assistant_entry, *turn_result.tool_result_entries]
        self._session_state.append_messages(new_messages)

        task.tool_call_log_ids.extend(turn_result.tool_call_log_ids)
        if turn_result.tool_call_log_ids:
            task.touch()

        if turn_result.has_tool_call:
            self.set_current_task(task.id, task)
        else:
            task.status = "index_memory_upsert"
            task.touch()
            self.set_current_task(task.id, task)

        tasks_to_sync = [task]
        with self._session_state.create_database_session() as session:
            self._session_state.append_messages_to_database(
                messages=new_messages,
                session=session,
            )
            self._session_state.append_tool_calls_to_database(
                tool_calls=turn_result.tool_call_records,
                session=session,
            )
            self._session_state.append_tasks_to_database(
                tasks=tasks_to_sync,
                session=session,
            )
            session.commit()
        return self._session_state

    def index_memory_instruction_text(self) -> str:
        task_view = TaskTreeRenderer(
            format="tree",
            depth=None,
        ).render(self.task)
        return render_prompt_template(
            USER_INDEX_MEMORY_INSTRUCTION_TEMPLATE,
            task_view=task_view,
        )

    async def run_index_memory_upsert_one_turn(
        self,
        *,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None = None,
    ) -> SessionState:
        task = self.task
        tools = self._agent_index.create_tools()
        user_instruction_message = UserMessage(
            content=[TextContent(text=self.index_memory_instruction_text())],
            timestamp=int(time.time() * 1000),
        )
        context_messages = list(self._session_state.messages)
        run_messages = [*self._session_state.message_values(), user_instruction_message]
        turn_result = await self.run_agent_turn(
            agent_process=agent_process,
            system_prompt=USER_TASK_INDEX_MEMORY_SYSTEM_PROMPT,
            messages=run_messages,
            tools=tools,
            cancel_event=cancel_event,
        )
        assistant_message = turn_result.assistant_message
        new_messages = [turn_result.assistant_entry, *turn_result.tool_result_entries]
        self._session_state.append_messages(new_messages)

        task.tool_call_log_ids.extend(turn_result.tool_call_log_ids)
        if turn_result.tool_call_log_ids:
            task.touch()

        if turn_result.has_tool_call:
            self.set_current_task(task.id, task)
        else:
            task.status = "compact_finished"
            task.touch()
            self.set_current_task(task.id, task)

        runtime_logger.log_handle_running(
            session_id=self._session_state._require_session_id(),
            messages=context_messages,
            user_instruction_message=user_instruction_message,
            assistant_message_id=turn_result.assistant_entry.id,
            assistant_message=assistant_message,
            tool_result_entries=turn_result.tool_result_entries,
        )

        tasks_to_sync = [task]
        with self._session_state.create_database_session() as session:
            self._session_state.append_messages_to_database(
                messages=new_messages,
                session=session,
            )
            self._session_state.append_tool_calls_to_database(
                tool_calls=turn_result.tool_call_records,
                session=session,
            )
            self._session_state.append_tasks_to_database(
                tasks=tasks_to_sync,
                session=session,
            )
            session.commit()
        return self._session_state

    def compact_finished_task(self) -> SessionState:
        task = self.task
        if task.start_message_id is None:
            raise TaskLifecycleError("Compact task is missing start message id")
        end_message_id = self._session_state.messages[-1].id

        compacted_messages = self.format_messages_from_user_task(task)
        replacement_entries = self._session_state.replace_message_range(
            start_message_id=task.start_message_id,
            end_message_id=end_message_id,
            replacement_messages=compacted_messages,
        )
        task.status = "done"
        task.touch()
        self.set_current_task(None, None)

        with self._session_state.create_database_session() as session:
            self._session_state.replace_message_range_in_database(
                start_message_id=task.start_message_id,
                end_message_id=end_message_id,
                replacement_messages=replacement_entries,
                session=session,
            )
            self._session_state.append_tasks_to_database(
                tasks=[task],
                session=session,
            )
            session.commit()
        return self._session_state

    def create_finish_common_task_tool(self) -> AgentTool:
        tool = AgentTool(
            name="finish_common_task",
            description=(
                "Mark the current common task as completed. Call when this "
                "task is fully satisfied and no child task is active."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "Optional concise result for this common task"},
                },
                "required": [],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            task = self.finish_task(
                result=params.get("result"),
            )
            return AgentToolResult(content=[TextContent(text=f"Common task finished: {task.result or task.title}")])

        tool.execute = execute
        return tool

    def create_compact_one_turn_tools(self) -> list[AgentTool]:
        record_tool = AgentTool(
            name="record_compacted_tool_call_log",
            description="Record one useful tool-call log id for the compacted user task.",
            parameters={
                "type": "object",
                "properties": {
                    "tool_call_log_id": {
                        "type": "integer",
                        "description": "Tool-call log id of the useful tool result to keep in compacted context.",
                    },
                },
                "required": ["tool_call_log_id"],
            },
        )

        async def record_execute(tool_call_id, params, cancel_event=None, on_update=None):
            self.record_compacted_tool_call_log(
                tool_call_log_id=params["tool_call_log_id"],
            )
            return AgentToolResult(content=[TextContent(text="recorded compacted tool call log")])

        record_tool.execute = record_execute
        return [record_tool]

    def record_compacted_tool_call_log(self, *, tool_call_log_id: int) -> None:
        if tool_call_log_id not in self.task.compacted_tool_call_log_ids:
            self.task.compacted_tool_call_log_ids.append(tool_call_log_id)
            self.task.touch()

    def format_messages_from_user_task(self, user_task: UserTask) -> list[UserMessage | AssistantMessage | ToolResultMessage]:
        compacted_tool_calls = self._session_state.compacted_tool_calls(
            user_task.compacted_tool_call_log_ids,
        )
        tool_calls = [
            tool_call
            for tool_call, _tool_result_message in compacted_tool_calls
            if tool_call is not None
        ]
        tool_result_messages = [
            tool_result_message
            for _tool_call, tool_result_message in compacted_tool_calls
        ]
        tool_refs = [
            message.tool_call_id
            for message in tool_result_messages
        ]
        result_text = user_task.result or user_task.title
        assistant_text = (
            f"Finished task: {user_task.title}\n"
            f"Result: {result_text}\n"
            f"Following tool calls preserve useful context: {tool_refs}"
        )
        return [
            UserMessage(content=[TextContent(text=user_task.title)], timestamp=int(time.time() * 1000)),
            AssistantMessage(role="assistant", content=[TextContent(text=assistant_text), *tool_calls]),
            *tool_result_messages,
        ]
