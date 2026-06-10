"""Common task lifecycle."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, cast

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage

from simple_agent.index.indexer import AgentIndex
from simple_agent.run_log import runtime_logger
from simple_agent.task_manager.base_lifecycle import (
    BaseTaskLifecycle,
    SessionState,
    TaskLifecycleError,
    USER_TASK_COMPACT_SYSTEM_PROMPT,
    USER_TASK_SYSTEM_PROMPT,
    render_prompt_template,
    task_instruction_text,
)
from simple_agent.task_manager.models import ManagedTask, ToolCallTask, CommonTask
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

IMPORTANT: Focus on current task: {{ task }}, if task is complex, consider decompose complex task and create sub task to do, and you can also use tools to first explore around and gather some useful context.
You can create these following sub tasks.

{% if task_instruction %}
{{ task_instruction }}
{% endif %}

## Reminder instruction
{% if is_early_run %}
- Early in the task, create a sub task before doing broad work.
{% elif is_mid_run %}
- Mid run, prefer creating a sub task for the next coherent unit of work.
{% else %}
- Create the next sub task now before continuing more work.
{% endif %}
- Keep each created task small, atomic, and directly tied to finishing the current task.
- Always finish the current task as soon as the requested work is complete.
</system-instruction>
"""


USER_COMPACTION_INSTRUCTION_TEMPLATE = """\
Runtime instruction for compacting phase:
- Complete the compacted user task information first: define the task result.
- Record every must-include tool call based on the compacted task result to avoid context loss.
- Use only compact tools: set the compacted user task result, record must-include tool calls, then finish it.

Task view to compact:
{{ task_view }}
"""


class CommonTaskLifecycle(BaseTaskLifecycle):
    task: CommonTask | None
    _agent_index: AgentIndex | None

    def set_data(self, session_state: SessionState) -> None:
        self._session_state = session_state
        self.created_task = None
        self.finished_task = None
        task = self._session_state.next_task
        if task is None:
            raise TaskLifecycleError("Session state has no next task")
        if task.kind != "user_task":
            raise TaskLifecycleError("Active lifecycle task is not a user task")
        self.task = cast(CommonTask, task)
        self._agent_index = AgentIndex(base_dir=self._session_state.base_dir)

    def clear_data(self) -> None:
        super().clear_data()
        self._agent_index = None

    def instruction_text(self) -> str:
        tool_call_count = _count_tool_calls(self.task.children)
        task_info = None
        if tool_call_count > 10:
            task_info = TaskTreeRenderer(format="tree", depth=1).render(self.task)
        return render_prompt_template(
            USER_TASK_INSTRUCTION_TEMPLATE,
            task=self.task.title,
            task_info=task_info,
            task_instruction=task_instruction_text(
                has_common_task=True,
                has_repo_memory_task=True,
            ),
            is_early_run=tool_call_count == 0,
            is_mid_run=0 < tool_call_count <= 5,
        )

    def finish_task(self, *, result: str | None = None) -> CommonTask:
        self.task.status = "done"
        self.task.result = result
        self.task.touch()
        self.finished_task = self.task
        return self.task

    def create_tools(self) -> list[AgentTool]:
        return [
            *self.create_next_task_tools(enabled_task_kinds=["common", "repo_memory"]),
            self.create_finish_common_task_tool(),
            self._agent_index.create_tree_tool(),
            *create_all_coding_tools("."),
        ]

    async def run(
        self,
        *,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None = None,
    ) -> SessionState:
        if self.task.status == "done":
            return await self.run_compact_one_turn(
                agent_process=agent_process,
                cancel_event=cancel_event,
            )

        return await self.run_one_turn(
            agent_process=agent_process,
            cancel_event=cancel_event,
        )

    def should_compact_after_turn(self) -> bool:
        return _count_tool_calls(self.task.children) > 10

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
            parent_task=task,
            cancel_event=cancel_event,
        )
        assistant_message = turn_result.assistant_message
        assistant_entry = turn_result.assistant_entry
        tool_result_entries = turn_result.tool_result_entries
        tool_call_records = turn_result.tool_call_records
        tool_call_tasks = turn_result.tool_call_tasks
        has_tool_call = turn_result.has_tool_call
        new_messages = [assistant_entry, *tool_result_entries]
        turn_end_message_id = new_messages[-1].id
        self._session_state.append_messages(new_messages)

        task.children.extend(tool_call_tasks)
        if tool_call_tasks:
            task.touch()

        if not has_tool_call and task.status != "done":
            self.finish_task()

        def route_after_turn() -> None:
            created_task = self.created_task
            if created_task is not None:
                created_task.id = self._session_state.allocate_task_id()
                if hasattr(created_task, "start_message_id"):
                    created_task.start_message_id = assistant_entry.id
                task.children.append(created_task)
                task.touch()
                self.set_next_task(created_task.id, created_task)
                return

            if self.finished_task is not None:
                self.stamp_finished_task(end_message_id=turn_end_message_id)
                if self.should_compact_after_turn():
                    self.set_next_task(task.id, task)
                    return
                self.set_next_task(task.parent_id, None)
                return

            if has_tool_call:
                self.set_next_task(task.id, task)
                return

        route_after_turn()

        self.stamp_finished_task(end_message_id=turn_end_message_id)

        runtime_logger.log_handle_running(
            session_id=self._session_state._require_session_id(),
            messages=context_messages,
            user_instruction_message=user_instruction_message,
            assistant_message_id=assistant_entry.id,
            assistant_message=assistant_message,
            tool_result_entries=tool_result_entries,
        )

        tasks_to_sync: list[ManagedTask] = [task, *task.children]
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
            parent_task=task,
            cancel_event=cancel_event,
        )
        new_messages = [turn_result.assistant_entry, *turn_result.tool_result_entries]

        task.children.extend(turn_result.tool_call_tasks)
        if turn_result.tool_call_tasks:
            task.touch()

        replacement_entries: list[MessageEntry] = []
        if turn_result.has_tool_call:
            self._session_state.append_messages(new_messages)
            self.set_next_task(task.id, task)
        else:
            if task.start_message_id is None:
                raise TaskLifecycleError("Compact task is missing start message id")
            end_message_id = task.end_message_id or self._session_state.messages[-1].id
            compacted_messages = self.format_messages_from_user_task(task)
            replacement_entries = self._session_state.replace_message_range(
                start_message_id=task.start_message_id,
                end_message_id=end_message_id,
                replacement_messages=compacted_messages,
            )
            self.set_next_task(task.parent_id, None)

        tasks_to_sync: list[ManagedTask] = [task, *task.children]
        with self._session_state.create_database_session() as session:
            self._session_state.append_tool_calls_to_database(
                tool_calls=turn_result.tool_call_records,
                session=session,
            )
            self._session_state.append_tasks_to_database(
                tasks=tasks_to_sync,
                session=session,
            )
            if replacement_entries:
                self._session_state.replace_message_range_in_database(
                    start_message_id=task.start_message_id,
                    end_message_id=end_message_id,
                    replacement_messages=replacement_entries,
                    session=session,
                )
            else:
                self._session_state.append_messages_to_database(
                    messages=new_messages,
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

    def format_messages_from_user_task(self, user_task: CommonTask) -> list[UserMessage | AssistantMessage | ToolResultMessage]:
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


def _count_tool_calls(tasks: list[ManagedTask]) -> int:
    count = 0
    for task in tasks:
        if task.kind == "tool_call":
            count += 1
        count += _count_tool_calls(task.children)
    return count
