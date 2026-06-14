"""Common task lifecycle."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, cast

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import TextContent, UserMessage

from simple_agent.index.indexer import AgentIndex
from simple_agent.run_log import runtime_logger
from simple_agent.task_manager.base_lifecycle import (
    BaseTaskLifecycle,
    SessionState,
    TaskLifecycleError,
    USER_TASK_SYSTEM_PROMPT,
    render_prompt_template,
)
from simple_agent.task_manager.models import ManagedTask, ToolCallTask, UserTask
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

IMPORTANT: Focus on current task: {{ task }}. Use tools to explore, search, and gather context. Inspect files, run commands, and collect facts needed to complete the task.
IMPORTANT: If you have finished the current task, you MUST immediately call `finish_common_task` to mark it as done.
</system-instruction>
"""



class CommonTaskLifecycle(BaseTaskLifecycle):
    task: UserTask | None
    _agent_index: AgentIndex | None

    def set_data(self, session_state: SessionState) -> None:
        self._session_state = session_state
        self.finished_task = None
        task = self._session_state.current_task
        if task is None:
            raise TaskLifecycleError("Session state has no next task")
        if task.kind != "user_task":
            raise TaskLifecycleError("Active lifecycle task is not a user task")
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
        )

    def finish_task(self, *, result: str | None = None) -> UserTask:
        self.task.status = "done"
        self.task.result = result
        self.task.touch()
        self.finished_task = self.task
        return self.task

    def create_tools(self) -> list[AgentTool]:
        return [
            self._agent_index.create_tree_tool(),
            self.create_finish_common_task_tool(),
            *create_all_coding_tools(self._session_state.workspace_dir),
        ]

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

    def _should_orchestrate(self) -> bool:
        return self.finished_task is not None

    async def run(
        self,
        *,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None = None,
    ) -> SessionState:
        return await self.run_one_turn(
            agent_process=agent_process,
            cancel_event=cancel_event,
        )

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

        if self.finished_task is not None:
            self.stamp_finished_task(end_message_id=turn_end_message_id)

        # Route after turn
        next_phase = "orchestrator" if self._should_orchestrate() else "common_task"
        self._session_state.next_phase = next_phase

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