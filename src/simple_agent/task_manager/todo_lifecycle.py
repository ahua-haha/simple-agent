"""Todo task lifecycle."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, cast

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import TextContent, UserMessage

from simple_agent.run_log import runtime_logger
from simple_agent.task_manager.base_lifecycle import (
    BaseTaskLifecycle,
    SessionState,
    TaskLifecycleError,
    USER_TASK_SYSTEM_PROMPT,
)
from simple_agent.task_manager.models import ManagedTask, TodoTask
from simple_agent.tool.common_tools import create_all_coding_tools

if TYPE_CHECKING:
    from simple_agent.process.agent_process import AgentProcess


class TodoTaskLifecycle(BaseTaskLifecycle):
    task: TodoTask | None

    def set_data(self, session_state: SessionState) -> None:
        self._session_state = session_state
        self.created_task = None
        self.finished_task = None
        task = self._session_state.next_task
        if task is None:
            raise TaskLifecycleError("Session state has no next task")
        if task.kind != "todo":
            raise TaskLifecycleError("Active lifecycle task is not a todo task")
        self.task = cast(TodoTask, task)

    def clear_data(self) -> None:
        super().clear_data()
        self.task = None

    def instruction_text(self) -> str:
        task = self._require_todo_task_data()
        if _count_tool_calls(task.children) > 10:
            return (
                "Runtime instruction for this turn:\n"
                "- More than 10 tool calls have run for the active todo.\n"
                "- Determine whether the active todo is finished.\n"
                "- If it is finished, call finish_todo now with a concise result.\n"
                "- If it is not finished, do only the next action needed to complete it."
            )
        return (
            "Runtime instruction for this turn:\n"
            f"- Focus on the active todo: {task.title}\n"
            "- Use tools only for work needed by this todo.\n"
            "- Call finish_todo immediately when it is complete."
        )

    def finish_task(self, *, result: str | None = None) -> TodoTask:
        task = self._require_todo_task_data()
        task.status = "done"
        task.result = result
        task.touch()
        self.finished_task = task
        self._session_state.next_task_id_to_run = task.parent_id
        self._session_state.next_task = None
        return task

    def error_task(self, *, error: str) -> TodoTask:
        task = self._require_todo_task_data()
        task.status = "error"
        task.error = error
        task.touch()
        self.finished_task = task
        self._session_state.next_task_id_to_run = task.parent_id
        self._session_state.next_task = None
        return task

    def _require_todo_task_data(self) -> TodoTask:
        if self.task is None:
            raise TaskLifecycleError("Task lifecycle has no todo task data")
        return self.task

    def create_tools(self) -> list[AgentTool]:
        return [
            self.create_finish_todo_tool(),
            self.create_error_todo_tool(),
            *create_all_coding_tools("."),
        ]

    async def run(
        self,
        *,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None = None,
    ) -> SessionState:
        task = self._require_todo_task_data()
        if task.status != "active":
            self._session_state.next_task_id_to_run = task.parent_id
            self._session_state.next_task = None
            return self._session_state
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
        task = self._require_todo_task_data()
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
        if turn_result.has_tool_call:
            self.stamp_finished_task(end_message_id=assistant_entry.id)
            task.children.extend(tool_call_tasks)
            if tool_call_tasks:
                task.touch()

        new_messages = [assistant_entry, *tool_result_entries]
        self._session_state.append_messages(new_messages)

        if not turn_result.has_tool_call:
            if task.status == "active":
                self.finish_task()
            self.stamp_finished_task(end_message_id=assistant_entry.id)
        elif task.status == "active":
            self.set_next_task(task.id, task)

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

    def create_finish_todo_tool(self) -> AgentTool:
        tool = AgentTool(
            name="finish_todo",
            description=(
                "Mark the active todo as completed. Call immediately when the "
                "todo is done before moving to the next item."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "Optional concise result for this todo"},
                },
                "required": [],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            task = self.finish_task(
                result=params.get("result"),
            )
            return AgentToolResult(content=[TextContent(text=f"Todo finished: {task.result or task.title}")])

        tool.execute = execute
        return tool

    def create_error_todo_tool(self) -> AgentTool:
        tool = AgentTool(
            name="error_todo",
            description=(
                "Cancel the active todo because it cannot be completed. If "
                "there is a clear next step, create a revised todo after this."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "error": {"type": "string", "description": "Error details for the active todo"},
                },
                "required": ["error"],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            task = self.error_task(
                error=params["error"],
            )
            return AgentToolResult(content=[TextContent(text=f"Todo errored: {task.error or task.title}")])

        tool.execute = execute
        return tool


def _count_tool_calls(tasks: list[ManagedTask]) -> int:
    return sum(1 for task in tasks if task.kind == "tool_call")
