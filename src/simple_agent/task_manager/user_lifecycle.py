"""User task lifecycle."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage

from simple_agent.message_store import MessageEntry
from simple_agent.run_log import runtime_logger
from simple_agent.task_manager.base_lifecycle import (
    BaseTaskLifecycle,
    SessionState,
    TaskLifecycleError,
    USER_TASK_COMPACT_SYSTEM_PROMPT,
    USER_TASK_SYSTEM_PROMPT,
    _assistant_has_tool_calls,
    _assistant_is_error,
    _next_task_action_text,
)
from simple_agent.task_manager.models import ManagedTask, TodoTask, ToolCallTask, UserTask
from simple_agent.task_manager.review import TaskTreeRenderer
from simple_agent.tool.common_tools import create_all_coding_tools

if TYPE_CHECKING:
    from simple_agent.process.agent_process import AgentProcess


class UserTaskLifecycle(BaseTaskLifecycle):
    task: UserTask | None

    def set_data(self, session_state: SessionState) -> None:
        self._session_state = session_state
        self.created_task = None
        self.finished_task = None
        task = self._session_state.next_task
        if task is None:
            raise TaskLifecycleError("Session state has no next task")
        if task.kind != "user_task":
            raise TaskLifecycleError("Active lifecycle task is not a user task")
        self.task = cast(UserTask, task)

    def clear_data(self) -> None:
        super().clear_data()

    def instruction_text(self) -> str:
        builder_instruction = self.next_task_instruction_text(enabled_task_kinds=["todo", "repo_memory"])
        if _count_user_task_tool_calls_after_latest_todo(self.task) > 5:
            return (
                "Runtime instruction for this turn:\n"
                "- More than 5 tool calls have run since the previous todo.\n"
                "- Stop and create the next enabled task before doing more work.\n"
                "- Use any enabled task kind that best captures the next coherent unit of work.\n"
                "- Keep the next task small and atomic so it can be completed cleanly.\n\n"
                f"{builder_instruction}"
            )
        return (
            "Runtime instruction for this turn:\n"
            "- Determine whether the user task is complex before doing more work.\n"
            "- If it is complex or long-running, decompose it into the next enabled task first.\n"
            "- Choose the enabled task kind that best moves the user task forward.\n"
            "- Keep each created task small, atomic, and directly tied to finishing the user task.\n"
            "- If it is simple, answer directly or use the needed tools.\n\n"
            f"{builder_instruction}"
        )

    def finish_task(self, *, result: str | None = None) -> UserTask:
        self.task.status = "done"
        self.task.result = result
        self.task.touch()
        self.finished_task = self.task
        return self.task

    def todo_status_text(self) -> str:
        return todo_status_text(self.task)

    def create_tools(self) -> list[AgentTool]:
        return [
            *self.create_next_task_tools(enabled_task_kinds=["todo", "repo_memory"]),
            self.create_finish_user_task_tool(),
            *create_all_coding_tools("."),
        ]

    async def run(
        self,
        *,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None = None,
    ) -> SessionState:
        if self.task.status == "done":
            self._session_state.next_task_id_to_run = self.task.parent_id
            self._session_state.next_task = None
            return self._session_state

        return await self.run_one_turn(
            agent_process=agent_process,
            cancel_event=cancel_event,
        )

    def should_compact_after_turn(self) -> bool:
        # TODO: implement compact trigger logic for completed user tasks.
        return False

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
        assistant_message = await agent_process.call_llm_step(
            system_prompt=USER_TASK_SYSTEM_PROMPT,
            messages=[*self._session_state.message_values(), user_instruction_message],
            tools=tools,
            cancel_event=cancel_event,
        )
        if _assistant_is_error(assistant_message):
            raise TaskLifecycleError(
                assistant_message.error_message or "assistant response stopped with error"
            )

        assistant_entry = MessageEntry(
            id=self._session_state.allocate_message_id(),
            message=assistant_message,
        )
        tool_results: list[ToolResultMessage] = []
        tool_call_records: list[tuple[int, ToolCall | None, ToolResultMessage]] = []
        tool_call_tasks: list[ToolCallTask] = []
        has_tool_call = _assistant_has_tool_calls(assistant_message)
        if has_tool_call:
            tool_results = await agent_process.run_tool_calls_step(
                tools=tools,
                assistant_message=assistant_message,
                cancel_event=cancel_event,
            )
            tool_call_records, tool_call_tasks = self._session_state.create_tool_call_record_task_entries(
                assistant_message=assistant_message,
                tool_result_messages=tool_results,
                parent_task=task,
            )

        tool_result_entries = [
            MessageEntry(id=self._session_state.allocate_message_id(), message=tool_result)
            for tool_result in tool_results
        ]
        new_messages = [assistant_entry, *tool_result_entries]
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
                self.stamp_finished_task(end_message_id=assistant_entry.id)
                if self.should_compact_after_turn():
                    self.set_next_task(task.id, task)
                    return
                self.set_next_task(task.parent_id, None)
                return

            if has_tool_call:
                self.set_next_task(task.id, task)
                return

        route_after_turn()

        self.stamp_finished_task(end_message_id=assistant_entry.id)

        runtime_logger.log_handle_running(
            session_id=self._session_state._require_session_id(),
            messages=context_messages,
            user_instruction_message=user_instruction_message,
            assistant_message_id=assistant_entry.id,
            assistant_message=assistant_message,
            tool_result_entries=tool_result_entries,
            next_action=_next_task_action_text(self._session_state),
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
        return (
            "Runtime instruction for compacting phase:\n"
            "- Complete the compacted user task information first: define the task result.\n"
            "- Record every must-include tool call based on the compacted task result to avoid context loss.\n"
            "- Use only compact tools: set the compacted user task result, record must-include tool calls, then finish it.\n"
            "\n"
            "Task view to compact:\n"
            f"{task_view}"
        )

    async def handle_compact(
        self,
        *,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None = None,
    ) -> SessionState:
        if not self.task.children:
            if cancel_event is not None:
                cancel_event.clear()
            self._session_state.next_task_id_to_run = self.task.parent_id
            self._session_state.next_task = None
            return self._session_state

        self.task.compacted_tool_calls = []
        self.task.compacted_user_task_finished = False

        compact_instruction = self.compaction_instruction_text()
        compact_messages = [
            *self._session_state.message_values(),
            UserMessage(
                content=[TextContent(text=compact_instruction)],
                timestamp=int(time.time() * 1000),
            ),
        ]
        compact_tools = self.create_compact_tools()
        while True:
            assistant_message = await agent_process.call_llm_step(
                system_prompt=USER_TASK_COMPACT_SYSTEM_PROMPT,
                messages=compact_messages,
                tools=compact_tools,
                cancel_event=cancel_event,
            )
            compact_messages.append(assistant_message)
            if _assistant_is_error(assistant_message):
                raise TaskLifecycleError(
                    assistant_message.error_message or "assistant response stopped with error"
                )
            if not _assistant_has_tool_calls(assistant_message):
                break
            tool_results = await agent_process.run_tool_calls_step(
                tools=compact_tools,
                assistant_message=assistant_message,
                cancel_event=cancel_event,
            )
            compact_messages.extend(tool_results)

        start_message_id, end_message_id, compacted_messages = self.compaction_result()
        replacement_entries = self._session_state.replace_message_range(
            start_message_id=start_message_id,
            end_message_id=end_message_id,
            replacement_messages=compacted_messages,
        )

        if cancel_event is not None:
            cancel_event.clear()

        # TODO: determine next action after compacting.
        self.task.children = list(self.task.compacted_tool_calls)
        self.task.touch()
        self._session_state.next_task_id_to_run = self.task.parent_id
        self._session_state.next_task = None
        runtime_logger.log_handle_compact_result(
            session_id=self._session_state._require_session_id(),
            compact_messages=compact_messages,
            start_message_id=start_message_id,
            end_message_id=end_message_id,
            compacted_messages=compacted_messages,
            replacement_messages=replacement_entries,
            next_action=_next_task_action_text(self._session_state),
        )
        with self._session_state.create_database_session() as session:
            self._session_state.replace_messages_in_database(session=session)
            self._session_state.replace_task_tree_in_database(task=self.task, session=session)
            session.commit()
        return self._session_state

    def create_compact_tools(self) -> list[AgentTool]:
        create_tool = AgentTool(
            name="create_compacted_user_task",
            description="Set the compacted user task result with a concise summary.",
            parameters={
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Compacted user task summary"},
                },
                "required": ["description"],
            },
        )

        async def create_execute(tool_call_id, params, cancel_event=None, on_update=None):
            task = self.create_compacted_user_task(
                description=params["description"],
            )
            return AgentToolResult(content=[TextContent(text=f"created compacted user task {task.id}")])

        create_tool.execute = create_execute

        record_tool = AgentTool(
            name="record_compacted_tool_call",
            description="Keep one useful runner tool-call log ID in the compacted user task.",
            parameters={
                "type": "object",
                "properties": {
                    "tool_call_log_id": {"type": "integer", "description": "Runner tool-call log ID"},
                },
                "required": ["tool_call_log_id"],
            },
        )

        async def record_execute(tool_call_id, params, cancel_event=None, on_update=None):
            self.record_compacted_tool_call(
                tool_call_log_id=params["tool_call_log_id"],
            )
            return AgentToolResult(content=[TextContent(text="recorded compacted tool call")])

        record_tool.execute = record_execute

        finish_tool = AgentTool(
            name="finish_compacted_user_task",
            description="Finish the compacted user task after selecting useful tool calls.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        async def finish_execute(tool_call_id, params, cancel_event=None, on_update=None):
            task = self.finish_compacted_user_task()
            return AgentToolResult(content=[TextContent(text=f"finished compacted user task {task.id}")])

        finish_tool.execute = finish_execute
        return [create_tool, record_tool, finish_tool]

    def create_compacted_user_task(self, *, description: str) -> UserTask:
        self.task.result = description
        self.task.touch()
        return self.task

    def record_compacted_tool_call(self, *, tool_call_log_id: int) -> None:
        tool_call_task = ToolCallTask(
            id=self._session_state.allocate_task_id(),
            status="done",
            parent_id=self.task.id,
            tool_call_log_id=tool_call_log_id,
        )
        self.task.compacted_tool_calls.append(tool_call_task)

    def finish_compacted_user_task(self) -> UserTask:
        if self.task.result is None:
            raise TaskLifecycleError("No compacted user task result")
        self.task.compacted_user_task_finished = True
        self.task.touch()
        return self.task

    def compaction_result(self) -> tuple[int, int, list[Any]]:
        if self.task.result is None:
            raise TaskLifecycleError("No compacted user task result")
        if not self.task.compacted_user_task_finished:
            raise TaskLifecycleError("Compacted user task is not finished")
        if self.task.start_message_id is None or self.task.end_message_id is None:
            raise TaskLifecycleError("Compact scope is missing message boundaries")
        tool_refs = [
            child.tool_call_log_id
            for child in self.task.compacted_tool_calls
            if child.tool_call_log_id is not None
        ]
        text = (
            f"Compacted user task: {self.task.result or self.task.title}\n"
            f"Useful tool calls: {tool_refs}"
        )
        return (
            self.task.start_message_id,
            self.task.end_message_id,
            [AssistantMessage(role="assistant", content=[TextContent(text=text)])],
        )

    def create_finish_user_task_tool(self) -> AgentTool:
        tool = AgentTool(
            name="finish_user_task",
            description=(
                "Mark the current user task as completed. Call when the user's "
                "request is fully satisfied and no todo is active."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "Optional concise result for this user task"},
                },
                "required": [],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            task = self.finish_task(
                result=params.get("result"),
            )
            return AgentToolResult(content=[TextContent(text=f"User task finished: {task.result or task.title}")])

        tool.execute = execute
        return tool


def todo_status_text(user_task: UserTask) -> str:
    todos = [child for child in user_task.children if child.kind == "todo"]
    if not todos:
        return "Todos: []"

    lines = ["Todos:"]
    for todo in todos:
        line = f"- [{todo.status}] {todo.title}"
        if todo.result:
            line += f" result={todo.result}"
        if todo.error:
            line += f" error={todo.error}"
        lines.append(line)
    return "\n".join(lines)


def _count_user_task_tool_calls_after_latest_todo(user_task: UserTask) -> int:
    latest_todo_index = -1
    for index, child in enumerate(user_task.children):
        if child.kind == "todo":
            latest_todo_index = index
    return _count_tool_calls(user_task.children[latest_todo_index + 1:])


def _count_tool_calls(tasks: list[ManagedTask]) -> int:
    return sum(1 for task in tasks if task.kind == "tool_call")
