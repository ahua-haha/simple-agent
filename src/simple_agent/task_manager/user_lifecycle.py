"""User task lifecycle."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, cast

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import AssistantMessage, TextContent, UserMessage

from simple_agent.run_log import runtime_logger
from simple_agent.task_manager.base_lifecycle import (
    BaseTaskLifecycle,
    SessionState,
    TaskLifecycleError,
    USER_TASK_COMPACT_SYSTEM_PROMPT,
    USER_TASK_SYSTEM_PROMPT,
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
            return await self.run_compact_one_turn(
                agent_process=agent_process,
                cancel_event=cancel_event,
            )

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
        turn_result = await self.run_agent_turn(
            agent_process=agent_process,
            system_prompt=USER_TASK_SYSTEM_PROMPT,
            user_instruction_message=user_instruction_message,
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
        turn_result = await self.run_agent_turn(
            agent_process=agent_process,
            system_prompt=USER_TASK_COMPACT_SYSTEM_PROMPT,
            user_instruction_message=user_instruction_message,
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
            compacted_messages = format_messages_from_user_task(task)
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

    def create_compact_one_turn_tools(self) -> list[AgentTool]:
        record_tool = AgentTool(
            name="record_compacted_tool_call_task",
            description="Record one useful tool-call task id for the compacted user task.",
            parameters={
                "type": "object",
                "properties": {
                    "tool_call_task_id": {
                        "type": "integer",
                        "description": "Task id of the useful tool_call task to keep in compacted context.",
                    },
                },
                "required": ["tool_call_task_id"],
            },
        )

        async def record_execute(tool_call_id, params, cancel_event=None, on_update=None):
            self.record_compacted_tool_call_task(
                tool_call_task_id=params["tool_call_task_id"],
            )
            return AgentToolResult(content=[TextContent(text="recorded compacted tool call task")])

        record_tool.execute = record_execute
        return [record_tool]

    def record_compacted_tool_call_task(self, *, tool_call_task_id: int) -> None:
        if tool_call_task_id not in self.task.compacted_tool_call_task_ids:
            self.task.compacted_tool_call_task_ids.append(tool_call_task_id)
            self.task.touch()


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


def format_messages_from_user_task(user_task: UserTask) -> list[AssistantMessage]:
    # TODO: format compacted messages from the user task result and selected
    # compacted tool-call task ids.
    text = (
        f"Compacted user task: {user_task.result or user_task.title}\n"
        f"Useful tool call tasks: {user_task.compacted_tool_call_task_ids}"
    )
    return [AssistantMessage(role="assistant", content=[TextContent(text=text)])]
