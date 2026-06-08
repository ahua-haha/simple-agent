"""Task lifecycle handlers.

Task data classes stay as records. Lifecycle classes own mutations,
agent-facing tools, turn instructions, and direct database sync.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, cast

from pi.agent import AgentTool, AgentToolResult
from pi.agent.types import AgentMessage
from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage

from simple_agent.message_store import MessageEntry
from simple_agent.run_log import runtime_logger
from simple_agent.task_manager.models import ManagedTask, TodoTask, ToolCallTask, UserTask
from simple_agent.task_manager.review import TaskTreeReviewRenderer, ToolCallReview
from simple_agent.tool.common_tools import create_all_coding_tools

if TYPE_CHECKING:
    from simple_agent.db.db import Database
    from simple_agent.process.agent_process import AgentProcess
    from sqlmodel import Session as SqlSession


class TaskLifecycleError(RuntimeError):
    """Raised when a task lifecycle cannot complete an operation."""


USER_TASK_SYSTEM_PROMPT = """You are a helpful coding agent.

Be concise, practical, and honest about uncertainty. Use available tools
when they are needed, and explain outcomes clearly.
"""

USER_TASK_COMPACT_SYSTEM_PROMPT = """Compact the finished user task into one compacted user task.
Use only the compact tools. Set the compacted user task result, record useful
tool-call log IDs, then finish the compacted user task."""


@dataclass
class SessionState:
    messages: list[MessageEntry]
    session_id: str | None = None
    database: Database | None = None
    next_message_id: int = 1
    next_tool_call_log_id: int = 0
    next_task_id_to_allocate: int | None = None
    next_task_id_to_run: int | None = None
    next_task: ManagedTask | None = None

    def allocate_message_id(self) -> int:
        message_id = self.next_message_id
        self.next_message_id += 1
        return message_id

    def allocate_task_id(self) -> int:
        if self.next_task_id_to_allocate is None:
            raise TaskLifecycleError("Session state is missing task allocation state")
        task_id = self.next_task_id_to_allocate
        self.next_task_id_to_allocate += 1
        return task_id

    def allocate_tool_call_log_id(self) -> int:
        tool_call_log_id = self.next_tool_call_log_id
        self.next_tool_call_log_id += 1
        return tool_call_log_id

    def message_values(self) -> list[AgentMessage]:
        return [entry.message for entry in self.messages]

    def append_message(self, message: AgentMessage) -> MessageEntry:
        entry = MessageEntry(id=self.allocate_message_id(), message=message)
        self.messages.append(entry)
        return entry

    def append_messages(self, messages: list[AgentMessage | MessageEntry]) -> list[MessageEntry]:
        entries: list[MessageEntry] = []
        for message in messages:
            if isinstance(message, MessageEntry):
                entry = message
                self.messages.append(entry)
                self.next_message_id = max(self.next_message_id, entry.id + 1)
            else:
                entry = self.append_message(message)
            entries.append(entry)
        return entries

    def replace_message_range(
        self,
        *,
        start_message_id: int,
        end_message_id: int,
        replacement_messages: list[AgentMessage],
    ) -> list[MessageEntry]:
        start_index = self._message_index(start_message_id)
        end_index = self._message_index(end_message_id)
        if end_index < start_index:
            raise RuntimeError("Compact end message is before compact start message")
        replacement_entries = [
            MessageEntry(id=self.allocate_message_id(), message=message)
            for message in replacement_messages
        ]
        self.messages = [
            *self.messages[:start_index],
            *replacement_entries,
            *self.messages[end_index + 1:],
        ]
        return replacement_entries

    def create_tool_call_record_task_entries(
        self,
        *,
        assistant_message: AssistantMessage,
        tool_result_messages: list[ToolResultMessage],
        parent_task: ManagedTask,
    ) -> tuple[list[tuple[int, ToolCall | None, ToolResultMessage]], list[ToolCallTask]]:
        tool_call_records: list[tuple[int, ToolCall | None, ToolResultMessage]] = []
        tool_call_tasks: list[ToolCallTask] = []
        for tool_result_message in tool_result_messages:
            log_id = self.allocate_tool_call_log_id()
            tool_call = _tool_call_for_result(
                assistant_message=assistant_message,
                tool_result_message=tool_result_message,
            )
            tool_call_records.append((log_id, tool_call, tool_result_message))
            tool_call_tasks.append(
                ToolCallTask(
                    id=self.allocate_task_id(),
                    title=f"Tool call {log_id}",
                    status="done",
                    parent_id=parent_task.id,
                    tool_call_log_id=log_id,
                )
            )
        return tool_call_records, tool_call_tasks

    def create_database_session(self) -> SqlSession:
        return self._require_database().create_session()

    def append_messages_to_database(
        self,
        *,
        messages: list[MessageEntry],
        session: SqlSession,
    ) -> None:
        database = self._require_database()
        session_id = self._require_session_id()
        for message in messages:
            database.insert_runner_message(
                session_id,
                message.message,
                id=message.id,
                session=session,
            )

    def append_tool_calls_to_database(
        self,
        *,
        tool_calls: list[tuple[int, ToolCall | None, ToolResultMessage]],
        session: SqlSession,
    ) -> None:
        database = self._require_database()
        session_id = self._require_session_id()
        for log_id, tool_call, tool_result in tool_calls:
            database.insert_runner_tool_call(
                id=log_id,
                session_id=session_id,
                tool_call_id=tool_result.tool_call_id,
                tool_name=tool_result.tool_name,
                tool_call_json=tool_call.model_dump_json() if tool_call is not None else "null",
                tool_result_json=tool_result.model_dump_json(),
                session=session,
            )

    def append_tasks_to_database(
        self,
        *,
        tasks: list[ManagedTask],
        session: SqlSession,
    ) -> None:
        database = self._require_database()
        for task in tasks:
            database.upsert_managed_task(task, session=session)

    def replace_messages_in_database(self, *, session: SqlSession) -> None:
        database = self._require_database()
        session_id = self._require_session_id()
        database.replace_runner_messages(
            session_id,
            [entry.message for entry in self.messages],
            ids=[entry.id for entry in self.messages],
            session=session,
        )

    def replace_task_tree_in_database(
        self,
        *,
        task: ManagedTask,
        session: SqlSession,
    ) -> None:
        database = self._require_database()
        database.replace_managed_task_tree(task, session=session)

    def set_next_task(self, task: ManagedTask | None, *, keep_instance: bool = False) -> None:
        self.next_task_id_to_run = task.id if task is not None else None
        self.next_task = task if task is not None and keep_instance else None

    def _message_index(self, message_id: int) -> int:
        for index, entry in enumerate(self.messages):
            if entry.id == message_id:
                return index
        raise RuntimeError(f"Could not find message id {message_id}")

    def _require_database(self) -> Database:
        if self.database is None:
            raise TaskLifecycleError("Session state is missing database")
        return self.database

    def _require_session_id(self) -> str:
        if self.session_id is None:
            raise TaskLifecycleError("Session state is missing session id")
        return self.session_id


class BaseTaskLifecycle:
    _session_state: SessionState
    current_assistant_message_id: int | None

    def clear_data(self) -> None:
        self.current_assistant_message_id = None

    def set_data(self, session_state: SessionState) -> None:
        self._session_state = session_state
        self.current_assistant_message_id: int | None = None
        raise NotImplementedError(f"{type(self).__name__}.set_data is not implemented")

    async def run(
        self,
        *,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None = None,
    ) -> SessionState:
        raise NotImplementedError(f"{type(self).__name__}.run is not implemented")


class UserTaskLifecycle(BaseTaskLifecycle):
    def set_data(self, session_state: SessionState) -> None:
        from simple_agent.task_manager.task_builder import NextTaskBuilder

        self._session_state = session_state
        self.current_assistant_message_id = None
        task = self._session_state.next_task
        if task is None:
            raise TaskLifecycleError("Session state has no next task")
        if task.kind != "user_task":
            raise TaskLifecycleError("Active lifecycle task is not a user task")
        self.task = cast(UserTask, task)
        self._task_builder = NextTaskBuilder(
            self._session_state,
            enabled_task_kinds=["todo", "repo_memory"],
            current_assistant_message_id=lambda: self.current_assistant_message_id,
        )
        self._compacted_tool_calls: list[ToolCallTask] = []
        self._compacted_user_task_finished = False

    def clear_data(self) -> None:
        super().clear_data()
        self.task = None
        self._task_builder = None

    def instruction_text(self) -> str:
        builder_instruction = self._task_builder.instruction_text()
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

    def create_todo_task(
        self,
        *,
        task_id: int | None = None,
        title: str,
    ) -> TodoTask:
        todo = TodoTask(
            id=task_id if task_id is not None else self._session_state.allocate_task_id(),
            title=title,
            parent_id=self.task.id,
            start_message_id=self.current_assistant_message_id,
        )
        self.task.children.append(todo)
        self.task.touch()
        self._session_state.set_next_task(todo, keep_instance=True)
        return todo

    def finish_task(self, *, result: str | None = None) -> UserTask:
        self.task.status = "done"
        self.task.result = result
        self.task.end_message_id = self.current_assistant_message_id
        self.task.touch()
        self._session_state.set_next_task(self.task, keep_instance=True)
        return self.task

    def todo_status_text(self) -> str:
        return todo_status_text(self.task)

    def create_tools(self) -> list[AgentTool]:
        return [
            *self._task_builder.create_tools(),
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
            return await self.handle_compact(
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
        if _assistant_has_tool_calls(assistant_message):
            self.current_assistant_message_id = assistant_entry.id
            try:
                tool_results = await agent_process.run_tool_calls_step(
                    tools=tools,
                    assistant_message=assistant_message,
                    cancel_event=cancel_event,
                )
            finally:
                self.current_assistant_message_id = None
            tool_call_records, tool_call_tasks = self._session_state.create_tool_call_record_task_entries(
                assistant_message=assistant_message,
                tool_result_messages=tool_results,
                parent_task=task,
            )
            task.children.extend(tool_call_tasks)
            if tool_call_tasks:
                task.touch()

        tool_result_entries = [
            MessageEntry(id=self._session_state.allocate_message_id(), message=tool_result)
            for tool_result in tool_results
        ]
        new_messages = [assistant_entry, *tool_result_entries]
        self._session_state.append_messages(new_messages)

        if not _assistant_has_tool_calls(assistant_message):
            self.current_assistant_message_id = assistant_entry.id
            try:
                if task.status != "done":
                    self.finish_task()
                if self.should_compact_after_turn():
                    self._session_state.set_next_task(task, keep_instance=True)
                else:
                    self._session_state.next_task_id_to_run = task.parent_id
                    self._session_state.next_task = None
            finally:
                self.current_assistant_message_id = None
        else:
            has_child_task = self._session_state.next_task is not None and self._session_state.next_task is not task
            if not has_child_task and task.status == "active":
                self._session_state.set_next_task(task, keep_instance=True)

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
        return self._session_state

    def create_create_todo_tool(self) -> AgentTool:
        tool = AgentTool(
            name="create_todo",
            description=(
                "Create the next todo item for the current session task list. "
                "Use for complex tasks with 3+ steps or when the user provides "
                "multiple tasks. Create items in priority order. Only one todo "
                "may be active at a time."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short content for the next coherent unit of work.",
                    },
                },
                "required": ["title"],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            self.create_todo_task(
                title=params["title"],
            )
            return AgentToolResult(content=[TextContent(text=todo_status_text(self.task))])

        tool.execute = execute
        return tool

    # ------------------------------------------------------------------
    # User-task compaction phase
    # ------------------------------------------------------------------

    def compaction_instruction_text(self, *, tool_calls: Mapping[int, ToolCallReview]) -> str:
        task_view = TaskTreeReviewRenderer(
            format="tree",
            depth=None,
            tool_calls=tool_calls,
        ).render(self.task)
        return (
            "Runtime instruction for compacting phase:\n"
            "- Complete the compacted user task information first: define the task result.\n"
            "- Record every must-include tool call based on the compacted task result to avoid context loss.\n"
            "- Use only compact tools: set the compacted user task result, record must-include tool calls, then finish it.\n"
            "\n"
            "Task view to compact:\n"
            f"{task_view.text}"
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

        self._compacted_tool_calls = []
        self._compacted_user_task_finished = False

        compact_instruction = self.compaction_instruction_text(tool_calls={})
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
        self.task.children = list(self._compacted_tool_calls)
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
            title=f"Tool call {tool_call_log_id}",
            status="done",
            parent_id=self.task.id,
            tool_call_log_id=tool_call_log_id,
        )
        self._compacted_tool_calls.append(tool_call_task)

    def finish_compacted_user_task(self) -> UserTask:
        if self.task.result is None:
            raise TaskLifecycleError("No compacted user task result")
        self._compacted_user_task_finished = True
        self.task.touch()
        return self.task

    def compaction_result(self) -> tuple[int, int, list[Any]]:
        if self.task.result is None:
            raise TaskLifecycleError("No compacted user task result")
        if not self._compacted_user_task_finished:
            raise TaskLifecycleError("Compacted user task is not finished")
        if self.task.start_message_id is None or self.task.end_message_id is None:
            raise TaskLifecycleError("Compact scope is missing message boundaries")
        tool_refs = [
            child.tool_call_log_id
            for child in self._compacted_tool_calls
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


class TodoTaskLifecycle(BaseTaskLifecycle):
    def set_data(self, session_state: SessionState) -> None:
        self._session_state = session_state
        self.current_assistant_message_id = None
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
        task.end_message_id = self.current_assistant_message_id
        task.touch()
        self._session_state.next_task_id_to_run = task.parent_id
        self._session_state.next_task = None
        return task

    def error_task(self, *, error: str) -> TodoTask:
        task = self._require_todo_task_data()
        task.status = "error"
        task.error = error
        task.end_message_id = self.current_assistant_message_id
        task.touch()
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
        if _assistant_has_tool_calls(assistant_message):
            self.current_assistant_message_id = assistant_entry.id
            try:
                tool_results = await agent_process.run_tool_calls_step(
                    tools=tools,
                    assistant_message=assistant_message,
                    cancel_event=cancel_event,
                )
            finally:
                self.current_assistant_message_id = None
            tool_call_records, tool_call_tasks = self._session_state.create_tool_call_record_task_entries(
                assistant_message=assistant_message,
                tool_result_messages=tool_results,
                parent_task=task,
            )
            task.children.extend(tool_call_tasks)
            if tool_call_tasks:
                task.touch()

        tool_result_entries = [
            MessageEntry(id=self._session_state.allocate_message_id(), message=tool_result)
            for tool_result in tool_results
        ]
        new_messages = [assistant_entry, *tool_result_entries]
        self._session_state.append_messages(new_messages)

        if not _assistant_has_tool_calls(assistant_message):
            self.current_assistant_message_id = assistant_entry.id
            try:
                if task.status == "active":
                    self.finish_task()
            finally:
                self.current_assistant_message_id = None
        elif task.status == "active":
            self._session_state.set_next_task(task, keep_instance=True)

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


def _next_task_action_text(session_state: SessionState) -> str:
    return "next_task" if session_state.next_task_id_to_run is not None else "wait_user_input"


def _assistant_has_tool_calls(message: AssistantMessage) -> bool:
    return any(isinstance(content, ToolCall) for content in message.content)


def _assistant_is_error(message: AssistantMessage) -> bool:
    return message.stop_reason == "error" or bool(message.error_message)


def _tool_call_for_result(
    *,
    assistant_message: AssistantMessage,
    tool_result_message: ToolResultMessage,
) -> ToolCall | None:
    for content in assistant_message.content:
        if isinstance(content, ToolCall) and content.id == tool_result_message.tool_call_id:
            return content
    return None
