"""Task lifecycle handlers.

Task data classes stay as records. Lifecycle classes own mutations,
agent-facing tools, turn instructions, and direct database sync.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage

from simple_agent.message_store import MessageEntry
from simple_agent.task_manager.models import ManagedTask, TodoTask, ToolCallTask, UserTask
from simple_agent.task_manager.review import TaskTreeReviewRenderer, ToolCallReview
from simple_agent.token_estimation import estimate_messages_tokens
from simple_agent.tool.common_tools import create_all_coding_tools

if TYPE_CHECKING:
    from simple_agent.process.agent_process import AgentProcess


class TaskLifecycleError(RuntimeError):
    """Raised when a task lifecycle cannot complete an operation."""


TaskLifecycleAction = Literal["normal_run", "compact", "handle_error", "wait_user_input"]

USER_TASK_SYSTEM_PROMPT = """You are a helpful coding agent.

Be concise, practical, and honest about uncertainty. Use available tools
when they are needed, and explain outcomes clearly.
"""

USER_TASK_COMPACT_SYSTEM_PROMPT = """Compact the finished user task into one compacted user task.
Use only the compact tools. Set the compacted user task result, record useful
tool-call log IDs, then finish the compacted user task."""


@dataclass(frozen=True)
class UserTaskRunTurnResult:
    user_instruction_message: UserMessage
    assistant_message: AssistantMessage
    new_messages: list[MessageEntry]
    tool_call_records: list[tuple[int, ToolCall | None, ToolResultMessage]]


@dataclass(frozen=True)
class TaskLifecycleRunResult:
    next_action: TaskLifecycleAction
    next_task: ManagedTask | None
    error: Exception | AssistantMessage | str | None = None
    new_messages: list[MessageEntry] | None = None
    tool_call_records: list[tuple[int, ToolCall | None, ToolResultMessage]] | None = None


@dataclass(frozen=True)
class UserTaskCompactRunResult:
    compact_messages: list[Any]
    start_message_id: int
    end_message_id: int
    compacted_messages: list[Any]


class BaseTaskLifecycle:
    def __init__(self, *, allocate_task_id: Callable[[], int] | None = None):
        self._allocate_task_id = allocate_task_id
        self.current_assistant_message_id: int | None = None
        self.next_task: ManagedTask | None = None
        self.messages: list[MessageEntry] = []
        self.next_message_id: int = 1
        self.next_tool_call_log_id: int = 0

    def allocate_task_id(self) -> int:
        if self._allocate_task_id is None:
            raise TaskLifecycleError("Task lifecycle needs an ID allocator")
        return self._allocate_task_id()

    def consume_next_task(self) -> ManagedTask | None:
        next_task = self.next_task
        self.next_task = None
        return next_task

    def load_messages(self, messages: list[MessageEntry], *, next_message_id: int) -> None:
        self.messages = list(messages)
        self.next_message_id = next_message_id

    def load_tool_call_log_id(self, next_tool_call_log_id: int) -> None:
        self.next_tool_call_log_id = next_tool_call_log_id

    def allocate_message_id(self) -> int:
        message_id = self.next_message_id
        self.next_message_id += 1
        return message_id

    def allocate_tool_call_log_id(self) -> int:
        tool_call_log_id = self.next_tool_call_log_id
        self.next_tool_call_log_id += 1
        return tool_call_log_id

    def message_values(self) -> list[Any]:
        return [entry.message for entry in self.messages]

    def append_message(self, message: Any) -> MessageEntry:
        entry = MessageEntry(id=self.allocate_message_id(), message=message)
        self.messages.append(entry)
        return entry

    def append_messages(self, messages: list[MessageEntry]) -> None:
        self.messages.extend(messages)

    def replace_message_range(
        self,
        *,
        start_message_id: int,
        end_message_id: int,
        replacement_messages: list[Any],
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

    def sync_messages(self, session_id: str, database: Any, messages: list[MessageEntry], *, session: Any) -> None:
        for pending in messages:
            database.insert_runner_message(
                session_id,
                pending.message,
                id=pending.id,
                session=session,
            )

    def sync_replaced_messages(
        self,
        session_id: str,
        database: Any,
        messages: list[MessageEntry],
        *,
        session: Any,
    ) -> None:
        database.replace_runner_messages(
            session_id,
            [entry.message for entry in messages],
            ids=[entry.id for entry in messages],
            session=session,
        )

    def sync_tool_calls(
        self,
        session_id: str,
        database: Any,
        tool_calls: list[tuple[int, ToolCall | None, ToolResultMessage]],
        *,
        session: Any,
    ) -> None:
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

    def create_tool_call_record_task_entries(
        self,
        *,
        assistant_message: AssistantMessage,
        tool_result_messages: list[ToolResultMessage],
    ) -> tuple[list[tuple[int, ToolCall | None, ToolResultMessage]], list[ToolCallTask]]:
        task = self.task
        tool_call_records = []
        tool_call_tasks = []
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
                    parent_id=task.id,
                    tool_call_log_id=log_id,
                )
            )
        return tool_call_records, tool_call_tasks

    def _message_index(self, message_id: int) -> int:
        for index, entry in enumerate(self.messages):
            if entry.id == message_id:
                return index
        raise RuntimeError(f"Could not find message id {message_id}")


class UserTaskLifecycle(BaseTaskLifecycle):
    def __init__(self, task: UserTask, *, allocate_task_id: Callable[[], int] | None = None):
        super().__init__(allocate_task_id=allocate_task_id)
        self.task = task
        self._compacting = False
        self._compacted_tool_calls: list[ToolCallTask] = []
        self._compacted_user_task_finished = False

    def instruction_text(self) -> str:
        if _count_user_task_tool_calls_after_latest_todo(self.task) > 5:
            return (
                "Runtime instruction for this turn:\n"
                "- More than 5 tool calls have run since the previous todo.\n"
                "- Stop and create a small atomic todo before doing more work.\n"
                "- The todo should describe only the next coherent unit of work."
            )
        return (
            "Runtime instruction for this turn:\n"
            "- Determine whether the user task is complex before doing more work.\n"
            "- If it is complex or long-running, create the next small atomic todo first.\n"
            "- If it is simple, answer directly or use the needed tools."
        )

    def create_todo_task(
        self,
        *,
        task_id: int | None = None,
        title: str,
    ) -> TodoTask:
        todo = TodoTask(
            id=task_id if task_id is not None else self.allocate_task_id(),
            title=title,
            parent_id=self.task.id,
            start_message_id=self.current_assistant_message_id,
        )
        self.task.children.append(todo)
        self.task.touch()
        self.next_task = todo
        return todo

    def finish_task(self, *, result: str | None = None) -> UserTask:
        self.task.status = "done"
        self.task.result = result
        self.task.end_message_id = self.current_assistant_message_id
        self.task.touch()
        self.next_task = None
        return self.task

    def todo_status_text(self) -> str:
        return todo_status_text(self.task)

    def create_tools(self) -> list[AgentTool]:
        return [
            self.create_create_todo_tool(),
            self.create_finish_user_task_tool(),
            *create_all_coding_tools("."),
        ]

    async def run(
        self,
        *,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None = None,
        context_token_threshold: int,
        tool_call_threshold: int,
    ) -> TaskLifecycleRunResult:
        if self.task.status == "done":
            return await self._run_compact(agent_process=agent_process, cancel_event=cancel_event)

        turn = await self.run_one_turn(
            agent_process=agent_process,
            cancel_event=cancel_event,
        )
        if _assistant_is_error(turn.assistant_message):
            return TaskLifecycleRunResult(
                next_action="handle_error",
                next_task=self.task,
                error=turn.assistant_message,
                new_messages=turn.new_messages,
                tool_call_records=turn.tool_call_records,
            )

        if self.task.status == "done":
            result = TaskLifecycleRunResult(
                next_action="compact",
                next_task=None,
                new_messages=turn.new_messages,
                tool_call_records=turn.tool_call_records,
            )
        elif not _assistant_has_tool_calls(turn.assistant_message):
            assistant_message_id = turn.new_messages[0].id if turn.new_messages else None
            self.current_assistant_message_id = assistant_message_id
            try:
                self.finish_task()
            finally:
                self.current_assistant_message_id = None
            result = TaskLifecycleRunResult(
                next_action="compact",
                next_task=None,
                new_messages=turn.new_messages,
                tool_call_records=turn.tool_call_records,
            )
        else:
            result = TaskLifecycleRunResult(
                next_action=self._next_action_after_thresholds(
                    context_token_threshold=context_token_threshold,
                    tool_call_threshold=tool_call_threshold,
                    cancel_event=cancel_event,
                ),
                next_task=self._next_task_after_turn(),
                new_messages=turn.new_messages,
                tool_call_records=turn.tool_call_records,
            )

        # TODO: sync normal run messages, tool calls, and task data to database here.
        return result

    async def _run_compact(
        self,
        *,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None,
    ) -> TaskLifecycleRunResult:
        if not self.begin_compaction():
            if cancel_event is not None:
                cancel_event.clear()
            return TaskLifecycleRunResult(next_action="wait_user_input", next_task=None)

        try:
            result = await self.handle_compact(
                agent_process=agent_process,
                cancel_event=cancel_event,
            )
        except Exception as exc:
            return TaskLifecycleRunResult(next_action="handle_error", next_task=self.task, error=exc)

        self.replace_message_range(
            start_message_id=result.start_message_id,
            end_message_id=result.end_message_id,
            replacement_messages=result.compacted_messages,
        )
        if cancel_event is not None:
            cancel_event.clear()

        # TODO: sync compacted messages and compacted task data to database here.
        return TaskLifecycleRunResult(next_action="wait_user_input", next_task=None)

    def _next_task_after_turn(self) -> ManagedTask | None:
        next_task = self.consume_next_task()
        if next_task is not None:
            return next_task
        if self.task.status == "active":
            return self.task
        return None

    def _next_action_after_thresholds(
        self,
        *,
        context_token_threshold: int,
        tool_call_threshold: int,
        cancel_event: asyncio.Event | None,
    ) -> TaskLifecycleAction:
        context_tokens = estimate_messages_tokens(self.message_values())
        tool_calls = _count_task_tree_tool_calls(self.task)
        if (
            context_tokens <= context_token_threshold
            and tool_calls <= tool_call_threshold
        ):
            return "normal_run"
        if cancel_event is not None:
            cancel_event.set()
        return "compact"

    async def run_one_turn(
        self,
        *,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event | None = None,
    ) -> UserTaskRunTurnResult:
        tools = self.create_tools()
        user_instruction_message = UserMessage(
            content=[TextContent(text=self.instruction_text())],
            timestamp=int(time.time() * 1000),
        )
        assistant_message = await agent_process.call_llm_step(
            system_prompt=USER_TASK_SYSTEM_PROMPT,
            messages=[*self.message_values(), user_instruction_message],
            tools=tools,
            cancel_event=cancel_event,
        )
        if _assistant_is_error(assistant_message):
            return UserTaskRunTurnResult(
                user_instruction_message=user_instruction_message,
                assistant_message=assistant_message,
                new_messages=[],
                tool_call_records=[],
            )

        assistant_entry = MessageEntry(id=self.allocate_message_id(), message=assistant_message)
        tool_results: list[ToolResultMessage] = []
        tool_call_records: list[tuple[int, ToolCall | None, ToolResultMessage]] = []
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
            tool_call_records, tool_call_tasks = self.create_tool_call_record_task_entries(
                assistant_message=assistant_message,
                tool_result_messages=tool_results,
            )
            self.task.children.extend(tool_call_tasks)
            if tool_call_tasks:
                self.task.touch()

        tool_result_entries = [
            MessageEntry(id=self.allocate_message_id(), message=tool_result)
            for tool_result in tool_results
        ]
        new_messages = [assistant_entry, *tool_result_entries]
        self.append_messages(new_messages)
        return UserTaskRunTurnResult(
            user_instruction_message=user_instruction_message,
            assistant_message=assistant_message,
            new_messages=new_messages,
            tool_call_records=tool_call_records,
        )

    def sync(self, database: Any, session: Any) -> None:
        database.upsert_managed_task(self.task, session=session)
        for child in sorted(self.task.children, key=lambda item: item.id or 0):
            database.upsert_managed_task(child, session=session)

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
            return AgentToolResult(content=[TextContent(text=self.todo_status_text())])

        tool.execute = execute
        return tool

    # ------------------------------------------------------------------
    # User-task compaction phase
    # ------------------------------------------------------------------

    def begin_compaction(self) -> bool:
        if self.task.status != "done" or not self.task.children:
            self._clear_compaction()
            return False
        self._compacting = True
        self._compacted_tool_calls = []
        self._compacted_user_task_finished = False
        return True

    def compaction_instruction_text(self, *, tool_calls: Mapping[int, ToolCallReview]) -> str:
        self._require_compaction()
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
    ) -> UserTaskCompactRunResult:
        compact_instruction = self.compaction_instruction_text(tool_calls={})
        compact_messages = [
            *self.message_values(),
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
        return UserTaskCompactRunResult(
            compact_messages=compact_messages,
            start_message_id=start_message_id,
            end_message_id=end_message_id,
            compacted_messages=compacted_messages,
        )

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
        self._require_compaction()
        self.task.result = description
        self.task.touch()
        return self.task

    def record_compacted_tool_call(self, *, tool_call_log_id: int) -> None:
        self._require_compaction()
        tool_call_task = ToolCallTask(
            id=self.allocate_task_id(),
            title=f"Tool call {tool_call_log_id}",
            status="done",
            parent_id=self.task.id,
            tool_call_log_id=tool_call_log_id,
        )
        self._compacted_tool_calls.append(tool_call_task)

    def finish_compacted_user_task(self) -> UserTask:
        self._require_compacted_result()
        self._compacted_user_task_finished = True
        self.task.touch()
        return self.task

    def compaction_result(self) -> tuple[int, int, list[Any]]:
        self._require_finished_compacted_user_task()
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

    def sync_compaction(self, database: Any, session: Any) -> UserTask:
        self._require_finished_compacted_user_task()
        self.task.children = list(self._compacted_tool_calls)
        self.task.touch()
        database.replace_managed_task_tree(self.task, session=session)
        self._clear_compaction()
        return self.task

    def _require_compaction(self) -> None:
        if not self._compacting:
            raise TaskLifecycleError("Compaction is not active")

    def _require_compacted_result(self) -> None:
        self._require_compaction()
        if self.task.result is None:
            raise TaskLifecycleError("No compacted user task result")

    def _require_finished_compacted_user_task(self) -> None:
        self._require_compacted_result()
        if not self._compacted_user_task_finished:
            raise TaskLifecycleError("Compacted user task is not finished")

    def _clear_compaction(self) -> None:
        self._compacting = False
        self._compacted_tool_calls = []
        self._compacted_user_task_finished = False

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
    def __init__(
        self,
        task: TodoTask,
        *,
        allocate_task_id: Callable[[], int] | None = None,
        user_task: UserTask | None = None,
    ):
        super().__init__(allocate_task_id=allocate_task_id)
        self.task = task
        self.user_task = user_task

    def instruction_text(self) -> str:
        if _count_tool_calls(self.task.children) > 10:
            return (
                "Runtime instruction for this turn:\n"
                "- More than 10 tool calls have run for the active todo.\n"
                "- Determine whether the active todo is finished.\n"
                "- If it is finished, call finish_todo now with a concise result.\n"
                "- If it is not finished, do only the next action needed to complete it."
            )
        return (
            "Runtime instruction for this turn:\n"
            f"- Focus on the active todo: {self.task.title}\n"
            "- Use tools only for work needed by this todo.\n"
            "- Call finish_todo immediately when it is complete."
        )

    def finish_task(self, *, result: str | None = None) -> TodoTask:
        self.task.status = "done"
        self.task.result = result
        self.task.end_message_id = self.current_assistant_message_id
        self.task.touch()
        self.next_task = self.user_task
        return self.task

    def error_task(self, *, error: str, end_message_id: int | None = None) -> TodoTask:
        self.task.status = "error"
        self.task.error = error
        self.task.end_message_id = (
            end_message_id
            if end_message_id is not None
            else self.current_assistant_message_id
        )
        self.task.touch()
        self.next_task = self.user_task
        return self.task

    def create_tools(self) -> list[AgentTool]:
        return [
            self.create_finish_todo_tool(),
            self.create_error_todo_tool(),
        ]

    def sync(self, database: Any, session: Any) -> None:
        database.upsert_managed_task(self.task, session=session)
        for child in sorted(self.task.children, key=lambda item: item.id or 0):
            database.upsert_managed_task(child, session=session)

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
            self.finish_task(
                result=params.get("result"),
            )
            text = (
                todo_status_text(self.user_task)
                if self.user_task is not None
                else f"Todo finished: {self.task.result or self.task.title}"
            )
            return AgentToolResult(content=[TextContent(text=text)])

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
            self.error_task(
                error=params["error"],
            )
            text = (
                todo_status_text(self.user_task)
                if self.user_task is not None
                else f"Todo errored: {self.task.error or self.task.title}"
            )
            return AgentToolResult(content=[TextContent(text=text)])

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


def _count_task_tree_tool_calls(task: ManagedTask) -> int:
    total = 1 if task.kind == "tool_call" else 0
    for child in task.children:
        total += _count_task_tree_tool_calls(child)
    return total


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
