"""Stateful task manager for agent-defined todos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, Literal, Mapping

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import TextContent

from simple_agent.task_manager.models import ManagedTask

if TYPE_CHECKING:
    from simple_agent.db.db import Database
    from sqlmodel import Session


@dataclass(frozen=True)
class CompactScope:
    compact_todos: list[ManagedTask]
    preserved_todos: list[ManagedTask]
    compact_tasks: list[ManagedTask]
    preserved_tasks: list[ManagedTask]


@dataclass
class CompactBuffer:
    todo: ManagedTask | None = None


TaskTreeReviewFormat = Literal["tree", "flat"]


@dataclass(frozen=True)
class ToolCallReview:
    name: str
    arguments: Any | None = None


@dataclass(frozen=True)
class TaskTreeReview:
    text: str
    tool_call_log_ids: dict[int, int]


class TaskManagerError(RuntimeError):
    """Raised when task-manager lifecycle rules are violated."""


class TaskManager:
    """Manage one user task and one active todo at a time."""

    _db: Database
    active_user_task_id: int | None
    active_todo_id: int | None
    _user_task: ManagedTask | None
    _active_todo: ManagedTask | None
    _next_task_id: int | None
    _compact_buffer: CompactBuffer | None
    current_assistant_message_id: int | None

    def __init__(self, db: Database):
        self._db = db
        self.active_user_task_id: int | None = None
        self.active_todo_id: int | None = None
        self._user_task: ManagedTask | None = None
        self._active_todo: ManagedTask | None = None
        self._next_task_id: int | None = None
        self._compact_buffer: CompactBuffer | None = None
        self.current_assistant_message_id: int | None = None

    def load(self, active_user_task_id: int | None, *, session: Session) -> None:
        self._user_task = None
        self._active_todo = None
        self.active_user_task_id = active_user_task_id
        self.active_todo_id = None
        self._next_task_id = self._db.next_managed_task_id(session=session)
        self._compact_buffer = None
        self.current_assistant_message_id = None
        if active_user_task_id is None:
            return
        self._user_task = self.build_task_tree(active_user_task_id, session=session)
        if self._user_task is None:
            raise TaskManagerError("Active user task is missing")
        for task in self._walk_tasks():
            if task.kind == "todo" and task.status == "active":
                self._active_todo = task
                self.active_todo_id = task.id
                break

    def save(self, *, session: Session) -> None:
        for task in sorted(self._walk_tasks(), key=lambda item: item.id or 0):
            self._db.upsert_managed_task(task, session=session)

    # ------------------------------------------------------------------
    # Normal running phase
    # ------------------------------------------------------------------

    def create_user_task(self, input: str) -> ManagedTask:
        if self.active_user_task_id is not None:
            raise TaskManagerError("Cannot create a second active user task")
        task = ManagedTask(kind="user_task", title=input)
        task.id = self._allocate_task_id()
        self._user_task = task
        self.active_user_task_id = task.id
        return task

    @property
    def active_user_task(self) -> ManagedTask | None:
        return self._user_task

    @property
    def active_todo(self) -> ManagedTask | None:
        return self._active_todo

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
            self.create_todo(params["title"])
            return AgentToolResult(content=[TextContent(text=self.todo_status_text())])

        tool.execute = execute
        return tool

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
            self.finish_task(params.get("result"))
            return AgentToolResult(content=[TextContent(text=self.todo_status_text())])

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
            self.error_task(params["error"])
            return AgentToolResult(content=[TextContent(text=self.todo_status_text())])

        tool.execute = execute
        return tool

    def create_todo(self, title: str, start_message_id: int | None = None) -> ManagedTask:
        user_task = self._require_active_user_task()
        if self._active_todo is not None:
            raise TaskManagerError("Cannot create todo while another active todo exists")

        todo = ManagedTask(
            kind="todo",
            title=title,
            parent_id=user_task.id,
            start_message_id=start_message_id if start_message_id is not None else self.current_assistant_message_id,
        )
        todo.id = self._allocate_task_id()

        user_task.children.append(todo)
        user_task.touch()
        self._active_todo = todo
        self.active_todo_id = todo.id
        return todo

    def finish_task(self, result: str | None = None, end_message_id: int | None = None) -> ManagedTask:
        todo = self._require_active_todo()
        todo.status = "done"
        todo.result = result
        todo.end_message_id = end_message_id if end_message_id is not None else self.current_assistant_message_id
        todo.touch()
        self._active_todo = None
        self.active_todo_id = None
        return todo

    def error_task(self, error: str, end_message_id: int | None = None) -> ManagedTask:
        todo = self._require_active_todo()
        todo.status = "error"
        todo.error = error
        todo.end_message_id = end_message_id if end_message_id is not None else self.current_assistant_message_id
        todo.touch()
        self._active_todo = None
        self.active_todo_id = None
        return todo

    def record_tool_call(self, tool_call_id: int) -> ManagedTask:
        if self._active_todo is not None:
            target = self._active_todo
        else:
            target = self._require_active_user_task()
        tool_call_task = ManagedTask(
            kind="tool_call",
            title=f"Tool call {tool_call_id}",
            status="done",
            parent_id=target.id,
            tool_call_log_id=tool_call_id,
        )
        tool_call_task.id = self._allocate_task_id()
        target.children.append(tool_call_task)
        target.touch()
        return tool_call_task

    def finish_user_task(self, result: str | None = None) -> ManagedTask:
        user_task = self._require_active_user_task()
        if self._active_todo is not None:
            raise TaskManagerError("Cannot finish user task while a todo is active")
        user_task.status = "done"
        user_task.result = result
        user_task.touch()
        self.active_user_task_id = None
        return user_task

    def todo_status_text(self) -> str:
        user_task = self._require_active_user_task()
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

    def user_instruction_text(self) -> str:
        if self._user_task is None:
            return (
                "Runtime instruction for this turn:\n"
                "- Wait for the user to provide a task before creating todos or doing tool work."
            )

        if self._active_todo is None:
            tool_calls_after_previous_todo = self._count_tool_calls_after_latest_todo(self._user_task)
            if tool_calls_after_previous_todo > 5:
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

        active_todo_tool_calls = self._count_tool_calls(self._active_todo.children)
        if active_todo_tool_calls > 10:
            return (
                "Runtime instruction for this turn:\n"
                "- More than 10 tool calls have run for the active todo.\n"
                "- Determine whether the active todo is finished.\n"
                "- If it is finished, call finish_todo now with a concise result.\n"
                "- If it is not finished, do only the next action needed to complete it."
            )
        return (
            "Runtime instruction for this turn:\n"
            f"- Focus on the active todo: {self._active_todo.title}\n"
            "- Use tools only for work needed by this todo.\n"
            "- Call finish_todo immediately when it is complete."
        )

    # ------------------------------------------------------------------
    # Compact phase
    # ------------------------------------------------------------------

    def compact_scope(self, *, run_done: bool) -> CompactScope | None:
        user_task = self._require_loaded_user_task() if run_done else self._require_active_user_task()

        if run_done:
            compact_tasks = list(user_task.children)
            if not compact_tasks:
                return None
            compact_todos = [task for task in compact_tasks if task.kind == "todo"]
            return CompactScope(
                compact_todos=compact_todos,
                preserved_todos=[],
                compact_tasks=compact_tasks,
                preserved_tasks=[],
            )

        todos = [task for task in user_task.children if task.kind == "todo"]
        finished = [todo for todo in todos if todo.status == "done"]
        if not finished:
            return None

        latest_finished = finished[-1]
        end_index = todos.index(latest_finished)
        compact_todos = todos[: end_index + 1]
        preserved_todos = todos[end_index + 1:]
        return CompactScope(
            compact_todos=compact_todos,
            preserved_todos=preserved_todos,
            compact_tasks=compact_todos,
            preserved_tasks=preserved_todos,
        )

    def compact_instruction_text(
        self,
        scope: CompactScope,
        *,
        session_id: str,
    ) -> str:
        user_task = self._require_loaded_user_task()
        compact_root = ManagedTask(
            id=user_task.id,
            kind=user_task.kind,
            title=user_task.title,
            status=user_task.status,
            result=user_task.result,
            error=user_task.error,
            children=list(scope.compact_tasks),
            created_at=user_task.created_at,
            updated_at=user_task.updated_at,
        )
        task_view = _TaskTreeReviewRenderer(
            format="tree",
            depth=None,
            tool_calls=self._load_tool_call_reviews(session_id),
        ).render(compact_root)
        return (
            "Runtime instruction for compacting phase:\n"
            "- Complete the compacted task information first: define the compacted task and its result.\n"
            "- Record every must-include tool call based on the compacted task result to avoid context loss.\n"
            "- Use only compact tools: create one compacted todo, record must-include tool calls, then finish it.\n"
            "\n"
            "Task view to compact:\n"
            f"{task_view.text}"
        )

    def begin_compact_buffer(self) -> None:
        self._compact_buffer = CompactBuffer()

    def create_compacted_todo(self, description: str) -> ManagedTask:
        if self._compact_buffer is None:
            raise TaskManagerError("Compact buffer is not active")
        if self._compact_buffer.todo is not None:
            raise TaskManagerError("Compacted todo already exists")
        todo = ManagedTask(kind="todo", title="Compacted work", status="active", result=description)
        todo.id = self._allocate_task_id()
        self._compact_buffer.todo = todo
        return todo

    def record_compacted_tool_call(self, tool_call_log_id: int) -> None:
        if self._compact_buffer is None or self._compact_buffer.todo is None:
            raise TaskManagerError("No compacted todo")
        tool_call_task = ManagedTask(
            kind="tool_call",
            title=f"Tool call {tool_call_log_id}",
            status="done",
            parent_id=self._compact_buffer.todo.id,
            tool_call_log_id=tool_call_log_id,
        )
        tool_call_task.id = self._allocate_task_id()
        self._compact_buffer.todo.children.append(tool_call_task)
        self._compact_buffer.todo.touch()

    def finish_compacted_todo(self) -> ManagedTask:
        if self._compact_buffer is None or self._compact_buffer.todo is None:
            raise TaskManagerError("No compacted todo")
        self._compact_buffer.todo.status = "done"
        return self._compact_buffer.todo

    def consume_compact_buffer(self) -> ManagedTask:
        if self._compact_buffer is None or self._compact_buffer.todo is None:
            raise TaskManagerError("No compacted todo")
        todo = self._compact_buffer.todo
        if todo.status != "done":
            raise TaskManagerError("Compacted todo is not finished")
        self._compact_buffer = None
        return todo

    def create_compact_tools(self) -> list[AgentTool]:
        create_tool = AgentTool(
            name="create_compacted_todo",
            description="Create the single compacted todo with a concise summary.",
            parameters={
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Compacted todo summary"},
                },
                "required": ["description"],
            },
        )

        async def create_execute(tool_call_id, params, cancel_event=None, on_update=None):
            todo = self.create_compacted_todo(params["description"])
            return AgentToolResult(content=[TextContent(text=f"created compacted todo {todo.id}")])

        create_tool.execute = create_execute

        record_tool = AgentTool(
            name="record_compacted_tool_call",
            description="Keep one useful runner tool-call log ID in the compacted todo.",
            parameters={
                "type": "object",
                "properties": {
                    "tool_call_log_id": {"type": "integer", "description": "Runner tool-call log ID"},
                },
                "required": ["tool_call_log_id"],
            },
        )

        async def record_execute(tool_call_id, params, cancel_event=None, on_update=None):
            self.record_compacted_tool_call(params["tool_call_log_id"])
            return AgentToolResult(content=[TextContent(text="recorded compacted tool call")])

        record_tool.execute = record_execute

        finish_tool = AgentTool(
            name="finish_compacted_todo",
            description="Finish the compacted todo after selecting useful tool calls.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        async def finish_execute(tool_call_id, params, cancel_event=None, on_update=None):
            todo = self.finish_compacted_todo()
            return AgentToolResult(content=[TextContent(text=f"finished compacted todo {todo.id}")])

        finish_tool.execute = finish_execute
        return [create_tool, record_tool, finish_tool]

    def replace_compact_scope(self, *, run_done: bool, session: Session) -> ManagedTask:
        user_task = self._require_loaded_user_task() if run_done else self._require_active_user_task()

        scope = self.compact_scope(run_done=run_done)
        if scope is None:
            raise TaskManagerError("No compact scope")
        compacted_todo = self.consume_compact_buffer()
        compacted_original_id = compacted_todo.id
        compacted_todo.id = scope.compact_tasks[0].id
        compacted_todo.parent_id = user_task.id
        for child in compacted_todo.children:
            if child.parent_id == compacted_original_id:
                child.parent_id = compacted_todo.id

        compacted_scope_id_set = {task.id for task in scope.compact_tasks}
        insert_index = self._first_child_index(user_task, scope.compact_tasks)
        user_task.children = [
            child for child in user_task.children
            if child.id not in compacted_scope_id_set
        ]
        user_task.children.insert(insert_index, compacted_todo)
        user_task.touch()

        self._db.replace_managed_task_tree(user_task, session=session)
        return compacted_todo

    # ------------------------------------------------------------------
    # Task tree and helper utilities
    # ------------------------------------------------------------------

    def review_task_tree(
        self,
        *,
        format: TaskTreeReviewFormat = "tree",
        depth: int | None = None,
        tool_calls: Mapping[int, ToolCallReview] | None = None,
    ) -> TaskTreeReview:
        user_task = self._require_active_user_task()
        renderer = _TaskTreeReviewRenderer(format=format, depth=depth, tool_calls=tool_calls or {})
        return renderer.render(user_task)

    def build_task_tree(self, root_task_id: int, *, session: Session) -> ManagedTask | None:
        root = self._db.get_managed_task(root_task_id, session=session)
        if root is None or root.id is None:
            return None

        def attach_children(task: ManagedTask) -> None:
            task.children = []
            for child in self._db.list_managed_task_children(task.id, session=session):
                if child.id is not None:
                    attach_children(child)
                    task.children.append(child)

        attach_children(root)
        return root

    def _allocate_task_id(self) -> int:
        if self._next_task_id is None:
            raise TaskManagerError("Task manager must be loaded before creating tasks")
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _require_active_user_task(self) -> ManagedTask:
        if self._user_task is None or self.active_user_task_id is None:
            raise TaskManagerError("No active user task")
        return self._user_task

    def _require_loaded_user_task(self) -> ManagedTask:
        if self._user_task is None:
            raise TaskManagerError("No loaded user task")
        return self._user_task

    def _require_active_todo(self) -> ManagedTask:
        if self._active_todo is None:
            raise TaskManagerError("No active todo")
        return self._active_todo

    def _walk_tasks(self) -> list[ManagedTask]:
        if self._user_task is None:
            return []
        return self._flatten_task_tree(self._user_task)

    def _flatten_task_tree(self, task: ManagedTask) -> list[ManagedTask]:
        tasks: list[ManagedTask] = []
        stack = [task]
        while stack:
            task = stack.pop()
            tasks.append(task)
            stack.extend(reversed(task.children))
        return tasks

    def _count_tool_calls_after_latest_todo(self, user_task: ManagedTask) -> int:
        latest_todo_index = -1
        for index, child in enumerate(user_task.children):
            if child.kind == "todo":
                latest_todo_index = index
        return self._count_tool_calls(user_task.children[latest_todo_index + 1:])

    def _count_tool_calls(self, tasks: list[ManagedTask]) -> int:
        return sum(1 for task in tasks if task.kind == "tool_call")

    def _load_tool_call_reviews(self, session_id: str) -> dict[int, ToolCallReview]:
        records = self._db.list_runner_tool_calls(session_id)
        return {
            record.id: ToolCallReview(
                name=record.tool_name,
                arguments=self._tool_call_arguments(record.tool_call_json),
            )
            for record in records
            if record.id is not None
        }

    def _tool_call_arguments(self, tool_call_json: str) -> object | None:
        try:
            payload = json.loads(tool_call_json)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload.get("arguments")

    def _first_child_index(self, user_task: ManagedTask, tasks: list[ManagedTask]) -> int:
        task_ids = {task.id for task in tasks}
        for index, child in enumerate(user_task.children):
            if child.id in task_ids:
                return index
        return len(user_task.children)


class _TaskTreeReviewRenderer:
    def __init__(
        self,
        *,
        format: TaskTreeReviewFormat,
        depth: int | None,
        tool_calls: Mapping[int, ToolCallReview],
    ):
        self._format = format
        self._depth = depth
        self._tool_calls = tool_calls
        self._lines: list[str] = ["Task tree:"]
        self._tool_call_log_ids: dict[int, int] = {}
        self._next_tool_call_seq = 1

    def render(self, user_task: ManagedTask) -> TaskTreeReview:
        self._render_task(user_task, depth=0)
        return TaskTreeReview(
            text="\n".join(self._lines),
            tool_call_log_ids=self._tool_call_log_ids,
        )

    def _render_task(self, task: ManagedTask, *, depth: int) -> None:
        self._append_task(task, depth=depth)
        if self._format == "flat":
            for tool_call in self._tool_calls_in_tree(task):
                self._append_tool_call(tool_call, depth=depth + 1)
            return

        if self._depth is not None and depth >= self._depth:
            return

        for child in task.children:
            if child.kind == "tool_call":
                self._append_tool_call(child, depth=depth + 1)
                continue
            self._render_task(child, depth=depth + 1)

    def _append_task(self, task: ManagedTask, *, depth: int) -> None:
        self._lines.append(f"{self._indent(depth)}- {task.kind} [{task.status}] {task.title}")
        if task.result:
            self._lines.append(f"{self._indent(depth + 1)}result: {task.result}")
        if task.error:
            self._lines.append(f"{self._indent(depth + 1)}error: {task.error}")

    def _append_tool_call(self, task: ManagedTask, *, depth: int) -> None:
        seq = self._next_tool_call_seq
        self._next_tool_call_seq += 1
        if task.tool_call_log_id is not None:
            self._tool_call_log_ids[seq] = task.tool_call_log_id

        details = self._tool_call_details(task)
        line = f"{self._indent(depth)}- tool_call {seq}. {details.name}"
        if details.arguments is not None:
            line += f" args: {self._format_arguments(details.arguments)}"
        self._lines.append(line)

    def _tool_call_details(self, task: ManagedTask) -> ToolCallReview:
        if task.tool_call_log_id is None:
            return ToolCallReview(name="unknown_tool")
        return self._tool_calls.get(task.tool_call_log_id, ToolCallReview(name="unknown_tool"))

    def _tool_calls_in_tree(self, task: ManagedTask) -> list[ManagedTask]:
        tool_calls: list[ManagedTask] = []
        stack = list(reversed(task.children))
        while stack:
            child = stack.pop()
            if child.kind == "tool_call":
                tool_calls.append(child)
            else:
                stack.extend(reversed(child.children))
        return tool_calls

    def _format_arguments(self, arguments: Any) -> str:
        if isinstance(arguments, str):
            return arguments
        if hasattr(arguments, "model_dump_json"):
            return arguments.model_dump_json()
        return json.dumps(arguments, separators=(",", ":"))

    def _indent(self, depth: int) -> str:
        return "  " * depth
