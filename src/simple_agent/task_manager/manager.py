"""Stateful task manager for agent-defined todos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import TextContent

from simple_agent.fractional_index import key_after
from simple_agent.task_manager.models import ManagedTask

if TYPE_CHECKING:
    from simple_agent.db.db import Database


@dataclass(frozen=True)
class CompactScope:
    compact_todos: list[ManagedTask]
    preserved_todos: list[ManagedTask]


@dataclass
class CompactBuffer:
    todo: ManagedTask | None = None


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
    _next_task_seq: str | None

    def __init__(self, db: Database):
        self._db = db
        self.active_user_task_id: int | None = None
        self.active_todo_id: int | None = None
        self._user_task: ManagedTask | None = None
        self._active_todo: ManagedTask | None = None
        self._next_task_id: int | None = None
        self._compact_buffer: CompactBuffer | None = None
        self._next_task_seq: str | None = None

    def load(self, active_user_task_id: int | None) -> None:
        self._user_task = None
        self._active_todo = None
        self.active_user_task_id = active_user_task_id
        self.active_todo_id = None
        self._next_task_id = self._db.next_managed_task_id()
        self._compact_buffer = None
        self._next_task_seq = self._db.next_managed_task_seq()
        if active_user_task_id is None:
            return
        self._user_task = self.build_task_tree(active_user_task_id)
        if self._user_task is None:
            raise TaskManagerError("Active user task is missing")
        for task in self._walk_tasks():
            if task.kind == "todo" and task.status == "active":
                self._active_todo = task
                self.active_todo_id = task.id
                break

    def save(self, session=None) -> None:
        if session is None:
            with self._db.create_session() as session:
                self.save(session=session)
                session.commit()
            return

        for task in sorted(self._walk_tasks(), key=lambda item: item.id or 0):
            self._db.upsert_managed_task(task, session=session)

    def create_user_task(self, input: str) -> ManagedTask:
        if self.active_user_task_id is not None:
            raise TaskManagerError("Cannot create a second active user task")
        task = ManagedTask(kind="user_task", title=input)
        task.id = self._allocate_task_id()
        self._assign_next_task_seq(task)
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
            self.create_todo(params["title"], tool_call_id=tool_call_id)
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
            self.finish_task(params.get("result"), tool_call_id=tool_call_id)
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
            self.error_task(params["error"], tool_call_id=tool_call_id)
            return AgentToolResult(content=[TextContent(text=self.todo_status_text())])

        tool.execute = execute
        return tool

    def create_todo(self, title: str, tool_call_id: str | None = None) -> ManagedTask:
        user_task = self._require_active_user_task()
        if self._active_todo is not None:
            raise TaskManagerError("Cannot create todo while another active todo exists")

        todo = ManagedTask(
            kind="todo",
            title=title,
            parent_id=user_task.id,
            create_tool_call_id=tool_call_id,
        )
        todo.id = self._allocate_task_id()

        self._assign_next_task_seq(todo)
        user_task.children.append(todo)
        user_task.children.sort(key=lambda task: task.seq)
        user_task.touch()
        self._active_todo = todo
        self.active_todo_id = todo.id
        return todo

    def todo_status_text(self) -> str:
        user_task = self._require_active_user_task()
        todos = [child for child in user_task.children if child.kind == "todo"]
        if not todos:
            return "Todos: []"

        lines = ["Todos:"]
        for todo in todos:
            line = f"- {todo.id}: [{todo.status}] {todo.title}"
            if todo.result:
                line += f" result={todo.result}"
            if todo.error:
                line += f" error={todo.error}"
            lines.append(line)
        return "\n".join(lines)

    def finish_task(self, result: str | None = None, tool_call_id: str | None = None) -> ManagedTask:
        todo = self._require_active_todo()
        todo.status = "done"
        todo.result = result
        todo.end_tool_call_id = tool_call_id
        todo.touch()
        self._active_todo = None
        self.active_todo_id = None
        return todo

    def error_task(self, error: str, tool_call_id: str | None = None) -> ManagedTask:
        todo = self._require_active_todo()
        todo.status = "error"
        todo.error = error
        todo.end_tool_call_id = tool_call_id
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
        self._assign_next_task_seq(tool_call_task)
        target.children.append(tool_call_task)
        target.children.sort(key=lambda task: task.seq)
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

    def compact_scope(self) -> CompactScope | None:
        user_task = self._require_active_user_task()

        todos = [task for task in user_task.children if task.kind == "todo"]
        finished = [todo for todo in todos if todo.status == "done"]
        if not finished:
            return None

        latest_finished = finished[-1]
        end_index = todos.index(latest_finished)
        compact_todos = todos[: end_index + 1]
        preserved_todos = todos[end_index + 1:]
        return CompactScope(compact_todos, preserved_todos)

    def begin_compact_buffer(self) -> None:
        self._compact_buffer = CompactBuffer()

    def create_compacted_todo(self, description: str) -> ManagedTask:
        if self._compact_buffer is None:
            raise TaskManagerError("Compact buffer is not active")
        if self._compact_buffer.todo is not None:
            raise TaskManagerError("Compacted todo already exists")
        todo = ManagedTask(kind="todo", title="Compacted work", status="active", result=description)
        todo.id = self._allocate_task_id()
        self._assign_next_task_seq(todo)
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
        self._assign_next_task_seq(tool_call_task)
        self._compact_buffer.todo.children.append(tool_call_task)
        self._compact_buffer.todo.children.sort(key=lambda task: task.seq)
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

    def replace_compact_scope(self, *, session=None) -> ManagedTask:
        if session is None:
            with self._db.create_session() as session:
                compacted_todo = self.replace_compact_scope(session=session)
                session.commit()
                return compacted_todo

        user_task = self._require_active_user_task()

        scope = self.compact_scope()
        if scope is None:
            raise TaskManagerError("No compact scope")
        compacted_todo = self.consume_compact_buffer()
        compacted_todo.parent_id = user_task.id
        compacted_todo.seq = scope.compact_todos[0].seq

        compacted_scope_ids = [
            task.id
            for todo in scope.compact_todos
            for task in self._flatten_task_tree(todo)
            if task.id is not None
        ]
        compacted_scope_id_set = {todo.id for todo in scope.compact_todos}
        user_task.children = [
            child for child in user_task.children
            if child.id not in compacted_scope_id_set
        ]
        user_task.children.append(compacted_todo)
        user_task.children.sort(key=lambda task: task.seq)
        user_task.touch()

        self._db.delete_managed_tasks(compacted_scope_ids, session=session)
        session.flush()
        self._db.upsert_managed_task(user_task, session=session)
        for task in self._flatten_task_tree(compacted_todo):
            self._db.upsert_managed_task(task, session=session)
        return compacted_todo

    def build_task_tree(self, root_task_id: int) -> ManagedTask | None:
        root = self._db.get_managed_task(root_task_id)
        if root is None or root.id is None:
            return None

        def attach_children(task: ManagedTask) -> None:
            task.children = []
            for child in self._db.list_managed_task_children(task.id):
                if child.id is not None:
                    attach_children(child)
                    task.children.append(child)
            task.children.sort(key=lambda child: child.seq)

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

    def _require_active_todo(self) -> ManagedTask:
        if self._active_todo is None:
            raise TaskManagerError("No active todo")
        return self._active_todo

    def _assign_next_task_seq(self, task: ManagedTask) -> None:
        if self._next_task_seq is None:
            raise TaskManagerError("Task manager must be loaded before creating tasks")
        task.seq = self._next_task_seq
        self._next_task_seq = key_after(task.seq)

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
