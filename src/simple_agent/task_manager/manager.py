"""Stateful task manager for agent-defined todos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, Literal, Mapping

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import AssistantMessage, TextContent, ToolResultMessage

from simple_agent.task_manager.models import ManagedTask, TaskRuntimeContext, TodoTask, ToolCallTask, UserTask

if TYPE_CHECKING:
    from simple_agent.db.db import Database
    from pi.agent.types import AgentMessage
    from sqlmodel import Session


class BaseCompaction:
    def __init__(self, *, user_task: ManagedTask):
        self.user_task = user_task
        self.to_compact_tasks = self._select_to_compact_tasks()
        self.result: ManagedTask | None = None

    def has_tasks(self) -> bool:
        return bool(self.to_compact_tasks)

    def create_result(self, *, task_id: int, description: str) -> ManagedTask:
        if self.result is not None:
            raise TaskManagerError("Compacted todo already exists")
        self.result = TodoTask(title="Compacted work", status="active", result=description, id=task_id)
        return self.result

    def record_tool_call(self, *, task_id: int, tool_call_log_id: int) -> None:
        result = self._require_result()
        tool_call_task = ToolCallTask(
            title=f"Tool call {tool_call_log_id}",
            status="done",
            parent_id=result.id,
            tool_call_log_id=tool_call_log_id,
            id=task_id,
        )
        result.children.append(tool_call_task)
        result.touch()

    def finish_result(self) -> ManagedTask:
        result = self._require_result()
        result.status = "done"
        return result

    def message_scope(self) -> tuple[int, int]:
        start_message_id, end_message_id = self._message_boundary()
        if start_message_id is None or end_message_id is None:
            raise TaskManagerError("Compact scope is missing message boundaries")
        return start_message_id, end_message_id

    def format_messages(self) -> list["AgentMessage"]:
        result = self._require_result()
        if result.status != "done":
            raise TaskManagerError("Compacted todo is not finished")
        tool_refs = [
            child.tool_call_log_id
            for child in result.children
            if child.kind == "tool_call" and child.tool_call_log_id is not None
        ]
        text = (
            f"Compacted todo: {result.result or result.title}\n"
            f"Useful tool calls: {tool_refs}"
        )
        return [AssistantMessage(role="assistant", content=[TextContent(text=text)])]

    def replace_in_user_task(self) -> ManagedTask:
        result = self._require_result()
        if result.status != "done":
            raise TaskManagerError("Compacted todo is not finished")
        original_result_id = result.id
        result.id = self.to_compact_tasks[0].id
        result.parent_id = self.user_task.id
        for child in result.children:
            if child.parent_id == original_result_id:
                child.parent_id = result.id

        compacted_ids = {task.id for task in self.to_compact_tasks}
        insert_index = self._first_child_index()
        self.user_task.children = [
            child for child in self.user_task.children
            if child.id not in compacted_ids
        ]
        self.user_task.children.insert(insert_index, result)
        self.user_task.touch()
        return result

    def _message_boundary(self) -> tuple[int | None, int | None]:
        raise NotImplementedError

    def _select_to_compact_tasks(self) -> list[ManagedTask]:
        raise NotImplementedError

    def _require_result(self) -> ManagedTask:
        if self.result is None:
            raise TaskManagerError("No compacted todo")
        return self.result

    def _first_child_index(self) -> int:
        task_ids = {task.id for task in self.to_compact_tasks}
        for index, child in enumerate(self.user_task.children):
            if child.id in task_ids:
                return index
        return len(self.user_task.children)


class UserTaskCompaction(BaseCompaction):
    def _select_to_compact_tasks(self) -> list[ManagedTask]:
        return list(self.user_task.children)

    def _message_boundary(self) -> tuple[int | None, int | None]:
        return self.user_task.start_message_id, self.user_task.end_message_id


class TodoTaskCompaction(BaseCompaction):
    def _select_to_compact_tasks(self) -> list[ManagedTask]:
        todos = [task for task in self.user_task.children if task.kind == "todo"]
        finished = [todo for todo in todos if todo.status == "done"]
        if not finished:
            return []
        latest_finished = finished[-1]
        end_index = todos.index(latest_finished)
        return todos[: end_index + 1]

    def _message_boundary(self) -> tuple[int | None, int | None]:
        return self.to_compact_tasks[0].start_message_id, self.to_compact_tasks[-1].end_message_id


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
    _compaction: BaseCompaction | None
    current_assistant_message_id: int | None

    def __init__(self, db: Database):
        self._db = db
        self.active_user_task_id: int | None = None
        self.active_todo_id: int | None = None
        self._user_task: ManagedTask | None = None
        self._active_todo: ManagedTask | None = None
        self._next_task_id: int | None = None
        self._compaction: BaseCompaction | None = None
        self.current_assistant_message_id: int | None = None

    def load(self, active_user_task_id: int | None, *, session: Session) -> None:
        self._user_task = None
        self._active_todo = None
        self.active_user_task_id = active_user_task_id
        self.active_todo_id = None
        self._next_task_id = self._db.next_managed_task_id(session=session)
        self._compaction = None
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
        if self._user_task is not None:
            self._user_task.sync(self._db, session)
        if self._active_todo is not None:
            self._active_todo.sync(self._db, session)

    # ------------------------------------------------------------------
    # Normal running phase
    # ------------------------------------------------------------------

    def create_user_task(self, input: str, start_message_id: int | None = None) -> ManagedTask:
        if self.active_user_task_id is not None:
            raise TaskManagerError("Cannot create a second active user task")
        task = UserTask(title=input, start_message_id=start_message_id)
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

    def create_tools(self) -> list[AgentTool]:
        if self._active_todo is not None:
            return self._active_todo.create_tools(self)
        user_task = self._require_active_user_task()
        return user_task.create_tools(self)

    def create_create_todo_tool(self) -> AgentTool:
        return self._require_active_user_task().create_create_todo_tool(self)

    def create_finish_user_task_tool(self) -> AgentTool:
        return self._require_active_user_task().create_finish_user_task_tool(self)

    def create_finish_todo_tool(self) -> AgentTool:
        return self._require_active_todo().create_finish_todo_tool(self)

    def create_error_todo_tool(self) -> AgentTool:
        return self._require_active_todo().create_error_todo_tool(self)

    def create_todo(self, title: str, start_message_id: int | None = None) -> ManagedTask:
        user_task = self._require_active_user_task()
        if self._active_todo is not None:
            raise TaskManagerError("Cannot create todo while another active todo exists")

        todo = user_task.create_todo_task(
            task_id=self._allocate_task_id(),
            title=title,
            start_message_id=start_message_id if start_message_id is not None else self.current_assistant_message_id,
        )
        self._active_todo = todo
        self.active_todo_id = todo.id
        return todo

    def finish_task(self, result: str | None = None, end_message_id: int | None = None) -> ManagedTask:
        todo = self._require_active_todo()
        todo.finish_task(
            result=result,
            end_message_id=end_message_id if end_message_id is not None else self.current_assistant_message_id,
        )
        self._active_todo = None
        self.active_todo_id = None
        return todo

    def error_task(self, error: str, end_message_id: int | None = None) -> ManagedTask:
        todo = self._require_active_todo()
        todo.error_task(
            error=error,
            end_message_id=end_message_id if end_message_id is not None else self.current_assistant_message_id,
        )
        self._active_todo = None
        self.active_todo_id = None
        return todo

    def record_tool_call(
        self,
        tool_call_id: int,
        *,
        assistant_message: AssistantMessage | None = None,
        tool_result_message: ToolResultMessage | None = None,
    ) -> ManagedTask:
        if self._active_todo is not None:
            target = self._active_todo
        else:
            target = self._require_active_user_task()
        return target.append_tool_call_task(
            task_id=self._allocate_task_id(),
            tool_call_log_id=tool_call_id,
            assistant_message=assistant_message,
            tool_result_message=tool_result_message,
        )

    def record_turn_tool_calls(
        self,
        *,
        assistant_message: AssistantMessage,
        tool_call_records: list[tuple[int, Any, ToolResultMessage]],
    ) -> list[ManagedTask]:
        tasks: list[ManagedTask] = []
        for log_id, _tool_call, tool_result in tool_call_records:
            tasks.append(
                self.record_tool_call(
                    log_id,
                    assistant_message=assistant_message,
                    tool_result_message=tool_result,
                )
            )
        return tasks

    def finish_user_task(self, result: str | None = None, end_message_id: int | None = None) -> ManagedTask:
        user_task = self._require_active_user_task()
        if self._active_todo is not None:
            raise TaskManagerError("Cannot finish user task while a todo is active")
        user_task.finish_task(result=result, end_message_id=end_message_id)
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

    def active_task_tool_call_count(self) -> int:
        if self._active_todo is not None:
            return self._count_tool_calls(self._active_todo.children)
        if self._user_task is not None:
            return self._count_tool_calls_after_latest_todo(self._user_task)
        return 0

    def user_instruction_text(self, context: TaskRuntimeContext) -> str:
        if self._user_task is None:
            return (
                "Runtime instruction for this turn:\n"
                "- Wait for the user to provide a task before creating todos or doing tool work."
            )

        if self._active_todo is not None:
            return self._active_todo.instruction_text(context)

        return self._user_task.instruction_text(context)

    # ------------------------------------------------------------------
    # Compact phase
    # ------------------------------------------------------------------

    def begin_compact(self, *, run_done: bool) -> bool:
        user_task = self._require_loaded_user_task() if run_done else self._require_active_user_task()
        if run_done:
            compaction = UserTaskCompaction(user_task=user_task)
        else:
            compaction = TodoTaskCompaction(user_task=user_task)
        if not compaction.has_tasks():
            self._compaction = None
            return False

        self._compaction = compaction
        return True

    def compact_instruction_text(
        self,
        *,
        session_id: str,
    ) -> str:
        compaction = self._require_compaction()
        user_task = self._require_loaded_user_task()
        compact_root = UserTask(
            id=user_task.id,
            title=user_task.title,
            status=user_task.status,
            result=user_task.result,
            error=user_task.error,
            children=list(compaction.to_compact_tasks),
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

    def compacted_messages(self) -> tuple[int, int, list["AgentMessage"]]:
        compaction = self._require_compaction()
        start_message_id, end_message_id = compaction.message_scope()
        return start_message_id, end_message_id, compaction.format_messages()

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
            todo = self._require_compaction().create_result(
                task_id=self._allocate_task_id(),
                description=params["description"],
            )
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
            self._require_compaction().record_tool_call(
                task_id=self._allocate_task_id(),
                tool_call_log_id=params["tool_call_log_id"],
            )
            return AgentToolResult(content=[TextContent(text="recorded compacted tool call")])

        record_tool.execute = record_execute

        finish_tool = AgentTool(
            name="finish_compacted_todo",
            description="Finish the compacted todo after selecting useful tool calls.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        async def finish_execute(tool_call_id, params, cancel_event=None, on_update=None):
            todo = self._require_compaction().finish_result()
            return AgentToolResult(content=[TextContent(text=f"finished compacted todo {todo.id}")])

        finish_tool.execute = finish_execute
        return [create_tool, record_tool, finish_tool]

    def sync_compaction(self, *, session: Session) -> ManagedTask:
        compaction = self._require_compaction()
        compacted_todo = compaction.replace_in_user_task()
        self._db.replace_managed_task_tree(compaction.user_task, session=session)
        self._compaction = None
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

    def _require_compaction(self) -> BaseCompaction:
        if self._compaction is None:
            raise TaskManagerError("Compaction is not active")
        return self._compaction

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
