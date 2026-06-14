# Task Manager Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current task-tree runtime control flow with a generic agent runtime plus a separate task manager backed by agent-callable todo tools.

**Architecture:** Add unified task models and a `TaskManager` that owns `UserTask -> ordered items -> Todo/ToolCall` state. Keep `AgentProcess` generic and move task behavior into tools and orchestration, then replace `Session.run`'s central-control loop with one runtime call using task tools and normal coding tools.

**Tech Stack:** Python 3.14, Pydantic, SQLModel/SQLite, pytest, pi-agent/pi-ai tool and message types.

---

## File Structure

- Create `src/simple_agent/task_manager/__init__.py`
  - Public exports for task manager models and service.
- Create `src/simple_agent/task_manager/models.py`
  - Unified `ManagedTask`, `TaskItem`, task status/kind literals, and serialization helpers.
- Create `src/simple_agent/task_manager/manager.py`
  - In-memory lifecycle operations, persistence coordination through `Database`, visible ordered items, and compaction replacement.
- Create `tests/test_task_manager.py`
  - Unit tests for lifecycle, ordering, and compaction.
- Modify `src/simple_agent/state/state.py`
  - Add SQLModel rows for the new managed tasks. Keep existing rows initially so old tests and rollback remain possible during migration.
- Modify `src/simple_agent/db/db.py`
  - Add CRUD helpers for managed tasks. Preserve existing tool-call persistence.
- Modify `src/simple_agent/tool/tool_mgr.py`
  - Add task tools backed by `TaskManager`. Add optional task-manager-aware wrapping so normal tool calls are recorded under the active todo/user task.
- Modify `src/simple_agent/session/session.py`
  - Replace central-control-driven run loop with a single generic agent runtime run over task tools plus normal tools.
- Modify `tests/test_session.py`
- Modify `tests/test_session_manager.py`
  - Update session expectations around the simplified run behavior.
- Replace old tree-focused tests after replacement:
  - `tests/test_task_tree.py`
  - `tests/test_central_control.py`
  - `tests/test_plan_runner.py`
  - `tests/test_explore_runner.py`

---

### Task 1: Add Unified Task Models

**Files:**
- Create: `src/simple_agent/task_manager/__init__.py`
- Create: `src/simple_agent/task_manager/models.py`
- Test: `tests/test_task_manager.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/test_task_manager.py` with:

```python
"""Tests for the replacement task manager."""

from __future__ import annotations

from simple_agent.task_manager.models import ManagedTask, TaskItem


def test_task_item_defaults_to_task_ref():
    item = TaskItem(kind="task", ref_id=10)
    assert item.kind == "task"
    assert item.ref_id == 10


def test_managed_task_defaults():
    task = ManagedTask(kind="user_task", title="Build feature")
    assert task.kind == "user_task"
    assert task.status == "active"
    assert task.items == []
    assert task.result is None
    assert task.error is None


def test_managed_task_accepts_mixed_ordered_items():
    task = ManagedTask(
        kind="user_task",
        title="Build feature",
        items=[
            TaskItem(kind="tool_call", ref_id=1),
            TaskItem(kind="task", ref_id=2),
            TaskItem(kind="tool_call", ref_id=3),
        ],
    )
    assert [(item.kind, item.ref_id) for item in task.items] == [
        ("tool_call", 1),
        ("task", 2),
        ("tool_call", 3),
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_task_manager.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'simple_agent.task_manager'`.

- [ ] **Step 3: Implement task models**

Create `src/simple_agent/task_manager/models.py`:

```python
"""Unified task-manager models."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

TaskKind = Literal["user_task", "todo", "aggregate"]
TaskStatus = Literal["active", "done", "error"]
TaskItemKind = Literal["task", "tool_call"]


class TaskItem(BaseModel):
    """A visible ordered reference owned by a managed task."""

    kind: TaskItemKind
    ref_id: int


class ManagedTask(BaseModel):
    """Unified task model for user tasks, todos, and aggregate tasks."""

    id: int | None = None
    parent_id: int | None = None
    kind: TaskKind
    title: str
    status: TaskStatus = "active"
    items: list[TaskItem] = Field(default_factory=list)
    result: str | None = None
    error: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()
```

Create `src/simple_agent/task_manager/__init__.py`:

```python
"""Task manager package."""

from simple_agent.task_manager.models import ManagedTask, TaskItem

__all__ = ["ManagedTask", "TaskItem"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_task_manager.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent/task_manager tests/test_task_manager.py
git commit -m "Add unified task manager models"
```

---

### Task 2: Persist Managed Tasks

**Files:**
- Modify: `src/simple_agent/state/state.py`
- Modify: `src/simple_agent/db/db.py`
- Modify: `tests/test_task_manager.py`

- [ ] **Step 1: Add failing persistence test**

Append to `tests/test_task_manager.py`:

```python
import tempfile

from simple_agent.db.db import Database


def _make_db() -> Database:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return Database(f.name)


def test_managed_task_roundtrip_preserves_items():
    db = _make_db()
    task = ManagedTask(
        kind="user_task",
        title="Build feature",
        items=[TaskItem(kind="tool_call", ref_id=1), TaskItem(kind="task", ref_id=2)],
    )

    task.id = db.upsert_managed_task(task)
    loaded = db.get_managed_task(task.id)

    assert loaded is not None
    assert loaded.id == task.id
    assert loaded.kind == "user_task"
    assert loaded.title == "Build feature"
    assert [(item.kind, item.ref_id) for item in loaded.items] == [
        ("tool_call", 1),
        ("task", 2),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_task_manager.py::test_managed_task_roundtrip_preserves_items -q`

Expected: FAIL with `AttributeError: 'Database' object has no attribute 'upsert_managed_task'`.

- [ ] **Step 3: Add managed task DB row and adapters**

In `src/simple_agent/state/state.py`, add imports:

```python
from simple_agent.task_manager.models import ManagedTask, TaskItem
```

Add after `TaskRecord`:

```python
class ManagedTaskRecord(SQLModel, table=True):
    """SQLite model for the replacement task manager."""

    id: int | None = Field(default=None, primary_key=True)
    parent_id: int | None = Field(default=None, index=True)
    kind: str = Field(index=True)
    title: str
    status: str = Field(default="active", index=True)
    items: str | None = None
    result: str | None = None
    error: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
```

Add below adapters:

```python
_task_item_adapter = TypeAdapter(list[TaskItem])


def managed_task_to_record(task: ManagedTask) -> ManagedTaskRecord:
    return ManagedTaskRecord(
        id=task.id,
        parent_id=task.parent_id,
        kind=task.kind,
        title=task.title,
        status=task.status,
        items=_task_item_adapter.dump_json(task.items).decode("utf-8"),
        result=task.result,
        error=task.error,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


def managed_task_from_record(record: ManagedTaskRecord) -> ManagedTask:
    return ManagedTask(
        id=record.id,
        parent_id=record.parent_id,
        kind=record.kind,
        title=record.title,
        status=record.status,
        items=_task_item_adapter.validate_json(record.items or "[]"),
        result=record.result,
        error=record.error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
```

- [ ] **Step 4: Add Database methods**

In `src/simple_agent/db/db.py`, extend the import from `simple_agent.state.state`:

```python
    ManagedTaskRecord,
    managed_task_from_record,
    managed_task_to_record,
```

Add methods after task operations:

```python
    # ------------------------------------------------------------------
    # ManagedTask operations
    # ------------------------------------------------------------------

    @standalone_or_compose
    def upsert_managed_task(self, task, *, session: Session | None = None) -> int:
        """Insert or update a replacement task-manager row."""
        record = session.merge(managed_task_to_record(task))
        session.flush()
        task.id = record.id
        return record.id

    @standalone_or_compose
    def get_managed_task(self, task_id: int, *, session: Session | None = None):
        """Return a managed task by ID, or None."""
        record = session.get(ManagedTaskRecord, task_id)
        if record is None:
            return None
        session.expunge(record)
        return managed_task_from_record(record)

    @standalone_or_compose
    def list_managed_tasks(self, *, session: Session | None = None):
        """Return all managed tasks ordered by ID."""
        records = list(session.exec(select(ManagedTaskRecord).order_by(ManagedTaskRecord.id)).all())
        for record in records:
            session.expunge(record)
        return [managed_task_from_record(record) for record in records]
```

- [ ] **Step 5: Run persistence test**

Run: `uv run pytest tests/test_task_manager.py::test_managed_task_roundtrip_preserves_items -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/simple_agent/state/state.py src/simple_agent/db/db.py tests/test_task_manager.py
git commit -m "Persist managed task records"
```

---

### Task 3: Implement TaskManager Lifecycle

**Files:**
- Create: `src/simple_agent/task_manager/manager.py`
- Modify: `src/simple_agent/task_manager/__init__.py`
- Modify: `tests/test_task_manager.py`

- [ ] **Step 1: Add failing lifecycle tests**

Append to `tests/test_task_manager.py`:

```python
import pytest

from simple_agent.task_manager import TaskManager, TaskManagerError


def test_create_user_task_sets_active_user_task():
    db = _make_db()
    manager = TaskManager(db)

    user_task = manager.create_user_task("Build feature")

    assert user_task.id is not None
    assert user_task.kind == "user_task"
    assert user_task.title == "Build feature"
    assert manager.active_user_task_id == user_task.id


def test_create_todo_appends_task_item_to_user_task():
    db = _make_db()
    manager = TaskManager(db)
    user_task = manager.create_user_task("Build feature")

    todo = manager.create_todo("Inspect files")
    loaded_user_task = db.get_managed_task(user_task.id)

    assert todo.parent_id == user_task.id
    assert manager.active_todo_id == todo.id
    assert [(item.kind, item.ref_id) for item in loaded_user_task.items] == [
        ("task", todo.id),
    ]


def test_create_todo_rejects_existing_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.create_user_task("Build feature")
    manager.create_todo("Inspect files")

    with pytest.raises(TaskManagerError, match="active todo"):
        manager.create_todo("Edit files")


def test_finish_todo_marks_done_and_clears_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")

    finished = manager.finish_todo("Found app.py")

    assert finished.id == todo.id
    assert finished.status == "done"
    assert finished.result == "Found app.py"
    assert manager.active_todo_id is None


def test_finish_todo_rejects_missing_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.create_user_task("Build feature")

    with pytest.raises(TaskManagerError, match="No active todo"):
        manager.finish_todo()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_task_manager.py -q`

Expected: FAIL with import error for `TaskManager`.

- [ ] **Step 3: Implement manager lifecycle**

Create `src/simple_agent/task_manager/manager.py`:

```python
"""Stateful task manager for agent-defined todos."""

from __future__ import annotations

from simple_agent.db.db import Database
from simple_agent.task_manager.models import ManagedTask, TaskItem


class TaskManagerError(RuntimeError):
    """Raised when task-manager lifecycle rules are violated."""


class TaskManager:
    """Manage one user task and one active todo at a time."""

    def __init__(self, db: Database):
        self._db = db
        self.active_user_task_id: int | None = None
        self.active_todo_id: int | None = None

    def create_user_task(self, input: str) -> ManagedTask:
        if self.active_user_task_id is not None:
            raise TaskManagerError("Cannot create a second active user task")
        task = ManagedTask(kind="user_task", title=input)
        task.id = self._db.upsert_managed_task(task)
        self.active_user_task_id = task.id
        return task

    def create_todo(self, title: str) -> ManagedTask:
        user_task = self._require_user_task()
        if self.active_todo_id is not None:
            raise TaskManagerError("Cannot create todo while another active todo exists")

        todo = ManagedTask(kind="todo", title=title, parent_id=user_task.id)
        todo.id = self._db.upsert_managed_task(todo)

        user_task.items.append(TaskItem(kind="task", ref_id=todo.id))
        user_task.touch()
        self._db.upsert_managed_task(user_task)
        self.active_todo_id = todo.id
        return todo

    def finish_todo(self, result: str | None = None) -> ManagedTask:
        todo = self._require_active_todo()
        todo.status = "done"
        todo.result = result
        todo.touch()
        self._db.upsert_managed_task(todo)
        self.active_todo_id = None
        return todo

    def error_todo(self, error: str) -> ManagedTask:
        todo = self._require_active_todo()
        todo.status = "error"
        todo.error = error
        todo.touch()
        self._db.upsert_managed_task(todo)
        self.active_todo_id = None
        return todo

    def finish_user_task(self, result: str | None = None) -> ManagedTask:
        user_task = self._require_user_task()
        if self.active_todo_id is not None:
            raise TaskManagerError("Cannot finish user task while a todo is active")
        user_task.status = "done"
        user_task.result = result
        user_task.touch()
        self._db.upsert_managed_task(user_task)
        self.active_user_task_id = None
        return user_task

    def _require_user_task(self) -> ManagedTask:
        if self.active_user_task_id is None:
            raise TaskManagerError("No active user task")
        task = self._db.get_managed_task(self.active_user_task_id)
        if task is None:
            raise TaskManagerError("Active user task is missing")
        return task

    def _require_active_todo(self) -> ManagedTask:
        if self.active_todo_id is None:
            raise TaskManagerError("No active todo")
        task = self._db.get_managed_task(self.active_todo_id)
        if task is None:
            raise TaskManagerError("Active todo is missing")
        return task
```

Modify `src/simple_agent/task_manager/__init__.py`:

```python
"""Task manager package."""

from simple_agent.task_manager.manager import TaskManager, TaskManagerError
from simple_agent.task_manager.models import ManagedTask, TaskItem

__all__ = ["ManagedTask", "TaskItem", "TaskManager", "TaskManagerError"]
```

- [ ] **Step 4: Run lifecycle tests**

Run: `uv run pytest tests/test_task_manager.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent/task_manager tests/test_task_manager.py
git commit -m "Add task manager lifecycle"
```

---

### Task 4: Record Tool Calls And Preserve Ordering

**Files:**
- Modify: `src/simple_agent/task_manager/manager.py`
- Modify: `tests/test_task_manager.py`

- [ ] **Step 1: Add failing tool-call placement tests**

Append to `tests/test_task_manager.py`:

```python
def test_record_tool_call_without_active_todo_attaches_to_user_task():
    db = _make_db()
    manager = TaskManager(db)
    user_task = manager.create_user_task("Build feature")

    manager.record_tool_call(7)
    loaded_user_task = db.get_managed_task(user_task.id)

    assert [(item.kind, item.ref_id) for item in loaded_user_task.items] == [
        ("tool_call", 7),
    ]


def test_record_tool_call_with_active_todo_attaches_to_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")

    manager.record_tool_call(8)
    loaded_todo = db.get_managed_task(todo.id)

    assert [(item.kind, item.ref_id) for item in loaded_todo.items] == [
        ("tool_call", 8),
    ]


def test_mixed_user_task_order_is_preserved():
    db = _make_db()
    manager = TaskManager(db)
    user_task = manager.create_user_task("Build feature")

    manager.record_tool_call(1)
    todo = manager.create_todo("Inspect files")
    manager.finish_todo()
    manager.record_tool_call(2)

    loaded_user_task = db.get_managed_task(user_task.id)
    assert [(item.kind, item.ref_id) for item in loaded_user_task.items] == [
        ("tool_call", 1),
        ("task", todo.id),
        ("tool_call", 2),
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_task_manager.py::test_record_tool_call_without_active_todo_attaches_to_user_task tests/test_task_manager.py::test_record_tool_call_with_active_todo_attaches_to_todo tests/test_task_manager.py::test_mixed_user_task_order_is_preserved -q`

Expected: FAIL with `AttributeError: 'TaskManager' object has no attribute 'record_tool_call'`.

- [ ] **Step 3: Implement `record_tool_call`**

Add to `TaskManager` in `src/simple_agent/task_manager/manager.py`:

```python
    def record_tool_call(self, tool_call_id: int) -> None:
        target = self._require_active_todo() if self.active_todo_id is not None else self._require_user_task()
        target.items.append(TaskItem(kind="tool_call", ref_id=tool_call_id))
        target.touch()
        self._db.upsert_managed_task(target)
```

- [ ] **Step 4: Run task-manager tests**

Run: `uv run pytest tests/test_task_manager.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent/task_manager/manager.py tests/test_task_manager.py
git commit -m "Record tool calls in task timelines"
```

---

### Task 5: Implement Explicit Compaction Replacement

**Files:**
- Modify: `src/simple_agent/task_manager/manager.py`
- Modify: `tests/test_task_manager.py`

- [ ] **Step 1: Add failing compaction tests**

Append to `tests/test_task_manager.py`:

```python
def test_compact_items_replaces_visible_tasks_with_aggregate_task():
    db = _make_db()
    manager = TaskManager(db)
    user_task = manager.create_user_task("Build feature")
    first = manager.create_todo("Inspect files")
    manager.record_tool_call(1)
    manager.finish_todo("Inspected files")
    second = manager.create_todo("Edit files")
    manager.record_tool_call(2)
    manager.finish_todo("Edited files")

    aggregate = manager.compact_items(
        parent_task_id=user_task.id,
        item_refs=[TaskItem(kind="task", ref_id=first.id), TaskItem(kind="task", ref_id=second.id)],
        title="Inspect and edit files",
        result="Inspected files and edited them.",
        items=[TaskItem(kind="tool_call", ref_id=2)],
    )
    loaded_user_task = db.get_managed_task(user_task.id)

    assert aggregate.kind == "aggregate"
    assert aggregate.parent_id == user_task.id
    assert aggregate.result == "Inspected files and edited them."
    assert [(item.kind, item.ref_id) for item in aggregate.items] == [("tool_call", 2)]
    assert [(item.kind, item.ref_id) for item in loaded_user_task.items] == [("task", aggregate.id)]


def test_compact_items_rejects_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    user_task = manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")

    with pytest.raises(TaskManagerError, match="active"):
        manager.compact_items(
            parent_task_id=user_task.id,
            item_refs=[TaskItem(kind="task", ref_id=todo.id)],
            title="Aggregate",
            result="Nope",
            items=[],
        )


def test_compact_items_rejects_duplicate_refs():
    db = _make_db()
    manager = TaskManager(db)
    user_task = manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")
    manager.finish_todo()

    with pytest.raises(TaskManagerError, match="duplicate"):
        manager.compact_items(
            parent_task_id=user_task.id,
            item_refs=[TaskItem(kind="task", ref_id=todo.id), TaskItem(kind="task", ref_id=todo.id)],
            title="Aggregate",
            result="Nope",
            items=[],
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_task_manager.py::test_compact_items_replaces_visible_tasks_with_aggregate_task tests/test_task_manager.py::test_compact_items_rejects_active_todo tests/test_task_manager.py::test_compact_items_rejects_duplicate_refs -q`

Expected: FAIL with `AttributeError: 'TaskManager' object has no attribute 'compact_items'`.

- [ ] **Step 3: Implement compaction**

Add to `TaskManager` in `src/simple_agent/task_manager/manager.py`:

```python
    def compact_items(
        self,
        parent_task_id: int,
        item_refs: list[TaskItem],
        title: str,
        result: str,
        items: list[TaskItem],
    ) -> ManagedTask:
        if not item_refs:
            raise TaskManagerError("Cannot compact an empty item list")

        seen = {(item.kind, item.ref_id) for item in item_refs}
        if len(seen) != len(item_refs):
            raise TaskManagerError("Cannot compact duplicate refs")

        parent = self._db.get_managed_task(parent_task_id)
        if parent is None:
            raise TaskManagerError("Parent task is missing")

        visible = [(item.kind, item.ref_id) for item in parent.items]
        selected = [(item.kind, item.ref_id) for item in item_refs]
        if not all(ref in visible for ref in selected):
            raise TaskManagerError("Cannot compact refs outside parent visible items")

        for item in item_refs:
            if item.kind != "task":
                continue
            task = self._db.get_managed_task(item.ref_id)
            if task is None:
                raise TaskManagerError("Cannot compact missing task")
            if task.status == "active":
                raise TaskManagerError("Cannot compact active task")

        aggregate = ManagedTask(
            kind="aggregate",
            parent_id=parent.id,
            title=title,
            status="done",
            result=result,
            items=list(items),
        )
        aggregate.id = self._db.upsert_managed_task(aggregate)

        selected_set = set(selected)
        new_items: list[TaskItem] = []
        inserted = False
        for item in parent.items:
            ref = (item.kind, item.ref_id)
            if ref in selected_set:
                if not inserted:
                    new_items.append(TaskItem(kind="task", ref_id=aggregate.id))
                    inserted = True
                continue
            new_items.append(item)

        parent.items = new_items
        parent.touch()
        self._db.upsert_managed_task(parent)
        return aggregate
```

- [ ] **Step 4: Run task-manager tests**

Run: `uv run pytest tests/test_task_manager.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent/task_manager/manager.py tests/test_task_manager.py
git commit -m "Add explicit task compaction"
```

---

### Task 6: Add Task Tools And Tool-Call Attachment

**Files:**
- Modify: `src/simple_agent/tool/tool_mgr.py`
- Modify: `tests/test_tool_storage.py` or create `tests/test_task_tools.py`

- [ ] **Step 1: Add failing task tool tests**

Create `tests/test_task_tools.py`:

```python
"""Tests for task-manager-backed tools."""

from __future__ import annotations

import asyncio
import tempfile

from simple_agent.db.db import Database
from simple_agent.task_manager import TaskManager
from simple_agent.tool.tool_mgr import ToolMgr


def _make_db() -> Database:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return Database(f.name)


def test_create_todo_tool_creates_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.create_user_task("Build feature")
    tools = ToolMgr(db, task_manager=manager)
    tool = tools.create_create_todo_tool()

    async def run():
        return await tool.execute("call_1", {"title": "Inspect files"})

    result = asyncio.run(run())

    assert "created todo" in result.content[0].text.lower()
    assert manager.active_todo_id is not None


def test_finish_todo_tool_finishes_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")
    tools = ToolMgr(db, task_manager=manager)
    tool = tools.create_finish_todo_tool()

    async def run():
        return await tool.execute("call_1", {"result": "Inspected files"})

    asyncio.run(run())
    loaded = db.get_managed_task(todo.id)

    assert loaded.status == "done"
    assert loaded.result == "Inspected files"
    assert manager.active_todo_id is None


def test_normal_tool_call_records_under_active_todo():
    db = _make_db()
    manager = TaskManager(db)
    manager.create_user_task("Build feature")
    todo = manager.create_todo("Inspect files")
    tools = ToolMgr(db, task_manager=manager)

    from pi.agent import AgentTool, AgentToolResult
    from pi.ai.types import TextContent

    async def execute(tool_call_id, params, cancel_event=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="raw output")])

    tool = AgentTool(
        name="sample",
        description="Sample",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    wrapped = tools.wrap_tools(tool)

    async def run():
        return await wrapped.execute("call_1", {})

    asyncio.run(run())
    loaded_todo = db.get_managed_task(todo.id)

    assert len(loaded_todo.items) == 1
    assert loaded_todo.items[0].kind == "tool_call"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_task_tools.py -q`

Expected: FAIL because `ToolMgr.__init__` does not accept `task_manager`.

- [ ] **Step 3: Extend ToolMgr with task manager support**

Modify `ToolMgr.__init__` in `src/simple_agent/tool/tool_mgr.py`:

```python
    def __init__(self, db: Database | None = None, task_manager=None):
        self.tools: list[AgentTool] = []
        self._db = db or Database()
        self._task_manager = task_manager
```

In `wrap_tools`, after `insert_tool_call`:

```python
            log_id = self._db.insert_tool_call(tool_exec)
            if self._task_manager is not None:
                self._task_manager.record_tool_call(log_id)
            return res
```

Replace the current insert/return block:

```python
            self._db.insert_tool_call(tool_exec)
            return res
```

- [ ] **Step 4: Add task tool factories**

Add methods to `ToolMgr`:

```python
    def create_create_todo_tool(self) -> AgentTool:
        tool = AgentTool(
            name="create_todo",
            description="Create a todo for the next coherent unit of work.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title for the todo"},
                },
                "required": ["title"],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            if self._task_manager is None:
                return AgentToolResult(content=[TextContent(text="task manager is not configured")])
            todo = self._task_manager.create_todo(params["title"])
            return AgentToolResult(content=[TextContent(text=f"created todo {todo.id}")])

        tool.execute = execute
        return tool

    def create_finish_todo_tool(self) -> AgentTool:
        tool = AgentTool(
            name="finish_todo",
            description="Mark the active todo as done.",
            parameters={
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "Optional concise result for this todo"},
                },
                "required": [],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            if self._task_manager is None:
                return AgentToolResult(content=[TextContent(text="task manager is not configured")])
            todo = self._task_manager.finish_todo(params.get("result"))
            return AgentToolResult(content=[TextContent(text=f"finished todo {todo.id}")])

        tool.execute = execute
        return tool

    def create_error_todo_tool(self) -> AgentTool:
        tool = AgentTool(
            name="error_todo",
            description="Mark the active todo as failed.",
            parameters={
                "type": "object",
                "properties": {
                    "error": {"type": "string", "description": "Error details for the active todo"},
                },
                "required": ["error"],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            if self._task_manager is None:
                return AgentToolResult(content=[TextContent(text="task manager is not configured")])
            todo = self._task_manager.error_todo(params["error"])
            return AgentToolResult(content=[TextContent(text=f"errored todo {todo.id}")])

        tool.execute = execute
        return tool
```

- [ ] **Step 5: Run task tool tests**

Run: `uv run pytest tests/test_task_tools.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/simple_agent/tool/tool_mgr.py tests/test_task_tools.py
git commit -m "Add task manager tools"
```

---

### Task 7: Simplify Session Runtime Orchestration

**Files:**
- Modify: `src/simple_agent/session/session.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: Add failing session structure test**

Append to `tests/test_session.py`:

```python
def test_session_initializes_task_manager(tmp_path):
    from simple_agent.session.session import Session

    session = Session(base_dir=str(tmp_path))

    assert session._task_manager is not None
    assert session._tools_mgr._task_manager is session._task_manager
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_session.py::test_session_initializes_task_manager -q`

Expected: FAIL because `Session` has no `_task_manager`.

- [ ] **Step 3: Wire TaskManager into Session**

In `src/simple_agent/session/session.py`, add import:

```python
from simple_agent.task_manager import TaskManager
```

Replace:

```python
        self._tools_mgr = ToolMgr(self._db)
```

with:

```python
        self._task_manager = TaskManager(self._db)
        self._tools_mgr = ToolMgr(self._db, task_manager=self._task_manager)
```

- [ ] **Step 4: Run session structure test**

Run: `uv run pytest tests/test_session.py::test_session_initializes_task_manager -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent/session/session.py tests/test_session.py
git commit -m "Wire task manager into sessions"
```

---

### Task 8: Replace Central-Control Run Loop

**Files:**
- Modify: `src/simple_agent/session/session.py`
- Modify: `tests/test_session.py`

- [ ] **Step 1: Add failing run behavior test**

Append to `tests/test_session.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_session_run_creates_user_task_and_calls_agent_once(tmp_path, monkeypatch):
    from simple_agent.session.session import Session
    from simple_agent.process.agent_process import AgentState

    calls = []

    async def fake_run(self, system_prompt, messages, tools, state, user_prompt=""):
        calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": [tool.name for tool in tools],
                "user_prompt": user_prompt,
            }
        )
        state.new_messages = []
        return state

    monkeypatch.setattr("simple_agent.process.agent_process.AgentProcess.run", fake_run)

    session = Session(base_dir=str(tmp_path))
    result = await session.run("Build feature")

    assert result is not None
    assert result.kind == "user_task"
    assert result.title == "Build feature"
    assert len(calls) == 1
    assert "create_todo" in calls[0]["tools"]
    assert "finish_todo" in calls[0]["tools"]
    assert "error_todo" in calls[0]["tools"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_session.py::test_session_run_creates_user_task_and_calls_agent_once -q`

Expected: FAIL because `Session.run` returns the old `Task` model or enters the old central-control path.

- [ ] **Step 3: Replace `Session.run` body**

In `src/simple_agent/session/session.py`, add a system prompt constant near `_log`:

```python
SYSTEM_PROMPT = """You are a helpful coding agent.

Use create_todo before starting a coherent unit of work.
Call finish_todo when the active todo is complete.
Call error_todo if the active todo cannot be completed.
Keep responses concise and use available tools to do the work.
"""
```

Replace `Session.run` with:

```python
    @logged(_log)
    async def run(self, user_input: str):
        """Run the generic agent runtime for one user task."""
        self._running = True
        if self.event_queue is None:
            self.event_queue = asyncio.Queue()

        user_task = self._task_manager.create_user_task(user_input)
        self._cursor_id = user_task.id
        self._checkpoint()

        state = AgentState()
        tools = [
            self._tools_mgr.create_create_todo_tool(),
            self._tools_mgr.create_finish_todo_tool(),
            self._tools_mgr.create_error_todo_tool(),
            *self._tools_mgr.create_all_tools("."),
        ]

        try:
            await self._agent_process.run(
                system_prompt=SYSTEM_PROMPT,
                messages=[],
                tools=tools,
                state=state,
                user_prompt=user_input,
            )
            if self._task_manager.active_todo_id is None:
                user_task = self._task_manager.finish_user_task()
        except Exception:
            _log.exception("run: session=%s failed", self._id)
            if self.event_queue is not None:
                self.event_queue.put_nowait({"type": "error"})
            raise
        finally:
            self._running = False
            if self.event_queue is not None:
                self.event_queue.put_nowait(None)
                self.event_queue = None

        _log.info("run: session=%s done, result=%s", self._id, user_task.id if user_task else None)
        return user_task
```

Keep `_checkpoint` for session metadata. Leave old central-control imports in place for this task; Task 10 removes them.

- [ ] **Step 4: Run session run test**

Run: `uv run pytest tests/test_session.py::test_session_run_creates_user_task_and_calls_agent_once -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent/session/session.py tests/test_session.py
git commit -m "Run sessions through generic runtime"
```

---

### Task 9: Update Session Manager Tests

**Files:**
- Modify: `tests/test_session_manager.py`

- [ ] **Step 1: Update run API mock to return a managed task**

In `tests/test_session_manager.py`, find the `test_run_session` test and update its `mock_run` helper to return a managed task:

```python
async def mock_run(self, user_input):
    from simple_agent.task_manager.models import ManagedTask
    return ManagedTask(id=1, kind="user_task", title=user_input, status="done")
```

Keep the response assertions unchanged. `src/simple_agent/web/session_api.py` streams session events and does not serialize the `Session.run` return value directly.

- [ ] **Step 2: Run session manager tests**

Run: `uv run pytest tests/test_session_manager.py -q`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_session_manager.py
git commit -m "Update session manager tests for managed tasks"
```

---

### Task 10: Retire Old Task Tree Runtime Coupling

**Files:**
- Modify: `src/simple_agent/session/session.py`
- Modify: `tests/test_task_tree.py`
- Modify: `tests/test_central_control.py`
- Modify: `tests/test_plan_runner.py`
- Modify: `tests/test_explore_runner.py`

- [ ] **Step 1: Run old coupling tests to verify they still target removed behavior**

Run: `uv run pytest tests/test_task_tree.py tests/test_central_control.py tests/test_plan_runner.py tests/test_explore_runner.py -q`

Expected: FAIL after replacement because old central-control semantics are no longer the runtime contract.

- [ ] **Step 2: Remove unused imports and wiring from Session**

In `src/simple_agent/session/session.py`, remove imports that are no longer used:

```python
from simple_agent.process.central_control import CentralControl
from simple_agent.process.runners import CollectRunner, SingleRunRunner
from simple_agent.process.explore_runner import ExploreRunner
from simple_agent.process.plan_runner import PlanRunner
from simple_agent.snapshot.ghost_indexer import RepoWatcher
from simple_agent.state.state import Task
```

Remove runner dictionary and `self._cc` initialization from `Session.__init__`.

Remove these old cursor-loading helpers:

```python
root
_load_cursor
_ensure_task_metadata
```

Keep `_checkpoint`, `pause`, `resume`, `park`, and session metadata behavior.

- [ ] **Step 3: Replace old task-tree test content**

Replace `tests/test_task_tree.py` content with a compatibility note test that points to the new model:

```python
"""Compatibility tests for the retired task tree runtime."""

from simple_agent.task_manager.models import ManagedTask, TaskItem


def test_replacement_task_model_uses_ordered_items():
    task = ManagedTask(
        kind="user_task",
        title="Build feature",
        items=[TaskItem(kind="tool_call", ref_id=1), TaskItem(kind="task", ref_id=2)],
    )

    assert [(item.kind, item.ref_id) for item in task.items] == [
        ("tool_call", 1),
        ("task", 2),
    ]
```

Replace `tests/test_central_control.py`, `tests/test_plan_runner.py`, and `tests/test_explore_runner.py` with this skipped module-level marker:

```python
import pytest

pytestmark = pytest.mark.skip(reason="Central-control task tree runtime was replaced by TaskManager")
```

- [ ] **Step 4: Run targeted tests**

Run: `uv run pytest tests/test_task_manager.py tests/test_task_tools.py tests/test_task_tree.py tests/test_session.py tests/test_session_manager.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simple_agent/session/session.py tests/test_task_tree.py tests/test_central_control.py tests/test_plan_runner.py tests/test_explore_runner.py
git commit -m "Retire old task tree runtime"
```

---

### Task 11: Full Verification And Cleanup

**Files:**
- Review: all modified files

- [ ] **Step 1: Search for stale runtime coupling**

Run: `rg -n "CentralControl|PlanRunner|ExploreRunner|running_task_id|finished_task_ids|define_task|determine_state|context_complete" src tests`

Expected: Remaining matches are either intentionally preserved old model compatibility code or skipped/deleted tests. No active `Session` runtime path should depend on these names.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest -q`

Expected: PASS.

- [ ] **Step 3: Run web app smoke import**

Run: `uv run python -c "from simple_agent.web.app import create_app; app = create_app(':memory:'); print(app.title)"`

Expected: prints `Simple Agent Web`.

- [ ] **Step 4: Review git diff**

Run: `git diff --stat HEAD~11..HEAD`

Expected: Changes are limited to task manager, DB persistence, tool wiring, session runtime, tests, and plan/spec docs.

- [ ] **Step 5: Commit cleanup**

```bash
git add src/simple_agent/session/session.py tests/test_task_tree.py tests/test_central_control.py tests/test_plan_runner.py tests/test_explore_runner.py
git commit -m "Clean up task manager replacement"
```
