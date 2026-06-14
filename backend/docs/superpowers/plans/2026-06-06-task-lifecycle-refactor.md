# Task Lifecycle Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move normal task lifecycle instruction behavior from `TaskManager` onto `UserTask` and `TodoTask`, while keeping `TaskManager` responsible for task tree ownership and database sync.

**Architecture:** Add a small `TaskRuntimeContext` model in `simple_agent.task_manager.models`. `UserTask` and `TodoTask` expose `instruction_text(context)` methods; `ToolCallTask` remains data-only. `TaskManager` computes the active lifecycle target, routes instruction generation to the task, and `SessionRunner` passes current runtime context into the manager.

**Tech Stack:** Python 3.14, Pydantic models, SQLModel-backed storage, pytest, `pi.agent` tools.

---

## File Structure

- Modify `src/simple_agent/task_manager/models.py`
  - Add `TaskRuntimeContext`.
  - Add lifecycle helper methods to `UserTask` and `TodoTask`.
  - Keep `ToolCallTask` data-only.
- Modify `src/simple_agent/task_manager/manager.py`
  - Change `user_instruction_text()` to accept `TaskRuntimeContext`.
  - Add `active_task_tool_call_count()` for runner context construction.
  - Keep task mutation and tool creation in the manager.
- Modify `src/simple_agent/task_manager/__init__.py`
  - Export `TaskRuntimeContext`.
- Modify `src/simple_agent/session/runner.py`
  - Build `TaskRuntimeContext` before creating the steering user instruction.
  - Pass the context to `TaskManager.user_instruction_text(context)`.
- Add `tests/test_task_lifecycle.py`
  - Focused model-level lifecycle tests.
- Modify `tests/test_task_manager.py`
  - Update manager instruction routing tests to pass context.
- Modify `tests/test_session_runner.py`
  - Keep runner behavior expectations passing after the new context argument.

---

### Task 1: Add Task Lifecycle Model Tests

**Files:**
- Create: `tests/test_task_lifecycle.py`
- Modify after tests fail: `src/simple_agent/task_manager/models.py`

- [ ] **Step 1: Write failing tests for task lifecycle instruction text**

Create `tests/test_task_lifecycle.py`:

```python
from simple_agent.task_manager.models import (
    TaskRuntimeContext,
    TodoTask,
    ToolCallTask,
    UserTask,
)


def _context(*, active_task_tool_calls: int) -> TaskRuntimeContext:
    return TaskRuntimeContext(
        session_id="session_a",
        context_tokens=100,
        total_tool_calls=active_task_tool_calls,
        active_task_tool_calls=active_task_tool_calls,
    )


def test_user_task_instruction_asks_for_complexity_check_when_tool_count_is_small():
    task = UserTask(title="Build feature")

    instruction = task.instruction_text(_context(active_task_tool_calls=2))

    assert "Runtime instruction for this turn" in instruction
    assert "Determine whether the user task is complex" in instruction
    assert "create the next small atomic todo first" in instruction


def test_user_task_instruction_requires_todo_after_many_tool_calls():
    task = UserTask(title="Build feature")

    instruction = task.instruction_text(_context(active_task_tool_calls=6))

    assert "More than 5 tool calls have run since the previous todo" in instruction
    assert "create a small atomic todo before doing more work" in instruction


def test_todo_task_instruction_focuses_active_todo_when_tool_count_is_small():
    task = TodoTask(title="Inspect files")

    instruction = task.instruction_text(_context(active_task_tool_calls=3))

    assert "Focus on the active todo: Inspect files" in instruction
    assert "Call finish_todo immediately when it is complete" in instruction


def test_todo_task_instruction_prompts_finish_check_after_many_tool_calls():
    task = TodoTask(title="Inspect files")

    instruction = task.instruction_text(_context(active_task_tool_calls=11))

    assert "More than 10 tool calls have run for the active todo" in instruction
    assert "call finish_todo now with a concise result" in instruction


def test_tool_call_task_remains_data_only():
    task = ToolCallTask(title="Tool call 1", tool_call_log_id=1)

    assert not hasattr(task, "instruction_text")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_task_lifecycle.py -q
```

Expected: FAIL because `TaskRuntimeContext` and `instruction_text` do not exist.

- [ ] **Step 3: Add `TaskRuntimeContext` and lifecycle methods**

Modify `src/simple_agent/task_manager/models.py`.

Add this class after `TaskStatus`:

```python
class TaskRuntimeContext(BaseModel):
    """Transient runtime data used by task lifecycle decisions."""

    session_id: str
    context_tokens: int
    total_tool_calls: int
    active_task_tool_calls: int
    current_assistant_message_id: int | None = None
    run_done: bool = False
```

Add this method to `UserTask`:

```python
    def instruction_text(self, context: TaskRuntimeContext) -> str:
        if context.active_task_tool_calls > 5:
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
```

Add this method to `TodoTask`:

```python
    def instruction_text(self, context: TaskRuntimeContext) -> str:
        if context.active_task_tool_calls > 10:
            return (
                "Runtime instruction for this turn:\n"
                "- More than 10 tool calls have run for the active todo.\n"
                "- Determine whether the active todo is finished.\n"
                "- If it is finished, call finish_todo now with a concise result.\n"
                "- If it is not finished, do only the next action needed to complete it."
            )
        return (
            "Runtime instruction for this turn:\n"
            f"- Focus on the active todo: {self.title}\n"
            "- Use tools only for work needed by this todo.\n"
            "- Call finish_todo immediately when it is complete."
        )
```

Do not add `instruction_text` to `ToolCallTask`.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/test_task_lifecycle.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/simple_agent/task_manager/models.py tests/test_task_lifecycle.py
git commit -m "Add task lifecycle instruction methods"
```

---

### Task 2: Route TaskManager Instructions Through Task Classes

**Files:**
- Modify: `src/simple_agent/task_manager/manager.py`
- Modify: `src/simple_agent/task_manager/__init__.py`
- Modify: `tests/test_task_manager.py`

- [ ] **Step 1: Update manager instruction tests to pass runtime context**

In `tests/test_task_manager.py`, import `TaskRuntimeContext`:

```python
from simple_agent.task_manager.models import TaskRuntimeContext, TodoTask, UserTask
```

Add this helper near `_save`:

```python
def _runtime_context(manager: TaskManager) -> TaskRuntimeContext:
    return TaskRuntimeContext(
        session_id="session_a",
        context_tokens=100,
        total_tool_calls=0,
        active_task_tool_calls=manager.active_task_tool_call_count(),
    )
```

Update each `manager.user_instruction_text()` call to:

```python
manager.user_instruction_text(_runtime_context(manager))
```

Keep the existing assertions unchanged.

- [ ] **Step 2: Add a routing test for active todo priority**

Add this test to `tests/test_task_manager.py` near the existing instruction tests:

```python
def test_user_instruction_routes_to_active_todo_before_user_task():
    db = _make_db()
    manager = TaskManager(db)
    _load(manager, None)
    manager.create_user_task("Build feature")
    manager.create_todo("Inspect files")

    instruction = manager.user_instruction_text(_runtime_context(manager))

    assert "Focus on the active todo: Inspect files" in instruction
    assert "Determine whether the user task is complex" not in instruction
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_task_manager.py -q
```

Expected: FAIL because `TaskManager.active_task_tool_call_count()` does not exist and `user_instruction_text()` does not accept context.

- [ ] **Step 4: Export `TaskRuntimeContext`**

Modify `src/simple_agent/task_manager/__init__.py`:

```python
from simple_agent.task_manager.models import (
    BaseTask,
    ManagedTask,
    TaskRuntimeContext,
    TodoTask,
    ToolCallTask,
    UserTask,
)

__all__ = [
    "TaskManager",
    "TaskManagerError",
    "TaskTreeReview",
    "ToolCallReview",
    "BaseTask",
    "ManagedTask",
    "TaskRuntimeContext",
    "TodoTask",
    "ToolCallTask",
    "UserTask",
]
```

- [ ] **Step 5: Change manager instruction routing**

Modify imports in `src/simple_agent/task_manager/manager.py`:

```python
from simple_agent.task_manager.models import (
    ManagedTask,
    TaskRuntimeContext,
    TodoTask,
    ToolCallTask,
    UserTask,
)
```

Replace `user_instruction_text` with:

```python
    def user_instruction_text(self, context: TaskRuntimeContext) -> str:
        if self._user_task is None:
            return (
                "Runtime instruction for this turn:\n"
                "- Wait for the user to provide a task before creating todos or doing tool work."
            )

        if self._active_todo is not None:
            return self._active_todo.instruction_text(context)

        return self._user_task.instruction_text(context)
```

Add this public helper near `todo_status_text`:

```python
    def active_task_tool_call_count(self) -> int:
        if self._active_todo is not None:
            return self._count_tool_calls(self._active_todo.children)
        if self._user_task is not None:
            return self._count_tool_calls_after_latest_todo(self._user_task)
        return 0
```

- [ ] **Step 6: Run task manager tests**

Run:

```bash
uv run pytest tests/test_task_lifecycle.py tests/test_task_manager.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/simple_agent/task_manager/__init__.py src/simple_agent/task_manager/manager.py tests/test_task_manager.py
git commit -m "Route task instructions through lifecycle methods"
```

---

### Task 3: Pass Runtime Context From SessionRunner

**Files:**
- Modify: `src/simple_agent/session/runner.py`
- Modify: `tests/test_session_runner.py`

- [ ] **Step 1: Add a runner assertion for lifecycle context routing**

In `tests/test_session_runner.py`, add this test near the instruction/logging tests:

```python
@pytest.mark.asyncio
async def test_session_runner_passes_runtime_context_to_task_instruction(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    agent_process = FakeFinalAgentProcess()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        agent_process=agent_process,
        cancel_event=asyncio.Event(),
    )

    await runner.run("Build feature")

    instruction_message = agent_process.calls[0]["messages"][-1]
    instruction = instruction_message.content[0].text
    assert "Runtime instruction for this turn" in instruction
    assert "Determine whether the user task is complex" in instruction
```

- [ ] **Step 2: Run runner tests to verify failure**

Run:

```bash
uv run pytest tests/test_session_runner.py -q
```

Expected: FAIL because `SessionRunner` still calls `TaskManager.user_instruction_text()` without context after Task 2.

- [ ] **Step 3: Import `TaskRuntimeContext` in runner**

Modify the type-checking import area in `src/simple_agent/session/runner.py` so runtime code can access the class:

```python
from simple_agent.task_manager import TaskRuntimeContext
```

Keep the existing `TYPE_CHECKING` imports for `TaskManager`.

- [ ] **Step 4: Add a context builder to `SessionRunner`**

Add this method near `pause_for_compaction_if_needed`:

```python
    def task_runtime_context(self, *, additional_tool_calls: int = 0) -> TaskRuntimeContext:
        context_tokens = estimate_messages_tokens(self._message_values())
        with self._db.create_session() as session:
            total_tool_calls = len(self._db.list_runner_tool_calls(self._session_id, session=session))
        return TaskRuntimeContext(
            session_id=self._session_id,
            context_tokens=context_tokens,
            total_tool_calls=total_tool_calls + additional_tool_calls,
            active_task_tool_calls=self._task_manager.active_task_tool_call_count(),
            current_assistant_message_id=self._task_manager.current_assistant_message_id,
            run_done=self._user_task_is_done(),
        )
```

- [ ] **Step 5: Pass context into `user_instruction_text`**

In `handle_running`, replace:

```python
        user_instruction_message = UserMessage(
            content=[TextContent(text=self._task_manager.user_instruction_text())],
            timestamp=int(time.time() * 1000),
        )
```

with:

```python
        user_instruction_context = self.task_runtime_context()
        user_instruction_message = UserMessage(
            content=[TextContent(text=self._task_manager.user_instruction_text(user_instruction_context))],
            timestamp=int(time.time() * 1000),
        )
```

- [ ] **Step 6: Keep compaction threshold logic behavior unchanged**

Leave `pause_for_compaction_if_needed` as-is in this task. It already uses persisted total tool calls and token estimates. This plan only routes instruction lifecycle logic through task classes.

- [ ] **Step 7: Run runner tests**

Run:

```bash
uv run pytest tests/test_session_runner.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add src/simple_agent/session/runner.py tests/test_session_runner.py
git commit -m "Pass task runtime context from session runner"
```

---

### Task 4: Focused Regression Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Run task-focused tests**

Run:

```bash
uv run pytest tests/test_task_lifecycle.py tests/test_task_manager.py tests/test_task_tools.py tests/test_task_tree.py -q
```

Expected: PASS.

- [ ] **Step 2: Run runner/session tests**

Run:

```bash
uv run pytest tests/test_session_runner.py tests/test_session.py tests/test_session_manager.py -q
```

Expected: PASS.

- [ ] **Step 3: Check formatting-sensitive diff issues**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 4: Inspect remaining working tree**

Run:

```bash
git status --short
```

Expected: only intentional changes are present. If `tests/manual_session_cli.py` is still modified from earlier unrelated work, leave it unstaged unless the user explicitly asks to include it.

- [ ] **Step 5: Commit verification-only cleanup if needed**

If verification required small fixes, commit those fixes:

```bash
git add src/simple_agent/task_manager/models.py src/simple_agent/task_manager/manager.py src/simple_agent/task_manager/__init__.py src/simple_agent/session/runner.py tests/test_task_lifecycle.py tests/test_task_manager.py tests/test_session_runner.py
git commit -m "Stabilize task lifecycle refactor"
```

If no fixes were needed, do not create an empty commit.
