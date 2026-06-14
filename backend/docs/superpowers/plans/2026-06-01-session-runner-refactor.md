# Session Runner Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `Session.run` so a dedicated `SessionRunner` owns session-run state, persistence, messages, tool wrapping, and task lifecycle while `AgentProcess` becomes a simple executor returning only messages.

**Architecture:** Add runner persistence models and DB helpers, extend `ToolExecutionLogger` to write runner tool-call logs, simplify `AgentProcess.run`, and introduce `SessionRunner` for the `Session.run` workflow. Keep `TaskManager` as the task source of truth in the same DB file and limit non-session runner changes to compatibility with the new `AgentProcess.run` signature.

**Tech Stack:** Python 3.14, SQLModel, Pydantic `TypeAdapter`, pytest, pytest-asyncio, `pi.agent` message/tool types.

---

## File Structure

- Create `src/simple_agent/session/runner.py`: `SessionRunner`, minimal phase machine, tool assembly, runner checkpointing, and message persistence orchestration.
- Modify `src/simple_agent/state/state.py`: add SQLModel records for `RunnerStateMetadataRecord`, `RunnerMessageRecord`, and `RunnerToolCallRecord`; add message serialization helpers.
- Modify `src/simple_agent/db/db.py`: add CRUD helpers for runner metadata, runner messages, and runner tool calls.
- Modify `src/simple_agent/tool/execution_logger.py`: add runner tool-call writes around every wrapped tool execution.
- Modify `src/simple_agent/process/agent_process.py`: remove `AgentState`; change `run` to accept `cancel_event` and return `list[AgentMessage]`.
- Modify `src/simple_agent/session/session.py`: delegate `run` to `SessionRunner`; keep event queue and running lifecycle in `Session`.
- Modify `src/simple_agent/process/explore_runner.py`: compatibility-only update for the new `AgentProcess.run` signature.
- Modify tests in `tests/test_agent_process.py`, `tests/test_runner_storage.py`, `tests/test_execution_logger.py`, `tests/test_session_runner.py`, and `tests/test_session.py`.

---

### Task 1: Add Runner Persistence Models and DB Helpers

**Files:**
- Modify: `src/simple_agent/state/state.py`
- Modify: `src/simple_agent/db/db.py`
- Create: `tests/test_runner_storage.py`

- [ ] **Step 1: Write failing storage tests**

Create `tests/test_runner_storage.py`:

```python
"""Tests for session runner persistence helpers."""

from __future__ import annotations

from pi.ai.types import AssistantMessage, TextContent

from simple_agent.db.db import Database


def test_runner_state_metadata_roundtrip(tmp_path):
    db = Database(str(tmp_path / "session.db"))

    db.upsert_runner_state_metadata(
        session_id="session_a",
        phase="running",
        status="running",
        active_user_task_id=42,
        last_error=None,
    )

    record = db.get_runner_state_metadata("session_a")

    assert record is not None
    assert record.session_id == "session_a"
    assert record.phase == "running"
    assert record.status == "running"
    assert record.active_user_task_id == 42
    assert record.last_error is None
    assert record.version == 1


def test_runner_messages_append_and_load_in_order(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    msg1 = AssistantMessage(role="assistant", content=[TextContent(text="one")])
    msg2 = AssistantMessage(role="assistant", content=[TextContent(text="two")])

    db.append_runner_messages("session_a", [msg1, msg2])

    messages = db.list_runner_messages("session_a")

    assert [m.content[0].text for m in messages] == ["one", "two"]


def test_runner_tool_call_roundtrip_success(tmp_path):
    db = Database(str(tmp_path / "session.db"))

    row_id = db.insert_runner_tool_call(
        session_id="session_a",
        tool_call_id="call_1",
        tool_name="example",
        params={"value": 1},
        result={"content": "ok"},
        status="success",
        started_at=10.0,
        finished_at=11.0,
        error=None,
    )

    records = db.list_runner_tool_calls("session_a")

    assert row_id == 0
    assert len(records) == 1
    assert records[0].tool_call_id == "call_1"
    assert records[0].tool_name == "example"
    assert records[0].params_json == '{"value": 1}'
    assert records[0].result_json == '{"content": "ok"}'
    assert records[0].status == "success"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner_storage.py -q`

Expected: FAIL with missing methods or missing imports such as `AttributeError: 'Database' object has no attribute 'upsert_runner_state_metadata'`.

- [ ] **Step 3: Add runner records and message helpers**

In `src/simple_agent/state/state.py`, add these imports near the existing imports:

```python
from typing import Literal
```

Add these record classes after `ManagedTaskRecord`:

```python
class RunnerStateMetadataRecord(SQLModel, table=True):
    """SQLite model for session-runner lifecycle metadata."""

    session_id: str = Field(primary_key=True)
    phase: str = Field(default="idle", index=True)
    status: str = Field(default="idle", index=True)
    active_user_task_id: int | None = Field(default=None, index=True)
    last_error: str | None = None
    version: int = Field(default=1)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class RunnerMessageRecord(SQLModel, table=True):
    """SQLite model for ordered session-runner messages."""

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    seq: int = Field(index=True)
    role: str = Field(index=True)
    content_json: str
    timestamp_ms: int | None = Field(default=None)
    created_at: float = Field(default_factory=time.time)


class RunnerToolCallRecord(SQLModel, table=True):
    """SQLite model for structured session-runner tool execution logs."""

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    tool_call_id: str = Field(index=True)
    tool_name: str = Field(index=True)
    params_json: str
    result_json: str | None = None
    status: str = Field(index=True)
    started_at: float
    finished_at: float | None = None
    error: str | None = None
```

Add these helpers near the existing adapters:

```python
_single_message_adapter = TypeAdapter(AgentMessage)


def agent_message_to_json(message: AgentMessage) -> str:
    return _single_message_adapter.dump_json(message).decode("utf-8")


def agent_message_from_json(payload: str) -> AgentMessage:
    return _single_message_adapter.validate_json(payload)
```

- [ ] **Step 4: Add DB helper imports**

In `src/simple_agent/db/db.py`, extend the import from `simple_agent.state.state`:

```python
from simple_agent.state.state import (
    ManagedTaskRecord,
    RunnerMessageRecord,
    RunnerStateMetadataRecord,
    RunnerToolCallRecord,
    SessionRecord,
    TaskRecord,
    ToolCallRecord,
    ToolExecMessage,
    agent_message_from_json,
    agent_message_to_json,
    managed_task_from_record,
    managed_task_to_record,
)
```

Add imports near the top:

```python
from pi.agent.types import AgentMessage
```

- [ ] **Step 5: Add DB runner metadata helpers**

In `src/simple_agent/db/db.py`, add this section before session metadata operations:

```python
    # ------------------------------------------------------------------
    # Runner state operations
    # ------------------------------------------------------------------

    @standalone_or_compose
    def upsert_runner_state_metadata(
        self,
        session_id: str,
        *,
        phase: str,
        status: str,
        active_user_task_id: int | None = None,
        last_error: str | None = None,
        session: Session | None = None,
    ) -> None:
        record = session.get(RunnerStateMetadataRecord, session_id)
        now = time.time()
        if record is None:
            record = RunnerStateMetadataRecord(session_id=session_id, created_at=now)
            session.add(record)
        record.phase = phase
        record.status = status
        record.active_user_task_id = active_user_task_id
        record.last_error = last_error
        record.updated_at = now

    @standalone_or_compose
    def get_runner_state_metadata(
        self,
        session_id: str,
        *,
        session: Session | None = None,
    ) -> RunnerStateMetadataRecord | None:
        record = session.get(RunnerStateMetadataRecord, session_id)
        if record is not None:
            session.expunge(record)
        return record
```

- [ ] **Step 6: Add DB runner message helpers**

In the same runner state operations section, add:

```python
    @standalone_or_compose
    def next_runner_message_seq(
        self,
        session_id: str,
        *,
        session: Session | None = None,
    ) -> int:
        record = session.exec(
            select(RunnerMessageRecord)
            .where(RunnerMessageRecord.session_id == session_id)
            .order_by(RunnerMessageRecord.seq.desc())
        ).first()
        return (record.seq + 1) if record else 0

    @standalone_or_compose
    def append_runner_messages(
        self,
        session_id: str,
        messages: list[AgentMessage],
        *,
        session: Session | None = None,
    ) -> None:
        seq = self.next_runner_message_seq(session_id, session=session)
        for message in messages:
            record = RunnerMessageRecord(
                session_id=session_id,
                seq=seq,
                role=message.role,
                content_json=agent_message_to_json(message),
                timestamp_ms=getattr(message, "timestamp", None),
            )
            session.add(record)
            seq += 1

    @standalone_or_compose
    def list_runner_messages(
        self,
        session_id: str,
        *,
        session: Session | None = None,
    ) -> list[AgentMessage]:
        records = list(
            session.exec(
                select(RunnerMessageRecord)
                .where(RunnerMessageRecord.session_id == session_id)
                .order_by(RunnerMessageRecord.seq)
            ).all()
        )
        return [agent_message_from_json(record.content_json) for record in records]
```

- [ ] **Step 7: Add DB runner tool-call helpers**

In the same runner state operations section, add:

```python
    @standalone_or_compose
    def next_runner_tool_call_id(
        self,
        session_id: str,
        *,
        session: Session | None = None,
    ) -> int:
        record = session.exec(
            select(RunnerToolCallRecord)
            .where(RunnerToolCallRecord.session_id == session_id)
            .order_by(RunnerToolCallRecord.id.desc())
        ).first()
        return (record.id + 1) if record else 0

    @standalone_or_compose
    def insert_runner_tool_call(
        self,
        *,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        params: dict,
        result: dict | None,
        status: str,
        started_at: float,
        finished_at: float | None,
        error: str | None,
        session: Session | None = None,
    ) -> int:
        next_id = self.next_runner_tool_call_id(session_id, session=session)
        record = RunnerToolCallRecord(
            id=next_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            params_json=json.dumps(params, sort_keys=True),
            result_json=json.dumps(result, sort_keys=True) if result is not None else None,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            error=error,
        )
        session.add(record)
        return next_id

    @standalone_or_compose
    def list_runner_tool_calls(
        self,
        session_id: str,
        *,
        session: Session | None = None,
    ) -> list[RunnerToolCallRecord]:
        records = list(
            session.exec(
                select(RunnerToolCallRecord)
                .where(RunnerToolCallRecord.session_id == session_id)
                .order_by(RunnerToolCallRecord.id)
            ).all()
        )
        for record in records:
            session.expunge(record)
        return records
```

- [ ] **Step 8: Run storage tests**

Run: `uv run pytest tests/test_runner_storage.py -q`

Expected: PASS.

- [ ] **Step 9: Commit storage work**

```bash
git add src/simple_agent/state/state.py src/simple_agent/db/db.py tests/test_runner_storage.py
git commit -m "feat: add session runner persistence"
```

---

### Task 2: Extend ToolExecutionLogger for Runner Tool Calls

**Files:**
- Modify: `src/simple_agent/tool/execution_logger.py`
- Create: `tests/test_execution_logger.py`

- [ ] **Step 1: Write failing logger tests**

Create `tests/test_execution_logger.py`:

```python
"""Tests for ToolExecutionLogger."""

from __future__ import annotations

import pytest

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import TextContent

from simple_agent.db.db import Database
from simple_agent.tool.execution_logger import ToolExecutionLogger


@pytest.mark.asyncio
async def test_wrap_tool_records_runner_tool_call_success(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    tool = AgentTool(name="example", description="Example", parameters={"type": "object", "properties": {}})

    async def execute(tool_call_id, params, cancel_event=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="hello")])

    tool.execute = execute
    logger = ToolExecutionLogger(db, session_id="session_a")
    wrapped = logger.wrap_tool(tool)

    await wrapped.execute("call_1", {"name": "Ada"})

    records = db.list_runner_tool_calls("session_a")
    assert len(records) == 1
    assert records[0].tool_call_id == "call_1"
    assert records[0].tool_name == "example"
    assert records[0].params_json == '{"name": "Ada"}'
    assert records[0].status == "success"
    assert records[0].error is None


@pytest.mark.asyncio
async def test_wrap_tool_records_runner_tool_call_error_and_reraises(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    tool = AgentTool(name="explode", description="Explode", parameters={"type": "object", "properties": {}})

    async def execute(tool_call_id, params, cancel_event=None, on_update=None):
        raise RuntimeError("boom")

    tool.execute = execute
    logger = ToolExecutionLogger(db, session_id="session_a")
    wrapped = logger.wrap_tool(tool)

    with pytest.raises(RuntimeError, match="boom"):
        await wrapped.execute("call_2", {"x": 1})

    records = db.list_runner_tool_calls("session_a")
    assert len(records) == 1
    assert records[0].tool_call_id == "call_2"
    assert records[0].tool_name == "explode"
    assert records[0].status == "error"
    assert records[0].error == "boom"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_execution_logger.py -q`

Expected: FAIL because `ToolExecutionLogger.__init__` does not accept `session_id`.

- [ ] **Step 3: Update ToolExecutionLogger constructor**

In `src/simple_agent/tool/execution_logger.py`, update `__init__`:

```python
    def __init__(self, db: Database | None = None, task_manager=None, session_id: str | None = None):
        self._db = db or Database()
        self._task_manager = task_manager
        self._session_id = session_id
```

- [ ] **Step 4: Add helper to serialize tool results**

In `src/simple_agent/tool/execution_logger.py`, add:

```python
def _tool_result_payload(result: AgentToolResult) -> dict[str, Any]:
    return result.model_dump(mode="json")
```

- [ ] **Step 5: Wrap execution with success/error runner logging**

Replace the body of `wrap_tool`'s inner `execute` function with:

```python
            started_at = time.time()
            try:
                result = await original(tool_call_id, params, cancel_event, on_update)
            except Exception as exc:
                if self._session_id is not None:
                    self._db.insert_runner_tool_call(
                        session_id=self._session_id,
                        tool_call_id=tool_call_id,
                        tool_name=tool.name,
                        params=params,
                        result=None,
                        status="error",
                        started_at=started_at,
                        finished_at=time.time(),
                        error=str(exc),
                    )
                raise

            raw_output = result.content[0].text
            if self._session_id is not None:
                self._db.insert_runner_tool_call(
                    session_id=self._session_id,
                    tool_call_id=tool_call_id,
                    tool_name=tool.name,
                    params=params,
                    result=_tool_result_payload(result),
                    status="success",
                    started_at=started_at,
                    finished_at=time.time(),
                    error=None,
                )

            next_id = self._db.next_tool_call_id()
            result = _format_tool_result(next_id, result)
            tool_exec = ToolExecMessage(
                tool_call=ToolCall(id=tool_call_id, arguments=params, name=tool.name),
                raw_output=raw_output,
                tool_result=result,
            )
            log_id = self._db.insert_tool_call(tool_exec)
            if self._task_manager is not None:
                self._task_manager.record_tool_call(log_id)
            return result
```

Add `import time` at the top of `execution_logger.py`.

- [ ] **Step 6: Run logger tests**

Run: `uv run pytest tests/test_execution_logger.py -q`

Expected: PASS.

- [ ] **Step 7: Run existing task tool tests**

Run: `uv run pytest tests/test_task_tools.py -q`

Expected: PASS.

- [ ] **Step 8: Commit logger work**

```bash
git add src/simple_agent/tool/execution_logger.py tests/test_execution_logger.py
git commit -m "feat: log runner tool executions"
```

---

### Task 3: Simplify AgentProcess

**Files:**
- Modify: `src/simple_agent/process/agent_process.py`
- Create: `tests/test_agent_process.py`
- Modify: `src/simple_agent/process/explore_runner.py`

- [ ] **Step 1: Write failing AgentProcess contract test**

Create `tests/test_agent_process.py`:

```python
"""Tests for AgentProcess executor contract."""

from __future__ import annotations

import asyncio

import pytest

from pi.ai.types import AssistantMessage, TextContent

from simple_agent.process.agent_process import AgentProcess


@pytest.mark.asyncio
async def test_agent_process_run_returns_messages_without_agent_state(monkeypatch):
    message = AssistantMessage(role="assistant", content=[TextContent(text="done")])

    class FakeStream:
        _background_task = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            from pi.agent.types import AgentEndEvent
            if getattr(self, "_sent", False):
                raise StopAsyncIteration
            self._sent = True
            return AgentEndEvent(messages=[message])

    captured = {}

    def fake_agent_loop(input_messages, context, loop_config, cancel_event=None):
        captured["cancel_event"] = cancel_event
        captured["tools"] = context.tools
        return FakeStream()

    monkeypatch.setattr("simple_agent.process.agent_process.agent_loop", fake_agent_loop)

    cancel_event = asyncio.Event()
    process = AgentProcess(model=object())
    result = await process.run(
        system_prompt="system",
        messages=[],
        tools=[],
        user_prompt="hello",
        cancel_event=cancel_event,
    )

    assert result == [message]
    assert captured["cancel_event"] is cancel_event
    assert captured["tools"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_process.py -q`

Expected: FAIL because `AgentProcess.run` still requires `state`.

- [ ] **Step 3: Remove AgentState class from agent_process.py**

In `src/simple_agent/process/agent_process.py`, delete the `AgentState` class and remove unused imports:

```python
from typing import Callable
```

Keep these imports:

```python
import asyncio
import logging
import time
from typing import Callable

from pi.agent import AgentTool
from pi.agent.loop import agent_loop
from pi.agent.types import AgentContext, AgentEndEvent, AgentLoopConfig, AgentMessage
from pi.ai.types import UserMessage, TextContent
```

- [ ] **Step 4: Update AgentProcess.__init__**

Remove the stored state line:

```python
        self.state: AgentState = AgentState()
```

- [ ] **Step 5: Update AgentProcess.run signature and body**

Replace `AgentProcess.run` with:

```python
    @logged(_log)
    async def run(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        user_prompt: str = "",
        cancel_event: asyncio.Event | None = None,
    ) -> list[AgentMessage]:
        """Execute a single agent run and return the new messages."""
        now_ms = int(time.time() * 1000)
        input_messages = []
        if user_prompt:
            input_messages.append(UserMessage(content=[TextContent(text=user_prompt)], timestamp=now_ms))

        context = AgentContext(
            system_prompt=system_prompt,
            messages=list(messages),
            tools=tools,
        )
        loop_config = AgentLoopConfig(
            model=self._model,
            convert_to_llm=lambda msgs: [m for m in msgs if m.role in ("user", "assistant", "tool_result")],
            get_api_key=self._api_key,
        )

        stream = agent_loop(
            input_messages,
            context,
            loop_config,
            cancel_event=cancel_event,
        )

        new_messages: list[AgentMessage] = []
        async for event in stream:
            if isinstance(event, AgentEndEvent):
                new_messages = event.messages
            self._emit(event)

        if stream._background_task is not None:
            exc = stream._background_task.exception()
            if exc is not None:
                raise exc

        return new_messages
```

- [ ] **Step 6: Compatibility update ExploreRunner imports**

In `src/simple_agent/process/explore_runner.py`, remove `AgentState` from the import:

```python
from simple_agent.process.agent_process import AgentProcess
```

This file will still fail until Task 7 compatibility is applied. Commit AgentProcess after its focused test passes.

- [ ] **Step 7: Run AgentProcess focused test**

Run: `uv run pytest tests/test_agent_process.py -q`

Expected: PASS.

- [ ] **Step 8: Commit AgentProcess contract**

```bash
git add src/simple_agent/process/agent_process.py tests/test_agent_process.py
git commit -m "refactor: simplify agent process contract"
```

---

### Task 4: Add SessionRunner

**Files:**
- Create: `src/simple_agent/session/runner.py`
- Create: `tests/test_session_runner.py`

- [ ] **Step 1: Write failing SessionRunner happy-path test**

Create `tests/test_session_runner.py`:

```python
"""Tests for SessionRunner."""

from __future__ import annotations

import asyncio

import pytest

from pi.ai.types import AssistantMessage, TextContent

from simple_agent.db.db import Database
from simple_agent.session.runner import SessionRunner
from simple_agent.task_manager import TaskManager
from simple_agent.tool.execution_logger import ToolExecutionLogger


class FakeAgentProcess:
    def __init__(self):
        self.calls = []

    async def run(self, system_prompt, messages, tools, user_prompt="", cancel_event=None):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": [tool.name for tool in tools],
                "user_prompt": user_prompt,
                "cancel_event": cancel_event,
            }
        )
        return [AssistantMessage(role="assistant", content=[TextContent(text="done")])]


@pytest.mark.asyncio
async def test_session_runner_creates_task_runs_agent_and_persists_messages(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    execution_logger = ToolExecutionLogger(db, task_manager=task_manager, session_id="session_a")
    agent_process = FakeAgentProcess()
    cancel_event = asyncio.Event()
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        execution_logger=execution_logger,
        agent_process=agent_process,
        cancel_event=cancel_event,
    )

    result = await runner.run("Build feature")

    assert result.kind == "user_task"
    assert result.title == "Build feature"
    assert result.status == "done"
    assert len(agent_process.calls) == 1
    assert agent_process.calls[0]["messages"] == []
    assert agent_process.calls[0]["user_prompt"] == "Build feature"
    assert agent_process.calls[0]["cancel_event"] is cancel_event
    assert "create_todo" in agent_process.calls[0]["tools"]
    assert "finish_todo" in agent_process.calls[0]["tools"]
    assert "error_todo" in agent_process.calls[0]["tools"]

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.phase == "done"
    assert metadata.status == "done"
    assert metadata.active_user_task_id == result.id
    assert db.list_runner_messages("session_a")[0].content[0].text == "done"
```

- [ ] **Step 2: Write failing SessionRunner error test**

Append to `tests/test_session_runner.py`:

```python
class FailingAgentProcess:
    async def run(self, system_prompt, messages, tools, user_prompt="", cancel_event=None):
        raise RuntimeError("agent failed")


@pytest.mark.asyncio
async def test_session_runner_persists_error_and_reraises(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    task_manager = TaskManager(db)
    execution_logger = ToolExecutionLogger(db, task_manager=task_manager, session_id="session_a")
    runner = SessionRunner(
        session_id="session_a",
        db=db,
        task_manager=task_manager,
        execution_logger=execution_logger,
        agent_process=FailingAgentProcess(),
        cancel_event=asyncio.Event(),
    )

    with pytest.raises(RuntimeError, match="agent failed"):
        await runner.run("Build feature")

    metadata = db.get_runner_state_metadata("session_a")
    assert metadata.phase == "error"
    assert metadata.status == "error"
    assert metadata.last_error == "agent failed"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_session_runner.py -q`

Expected: FAIL because `simple_agent.session.runner` does not exist.

- [ ] **Step 4: Implement SessionRunner skeleton and constructor**

Create `src/simple_agent/session/runner.py`:

```python
"""SessionRunner owns the persisted Session.run workflow."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from simple_agent.tool.common_tools import create_all_coding_tools

if TYPE_CHECKING:
    from simple_agent.db.db import Database
    from simple_agent.process.agent_process import AgentProcess
    from simple_agent.task_manager import TaskManager
    from simple_agent.tool.execution_logger import ToolExecutionLogger

_log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful coding agent.

Use create_todo before starting a coherent unit of work.
Call finish_todo when the active todo is complete.
Call error_todo if the active todo cannot be completed.
Keep responses concise and use available tools to do the work.
"""


class SessionRunner:
    """Persisted runner for one Session.run invocation at a time."""

    def __init__(
        self,
        *,
        session_id: str,
        db: Database,
        task_manager: TaskManager,
        execution_logger: ToolExecutionLogger,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event,
    ):
        self._session_id = session_id
        self._db = db
        self._task_manager = task_manager
        self._execution_logger = execution_logger
        self._agent_process = agent_process
        self._cancel_event = cancel_event
        self._phase = "idle"
        self._active_user_task_id: int | None = None
```

- [ ] **Step 5: Implement metadata load/checkpoint helpers**

Add to `SessionRunner`:

```python
    def load(self) -> None:
        metadata = self._db.get_runner_state_metadata(self._session_id)
        if metadata is None:
            self._phase = "idle"
            self._active_user_task_id = None
            return
        self._phase = metadata.phase
        self._active_user_task_id = metadata.active_user_task_id
        self._task_manager.active_user_task_id = metadata.active_user_task_id

    def checkpoint(self, *, status: str | None = None, last_error: str | None = None) -> None:
        self._db.upsert_runner_state_metadata(
            self._session_id,
            phase=self._phase,
            status=status or self._phase,
            active_user_task_id=self._active_user_task_id,
            last_error=last_error,
        )
```

- [ ] **Step 6: Implement tool assembly**

Add to `SessionRunner`:

```python
    def _create_tools(self):
        tools = [
            self._task_manager.create_create_todo_tool(),
            self._task_manager.create_finish_todo_tool(),
            self._task_manager.create_error_todo_tool(),
            *create_all_coding_tools("."),
        ]
        return self._execution_logger.wrap_tools(tools)
```

- [ ] **Step 7: Implement handlers and run loop**

Add to `SessionRunner`:

```python
    async def run(self, user_input: str):
        self.load()
        try:
            while self._phase != "done":
                if self._phase in ("idle", "done", "error"):
                    await self.handle_idle(user_input)
                    continue
                if self._phase == "running":
                    await self.handle_running(user_input)
                    continue
                raise RuntimeError(f"Unknown runner phase: {self._phase}")
        except Exception as exc:
            self.handle_error(exc)
            raise

        if self._active_user_task_id is None:
            return None
        return self._db.get_managed_task(self._active_user_task_id)

    async def handle_idle(self, user_input: str) -> None:
        user_task = self._task_manager.create_user_task(user_input)
        self._active_user_task_id = user_task.id
        self._phase = "running"
        self.checkpoint(status="running")

    async def handle_running(self, user_input: str) -> None:
        messages = self._db.list_runner_messages(self._session_id)
        new_messages = await self._agent_process.run(
            system_prompt=SYSTEM_PROMPT,
            messages=messages,
            tools=self._create_tools(),
            user_prompt=user_input,
            cancel_event=self._cancel_event,
        )
        self._db.append_runner_messages(self._session_id, new_messages)
        if self._task_manager.active_todo_id is None:
            self._task_manager.finish_user_task()
        self._phase = "done"
        self.checkpoint(status="done")

    def handle_error(self, exc: Exception) -> None:
        _log.exception("session runner failed: session=%s", self._session_id)
        self._phase = "error"
        self.checkpoint(status="error", last_error=str(exc))
```

- [ ] **Step 8: Run SessionRunner tests**

Run: `uv run pytest tests/test_session_runner.py -q`

Expected: PASS.

- [ ] **Step 9: Commit SessionRunner**

```bash
git add src/simple_agent/session/runner.py tests/test_session_runner.py
git commit -m "feat: add persisted session runner"
```

---

### Task 5: Delegate Session.run to SessionRunner

**Files:**
- Modify: `src/simple_agent/session/session.py`
- Modify: `tests/test_session.py`

- [ ] **Step 1: Update Session.run test for new AgentProcess signature**

In `tests/test_session.py`, replace `test_session_run_creates_user_task_and_calls_agent_once` fake run with:

```python
    async def fake_run(self, system_prompt, messages, tools, user_prompt="", cancel_event=None):
        calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": [tool.name for tool in tools],
                "user_prompt": user_prompt,
                "cancel_event": cancel_event,
            }
        )
        from pi.ai.types import AssistantMessage, TextContent
        return [AssistantMessage(role="assistant", content=[TextContent(text="done")])]
```

Add assertion:

```python
    assert calls[0]["cancel_event"] is session._cancel_event
```

- [ ] **Step 2: Run session test to verify failure**

Run: `uv run pytest tests/test_session.py::test_session_run_creates_user_task_and_calls_agent_once -q`

Expected: FAIL because `Session.run` still passes `state=`.

- [ ] **Step 3: Update session imports**

In `src/simple_agent/session/session.py`, remove:

```python
from simple_agent.process.agent_process import AgentProcess, AgentState
from simple_agent.tool.common_tools import create_all_coding_tools
```

Add:

```python
from simple_agent.process.agent_process import AgentProcess
from simple_agent.session.runner import SessionRunner
```

- [ ] **Step 4: Pass session_id into ToolExecutionLogger**

In `Session.__init__`, replace:

```python
        self._execution_logger = ToolExecutionLogger(self._db, task_manager=self._task_manager)
```

with:

```python
        self._execution_logger = ToolExecutionLogger(
            self._db,
            task_manager=self._task_manager,
            session_id=self._id,
        )
```

- [ ] **Step 5: Add SessionRunner factory**

In `Session`, add:

```python
    def _create_runner(self) -> SessionRunner:
        return SessionRunner(
            session_id=self._id,
            db=self._db,
            task_manager=self._task_manager,
            execution_logger=self._execution_logger,
            agent_process=self._agent_process,
            cancel_event=self._cancel_event,
        )
```

- [ ] **Step 6: Replace Session.run internals**

Replace `Session.run` with:

```python
    @logged(_log)
    async def run(self, user_input: str):
        """Run the persisted session runner for one user task."""
        self._running = True
        if self.event_queue is None:
            self.event_queue = asyncio.Queue()

        try:
            runner = self._create_runner()
            user_task = await runner.run(user_input)
            self._cursor_id = user_task.id if user_task is not None else None
            self._checkpoint()
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

        _log.info("run: session=%s done, result=%s", self._id, self._cursor_id)
        return user_task
```

- [ ] **Step 7: Run focused session tests**

Run: `uv run pytest tests/test_session.py -q`

Expected: PASS.

- [ ] **Step 8: Commit Session delegation**

```bash
git add src/simple_agent/session/session.py tests/test_session.py
git commit -m "refactor: delegate session run to runner"
```

---

### Task 6: Compatibility Updates for ExploreRunner

**Files:**
- Modify: `src/simple_agent/process/explore_runner.py`
- Delete references to `AgentState` from `src/simple_agent/process/agent_process.py` already done in Task 3.

- [ ] **Step 1: Inspect current ExploreRunner state-tool behavior**

Run: `sed -n '1,180p' src/simple_agent/process/explore_runner.py`

Expected: identify usages of `AgentState`, `state.tool_results`, and `state.new_messages`.

- [ ] **Step 2: Add a local record-state helper to ExploreRunner**

In `src/simple_agent/process/explore_runner.py`, add imports:

```python
import asyncio
from typing import Any

from pi.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi.ai.types import TextContent
```

Add this helper class near the prompts:

```python
class _RecordState(asyncio.Event):
    def __init__(self):
        super().__init__()
        self.tool_results: dict[str, list] = {}
        self.stop_on_tool: str | None = None

    def is_set(self) -> bool:
        if self.stop_on_tool is not None and self.stop_on_tool in self.tool_results:
            return True
        return super().is_set()

    def create_record_tool(self, model_class: type, name: str, description: str, parameters: dict[str, Any]) -> AgentTool:
        tool = AgentTool(name=name, description=description, parameters=parameters)

        async def execute(
            tool_call_id: str,
            params: dict[str, Any],
            cancel_event: asyncio.Event | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            try:
                item = model_class.model_validate(params)
                self.tool_results.setdefault(name, []).append(item)
                return AgentToolResult(content=[TextContent(text="ok")])
            except Exception as exc:
                return AgentToolResult(content=[TextContent(text=f"validation failed: {exc}")])

        tool.execute = execute
        return tool

    def create_determine_state_tool(self) -> AgentTool:
        from simple_agent.state.state import StateClarification

        return self.create_record_tool(
            model_class=StateClarification,
            name="determine_state",
            description="Determine the current state based on context.",
            parameters={
                "type": "object",
                "properties": {
                    "state": {"type": "string", "enum": ["finished", "error"]},
                    "reason": {"type": "string", "description": "Reason for choosing this state"},
                },
                "required": ["state", "reason"],
            },
        )

    def create_record_textresult_tool(self) -> AgentTool:
        from simple_agent.state.state import TEXT_RESULT_JSON_SCHEMA, TextResult

        return self.create_record_tool(
            model_class=TextResult,
            name="record_textresult",
            description="Record a TextResult instance capturing a final outcome.",
            parameters=TEXT_RESULT_JSON_SCHEMA,
        )
```

- [ ] **Step 3: Update ExploreRunner execute phase**

In `_execute`, replace `state = AgentState()` and `state.stop_condition = ...` with:

```python
        state = _RecordState()
        state.stop_on_tool = "determine_state"
```

Replace the agent call with:

```python
        new_messages = await self._agent_process.run(
            system_prompt=EXECUTE_SYSTEM_PROMPT,
            messages=task.metadata["context_msgs"],
            tools=tools,
            user_prompt=task.input,
            cancel_event=state,
        )
        task.messages.extend(new_messages)
```

- [ ] **Step 4: Update ExploreRunner collect phase**

In `_collect`, replace `collect_state = AgentState()` with:

```python
        collect_state = _RecordState()
```

Replace the agent call with:

```python
        new_messages = await self._agent_process.run(
            system_prompt=COLLECT_SYSTEM_PROMPT,
            messages=task.metadata["context_msgs"],
            tools=collect_tools,
            cancel_event=collect_state,
        )
        task.metadata["context_msgs"].extend(new_messages)
        task.messages.extend(new_messages)
```

- [ ] **Step 5: Run existing process/session tests**

Run: `uv run pytest tests/test_session.py tests/test_session_manager.py -q`

Expected: PASS.

- [ ] **Step 6: Commit compatibility updates**

```bash
git add src/simple_agent/process/explore_runner.py
git commit -m "refactor: adapt explore runner to simple agent process"
```

---

### Task 7: Full Focused Verification

**Files:**
- No code files unless verification exposes a real bug.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
uv run pytest \
  tests/test_runner_storage.py \
  tests/test_execution_logger.py \
  tests/test_agent_process.py \
  tests/test_session_runner.py \
  tests/test_session.py \
  tests/test_session_manager.py \
  tests/test_task_manager.py \
  tests/test_task_tools.py \
  tests/test_diff_tool.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Search for removed AgentState usage**

Run: `rg -n "AgentState|state=state|state=" src tests`

Expected: no `AgentState` imports or `AgentProcess.run(... state=...)` call sites remain. Ignore unrelated test variable names if they do not pass state into `AgentProcess.run`.

- [ ] **Step 3: Search for unwrapped session coding tools**

Run: `rg -n "create_all_coding_tools\\(|wrap_tools\\(" src/simple_agent/session src/simple_agent/tool src/simple_agent/process`

Expected: session-run tools are assembled raw and wrapped centrally by `ToolExecutionLogger.wrap_tools`.

- [ ] **Step 4: Commit verification fixes if needed**

If Step 1, 2, or 3 required code fixes, stage the exact files changed by those fixes. For example, if the compatibility pass found one remaining `AgentProcess.run(... state=...)` call in `src/simple_agent/session/session.py`, run:

```bash
git add src/simple_agent/session/session.py
git commit -m "fix: complete session runner refactor"
```

If no fixes were needed, do not create an empty commit.

---

## Self-Review

Spec coverage:

- Session-run-only scope: covered by `SessionRunner` and compatibility-only ExploreRunner task.
- One database file with task data, runner metadata, messages, and tool-call log: covered by Task 1 and Task 4.
- Minimal phases `idle`, `running`, `done`, `error`: covered by Task 4.
- `AgentProcess` takes cancel event and returns only messages: covered by Task 3.
- Tool-call log written by execution wrapper: covered by Task 2.
- Coding tools raw and centrally wrapped when initialized in session runner: covered by Task 4 and Task 7.

Placeholder scan:

- The plan contains no `TBD`, `TODO`, or deferred implementation placeholders.

Type consistency:

- `Database` helper names in tests match helper names introduced in Task 1.
- `SessionRunner` constructor names match the call from `Session._create_runner`.
- `AgentProcess.run` signature matches the fake run signatures in tests.
