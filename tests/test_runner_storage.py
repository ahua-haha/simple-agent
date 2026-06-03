"""Tests for session runner persistence helpers."""

from __future__ import annotations

from pi.ai.types import AssistantMessage, TextContent

from simple_agent.db.db import Database
from simple_agent.task_manager.models import ManagedTask


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

    seqs = db.append_runner_messages("session_a", [msg1, msg2])

    messages = db.list_runner_messages("session_a")

    assert isinstance(seqs[0], str)
    assert seqs[0] < seqs[1]
    assert [m.content[0].text for m in messages] == ["one", "two"]


def test_replace_runner_messages_from_deletes_tail_and_inserts_ordered_seq(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    old = [
        AssistantMessage(role="assistant", content=[TextContent(text="zero")]),
        AssistantMessage(role="assistant", content=[TextContent(text="one")]),
        AssistantMessage(role="assistant", content=[TextContent(text="two")]),
    ]
    new = [AssistantMessage(role="assistant", content=[TextContent(text="compact")])]

    seqs = db.append_runner_messages("session_a", old)
    replacement_seqs = db.replace_runner_messages_from("session_a", seqs[1], new)

    messages = db.list_runner_messages("session_a")
    assert replacement_seqs[0] == seqs[1]
    assert [m.content[0].text for m in messages] == ["zero", "compact"]


def test_next_managed_task_id_uses_highest_existing_id(tmp_path):
    db = Database(str(tmp_path / "session.db"))

    assert db.next_managed_task_id() == 1

    task = ManagedTask(kind="user_task", title="Build feature")
    task.id = db.upsert_managed_task(task)

    assert task.id == 1
    assert db.next_managed_task_id() == 2


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
