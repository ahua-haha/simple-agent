"""Tests for session runner persistence helpers."""

from __future__ import annotations

import sqlite3

from pi.ai.types import AssistantMessage, TextContent

from simple_agent.db.db import Database
from simple_agent.task_manager import TaskManager
from simple_agent.task_manager.models import ManagedTask


def test_runner_state_metadata_roundtrip(tmp_path):
    db = Database(str(tmp_path / "session.db"))

    db.upsert_runner_state_metadata(
        session_id="session_a",
        next_action="normal_run",
        active_user_task_id=42,
        last_error=None,
    )

    record = db.get_runner_state_metadata("session_a")

    assert record is not None
    assert record.session_id == "session_a"
    assert record.next_action == "normal_run"
    assert record.active_user_task_id == 42
    assert record.last_error is None
    assert record.version == 1


def test_runner_messages_insert_and_load_in_append_order(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    msg1 = AssistantMessage(role="assistant", content=[TextContent(text="one")])
    msg2 = AssistantMessage(role="assistant", content=[TextContent(text="two")])

    db.insert_runner_message("session_a", msg1)
    db.insert_runner_message("session_a", msg2)

    messages = db.list_runner_messages("session_a")

    assert [m.content[0].text for m in messages] == ["one", "two"]


def test_replace_runner_messages_rewrites_whole_message_table(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    messages = [
        AssistantMessage(role="assistant", content=[TextContent(text="zero")]),
        AssistantMessage(role="assistant", content=[TextContent(text="one")]),
        AssistantMessage(role="assistant", content=[TextContent(text="two")]),
    ]
    for message in messages:
        db.insert_runner_message("session_a", message)

    db.replace_runner_messages(
        "session_a",
        [
            AssistantMessage(role="assistant", content=[TextContent(text="replacement")]),
            messages[-1],
        ],
    )

    messages = db.list_runner_messages("session_a")
    assert [m.content[0].text for m in messages] == ["replacement", "two"]


def test_next_managed_task_id_uses_highest_existing_id(tmp_path):
    db = Database(str(tmp_path / "session.db"))

    assert db.next_managed_task_id() == 1

    task = ManagedTask(kind="user_task", title="Build feature")
    task.id = db.upsert_managed_task(task)

    assert task.id == 1
    assert db.next_managed_task_id() == 2


def test_managed_task_record_has_no_seq_column(tmp_path):
    db_path = tmp_path / "session.db"
    Database(str(db_path))

    with sqlite3.connect(db_path) as conn:
        columns = conn.execute("PRAGMA table_info(managedtaskrecord)").fetchall()

    assert "seq" not in {column[1] for column in columns}


def test_replace_managed_task_tree_deletes_all_tasks_after_root_id(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    user_task = ManagedTask(kind="user_task", title="Build feature")
    user_task.id = db.upsert_managed_task(user_task)
    stale_child = ManagedTask(kind="todo", title="Old child", parent_id=user_task.id)
    stale_child.id = db.upsert_managed_task(stale_child)
    stale_orphan = ManagedTask(kind="todo", title="Old orphan", parent_id=None)
    stale_orphan.id = db.upsert_managed_task(stale_orphan)
    replacement_child = ManagedTask(kind="todo", title="New child", parent_id=user_task.id)
    replacement_child.id = stale_child.id
    user_task.children = [replacement_child]

    db.replace_managed_task_tree(user_task)

    loaded = TaskManager(db)
    loaded.load(user_task.id)
    assert db.get_managed_task(stale_orphan.id) is None
    assert [child.title for child in loaded.active_user_task.children] == ["New child"]


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
