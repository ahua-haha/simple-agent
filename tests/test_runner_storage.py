"""Tests for session runner persistence helpers."""

from __future__ import annotations

import json
import sqlite3

from pi.ai.types import AssistantMessage, TextContent

from simple_agent.db.db import Database
from simple_agent.task_manager.models import CommonTask


def test_runner_state_metadata_roundtrip(tmp_path):
    db = Database(str(tmp_path / "session.db"))

    db.upsert_runner_state_metadata(
        session_id="session_a",
        workspace_dir="/workspace",
        active_user_task_id=42,
        last_error=None,
    )
    db.upsert_runner_state_metadata(
        session_id="session_a",
        active_user_task_id=43,
        last_error="boom",
    )

    record = db.get_runner_state_metadata("session_a")

    assert record is not None
    assert record.session_id == "session_a"
    assert record.workspace_dir == "/workspace"
    assert record.active_user_task_id == 43
    assert record.last_error == "boom"
    assert record.version == 1


def test_runner_messages_insert_and_load_in_append_order(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    msg1 = AssistantMessage(role="assistant", content=[TextContent(text="one")])
    msg2 = AssistantMessage(role="assistant", content=[TextContent(text="two")])

    id1 = db.insert_runner_message("session_a", msg1)
    id2 = db.insert_runner_message("session_a", msg2)

    messages = db.list_runner_messages("session_a")

    assert id1 != id2
    assert [m.content[0].text for m in messages] == ["one", "two"]


def test_runner_message_record_uses_seq_for_order_and_id_for_identity(tmp_path):
    db_path = tmp_path / "session.db"
    db = Database(str(db_path))
    msg1 = AssistantMessage(role="assistant", content=[TextContent(text="one")])
    msg2 = AssistantMessage(role="assistant", content=[TextContent(text="two")])

    stable_id = db.insert_runner_message("session_a", msg1)
    db.insert_runner_message("session_a", msg2)

    with sqlite3.connect(db_path) as conn:
        columns = conn.execute("PRAGMA table_info(runnermessagerecord)").fetchall()
        column_by_name = {column[1]: column for column in columns}
        assert column_by_name["seq"][5] == 1
        assert column_by_name["id"][5] == 0
        conn.execute(
            "UPDATE runnermessagerecord SET seq = 99 WHERE id = ?",
            (stable_id,),
        )

    with sqlite3.connect(db_path) as conn:
        records = conn.execute(
            """
            SELECT id
            FROM runnermessagerecord
            WHERE session_id = ?
            ORDER BY seq
            """,
            ("session_a",),
        ).fetchall()
    messages = db.list_runner_messages("session_a")

    assert [record[0] for record in records] == [stable_id + 1, stable_id]
    assert [message.content[0].text for message in messages] == ["two", "one"]


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


def test_delete_runner_message_seq_range_removes_only_matching_session_rows(tmp_path):
    db_path = tmp_path / "session.db"
    db = Database(str(db_path))
    for message_id, text in [(10, "one"), (40, "two"), (20, "three"), (30, "four")]:
        db.insert_runner_message(
            "session_a",
            AssistantMessage(role="assistant", content=[TextContent(text=text)]),
            id=message_id,
        )
    db.insert_runner_message(
        "session_b",
        AssistantMessage(role="assistant", content=[TextContent(text="other")]),
        id=5,
    )

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT seq, id
            FROM runnermessagerecord
            WHERE session_id = ?
            ORDER BY seq
            """,
            ("session_a",),
        ).fetchall()
    start_seq = rows[1][0]
    end_seq = rows[2][0]

    db.delete_runner_message_seq_range("session_a", start_seq=start_seq, end_seq=end_seq)

    assert [message.content[0].text for message in db.list_runner_messages("session_a")] == ["one", "four"]
    assert [message.content[0].text for message in db.list_runner_messages("session_b")] == ["other"]


def test_next_managed_task_id_uses_highest_existing_id(tmp_path):
    db = Database(str(tmp_path / "session.db"))

    assert db.next_managed_task_id() == 1

    task = CommonTask(title="Build feature")
    task.id = db.upsert_managed_task(task)

    assert task.id == 1
    assert db.next_managed_task_id() == 2


def test_task_record_has_generic_metadata_schema(tmp_path):
    db_path = tmp_path / "session.db"
    Database(str(db_path))

    with sqlite3.connect(db_path) as conn:
        columns = conn.execute("PRAGMA table_info(taskrecord)").fetchall()

    names = {column[1] for column in columns}
    assert names == {"id", "parent_id", "kind", "status", "metadata"}


def test_replace_managed_task_tree_deletes_all_tasks_after_root_id(tmp_path):
    db = Database(str(tmp_path / "session.db"))
    user_task = CommonTask(title="Build feature")
    user_task.id = db.upsert_managed_task(user_task)
    stale_child = CommonTask(title="Old child", parent_id=user_task.id)
    stale_child.id = db.upsert_managed_task(stale_child)
    stale_orphan = CommonTask(title="Old orphan", parent_id=None)
    stale_orphan.id = db.upsert_managed_task(stale_orphan)
    replacement_child = CommonTask(title="New child", parent_id=user_task.id)
    replacement_child.id = stale_child.id
    user_task.children = [replacement_child]

    db.replace_managed_task_tree(user_task)

    assert db.get_managed_task(stale_orphan.id) is None
    assert [child.title for child in db.list_managed_task_children(user_task.id)] == ["New child"]


def test_runner_tool_call_roundtrip_success(tmp_path):
    db = Database(str(tmp_path / "session.db"))

    row_id = db.insert_runner_tool_call(
        session_id="session_a",
        tool_call_id="call_1",
        tool_name="example",
        tool_call_json='{"arguments":{"value":1},"id":"call_1","name":"example"}',
        tool_result_json='{"content":"ok"}',
    )

    records = db.list_runner_tool_calls("session_a")

    assert row_id == 0
    assert len(records) == 1
    assert records[0].tool_call_id == "call_1"
    assert records[0].tool_name == "example"
    assert json.loads(records[0].tool_call_json) == {
        "arguments": {"value": 1},
        "id": "call_1",
        "name": "example",
    }
    assert json.loads(records[0].tool_result_json) == {"content": "ok"}


def test_runner_tool_call_record_has_simplified_columns(tmp_path):
    db_path = tmp_path / "session.db"
    Database(str(db_path))

    with sqlite3.connect(db_path) as conn:
        columns = conn.execute("PRAGMA table_info(runnertoolcallrecord)").fetchall()

    names = {column[1] for column in columns}
    assert {
        "id",
        "session_id",
        "tool_call_id",
        "tool_name",
        "tool_call_json",
        "tool_result_json",
    }.issubset(names)
    assert "params_json" not in names
    assert "result_json" not in names
    assert "status" not in names
    assert "started_at" not in names
    assert "finished_at" not in names
    assert "error" not in names
