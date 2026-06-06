"""Tests for session log playback helpers."""

from __future__ import annotations

from simple_agent.cli.session_log_playback import (
    PlaybackContext,
    PlaybackState,
    brief_move,
    build_context,
    format_tool_result_text,
    handle_command,
    tool_result_text,
)
from simple_agent.db.db import Database


def test_brief_move_lists_running_response_with_tool_names():
    move = {
        "event": "handle_running",
        "messages": [{"message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}}],
        "assistant_message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will inspect files"},
                {"type": "tool_call", "id": "call_1", "name": "ls", "arguments": {"path": "."}},
            ],
        },
    }

    assert brief_move(move) == "running - 1 context messages, assistant: I will inspect files tools: ls"


def test_brief_move_lists_compact_scope_and_replacement_count():
    move = {
        "event": "handle_compact_result",
        "message_scope": {"start_message_id": 2, "end_message_id": 7},
        "replacement_messages": [{}, {}, {}],
    }

    assert brief_move(move) == "compact result - scope 2..7 replacement 3 messages"


def test_format_tool_result_text_respects_character_limit():
    state = PlaybackState(tool_result_mode="full", tool_result_limit=5)

    assert format_tool_result_text("abcdefgh", state) == "abcde... [truncated 3 chars]"


def test_format_tool_result_text_can_show_full_content():
    state = PlaybackState(tool_result_mode="full", tool_result_limit=None)

    assert format_tool_result_text("abcdefgh", state) == "abcdefgh"


def test_show_command_accepts_inline_tool_result_limit(capsys):
    move = {
        "event": "handle_running",
        "messages": [],
        "user_instruction_message": {"role": "user", "content": []},
        "assistant_message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        "tool_results": [
            {
                "tool_name": "ls",
                "tool_call_id": "call_1",
                "content": "abcdefgh",
            }
        ],
    }
    state = PlaybackState()

    handle_command("show 1 5", [move], state)

    output = capsys.readouterr().out
    assert "abcde... [truncated 3 chars]" in output
    assert state.tool_result_mode == "none"


def test_build_context_loads_tool_result_json_by_tool_call_id(tmp_path):
    db_path = tmp_path / "session_a.db"
    db = Database(str(db_path))
    db.insert_runner_tool_call(
        session_id="session_a",
        tool_call_id="call_1",
        tool_name="ls",
        tool_call_json="{}",
        tool_result_json='{"content":[{"type":"text","text":"abcdef"}]}',
    )
    moves = [{"session_id": "session_a"}]

    context = build_context(
        moves,
        db_path=db_path,
        sessions_dir=tmp_path,
        log_file=tmp_path / "session_a.jsonl",
    )

    assert context.tool_results == {"call_1": "abcdef"}


def test_show_command_can_display_db_tool_result_content(capsys):
    move = {
        "event": "handle_running",
        "messages": [],
        "user_instruction_message": {"role": "user", "content": []},
        "assistant_message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        "tool_results": [{"tool_name": "ls", "tool_call_id": "call_1"}],
    }
    context = PlaybackContext(tool_results={"call_1": "abcdefgh"})

    handle_command("show 1 5", [move], PlaybackState(), context)

    output = capsys.readouterr().out
    assert "abcde... [truncated 3 chars]" in output


def test_tool_result_text_prefers_log_content_when_present():
    context = PlaybackContext(tool_results={"call_1": "from db"})

    assert tool_result_text({"tool_call_id": "call_1", "content": "from log"}, context) == "from log"


def test_build_context_extracts_string_content_from_tool_result_json(tmp_path):
    db_path = tmp_path / "session_a.db"
    db = Database(str(db_path))
    db.insert_runner_tool_call(
        session_id="session_a",
        tool_call_id="call_1",
        tool_name="ls",
        tool_call_json="{}",
        tool_result_json='{"content":"plain text"}',
    )

    context = build_context(
        [{"session_id": "session_a"}],
        db_path=db_path,
        sessions_dir=tmp_path,
        log_file=tmp_path / "session_a.jsonl",
    )

    assert context.tool_results == {"call_1": "plain text"}
