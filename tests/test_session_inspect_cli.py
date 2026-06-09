"""Tests for the unified session inspection CLI."""

from __future__ import annotations

import os

from simple_agent.db.db import Database
from simple_agent.index import AgentIndex
from simple_agent.cli.session_inspect import (
    InspectState,
    _parse_args,
    discover_sessions,
    handle_repl_command,
)
from simple_agent.task_manager.models import RepoMemoryTask, ToolCallTask, UserTask


def test_session_inspect_parses_repl_directories(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["session-inspect", "--log-dir", "custom-logs", "--sessions-dir", "custom-sessions"],
    )

    args = _parse_args()

    assert args.log_dir == "custom-logs"
    assert args.sessions_dir == "custom-sessions"


def test_session_inspect_opens_repl_without_subcommand(monkeypatch):
    monkeypatch.setattr("sys.argv", ["session-inspect"])

    args = _parse_args()

    assert args.log_dir == "./logs/session_runs"
    assert args.sessions_dir == "./sessions"


def test_discover_sessions_pairs_logs_and_databases(tmp_path):
    log_dir = tmp_path / "logs"
    sessions_dir = tmp_path / "sessions"
    log_dir.mkdir()
    sessions_dir.mkdir()
    log_file = log_dir / "session_a.jsonl"
    db_file = sessions_dir / "session_a.db"
    log_file.write_text('{"session_id":"session_a"}\n', encoding="utf-8")
    db_file.write_text("", encoding="utf-8")
    os.utime(log_file, (1, 1))
    os.utime(db_file, (2, 2))

    sessions = discover_sessions(log_dir=log_dir, sessions_dir=sessions_dir)

    assert len(sessions) == 1
    assert sessions[0].session_id == "session_a"
    assert sessions[0].log_file == log_file
    assert sessions[0].db_file == db_file


def test_repl_can_select_session_and_route_log_command(tmp_path, capsys):
    log_file = tmp_path / "session_a.jsonl"
    log_file.write_text(
        '{"session_id":"session_a","event":"handle_running","messages":[],"assistant_message":{"content":[]}}\n',
        encoding="utf-8",
    )
    state = InspectState()
    state.sessions = discover_sessions(log_dir=tmp_path, sessions_dir=tmp_path)

    handle_repl_command("use 1", state)
    handle_repl_command("list", state)

    output = capsys.readouterr().out
    assert "[selected] 1. session_a" in output
    assert "[moves] count=1" in output


def test_repl_task_command_renders_selected_task_tree(tmp_path, capsys):
    db_path = tmp_path / "session_a.db"
    db = Database(str(db_path))
    db.upsert_managed_task(UserTask(id=1, title="Build feature"))
    db.upsert_managed_task(
        ToolCallTask(
            id=2,
            parent_id=1,
            tool_call_log_id=7,
            tool_call_name="ls",
            tool_call_args={"path": "."},
        )
    )
    state = InspectState(sessions_dir=tmp_path)
    state.sessions = discover_sessions(log_dir=tmp_path, sessions_dir=tmp_path)

    handle_repl_command("use 1", state)
    handle_repl_command("tasks --depth 1", state)

    output = capsys.readouterr().out
    assert "- user_task [active] Build feature" in output
    assert '- tool_call 1. ls args: {"path":"."}' in output


def test_repl_index_list_shows_repo_memory_tasks(tmp_path, capsys):
    db_path = tmp_path / "session_a.db"
    index_db_path = tmp_path / "index.db"
    db = Database(str(db_path))
    db.upsert_managed_task(
        RepoMemoryTask(
            id=1,
            title="Write repo memory",
            repo_path=str(tmp_path),
            index_db_path=str(index_db_path),
        )
    )
    state = InspectState(sessions_dir=tmp_path)
    state.sessions = discover_sessions(log_dir=tmp_path, sessions_dir=tmp_path)

    handle_repl_command("use 1", state)
    handle_repl_command("index list", state)

    output = capsys.readouterr().out
    assert "[index] count=1" in output
    assert "1. Write repo memory" in output
    assert str(index_db_path) in output


def test_repl_index_tree_renders_agent_index(tmp_path, capsys):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    source_file = repo_dir / "app.py"
    source_file.write_text("print('hi')\n", encoding="utf-8")
    index_db_path = tmp_path / "index.db"
    AgentIndex(str(index_db_path), base_dir=str(repo_dir)).upsert_entry(
        "app.py",
        {"description": "Application entrypoint"},
    )
    session_db_path = tmp_path / "session_a.db"
    session_db = Database(str(session_db_path))
    session_db.upsert_managed_task(
        RepoMemoryTask(
            id=1,
            title="Write repo memory",
            repo_path=str(repo_dir),
            index_db_path=str(index_db_path),
        )
    )
    state = InspectState(sessions_dir=tmp_path)
    state.sessions = discover_sessions(log_dir=tmp_path, sessions_dir=tmp_path)

    handle_repl_command("use 1", state)
    handle_repl_command("index use 1", state)
    handle_repl_command("index tree app.py", state)

    output = capsys.readouterr().out
    assert "app.py" in output
    assert "# Application entrypoint" in output


def test_repl_index_tree_accepts_explicit_db_and_repo(tmp_path, capsys):
    repo_dir = tmp_path / "repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)
    app_file = src_dir / "app.py"
    app_file.write_text("print('hi')\n", encoding="utf-8")
    index_db_path = tmp_path / "index.db"
    AgentIndex(str(index_db_path), base_dir=str(repo_dir)).upsert_entry(
        "src/app.py",
        {"description": "Application entry point"},
    )
    session_db_path = tmp_path / "session_a.db"
    Database(str(session_db_path))
    state = InspectState(sessions_dir=tmp_path)
    state.sessions = discover_sessions(log_dir=tmp_path, sessions_dir=tmp_path)

    handle_repl_command("use 1", state)
    handle_repl_command(
        f"index tree src --db {index_db_path} --repo {repo_dir} --depth 2",
        state,
    )

    output = capsys.readouterr().out
    assert "src/" in output
    assert "app.py  # Application entry point" in output
