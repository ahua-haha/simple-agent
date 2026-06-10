"""Unified CLI for inspecting session logs and task trees."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from simple_agent.db.db import Database
from simple_agent.index import AgentIndex
from simple_agent.task_manager.models import RepoMemoryTask
from simple_agent.task_manager.review import TaskTreeRenderer, build_task_tree

ToolResultMode = Literal["none", "full"]


@dataclass
class PlaybackState:
    tool_result_mode: ToolResultMode = "none"
    tool_result_limit: int | None = None


@dataclass
class PlaybackContext:
    tool_results: dict[str, str]


@dataclass
class SessionInfo:
    session_id: str
    log_file: Path | None = None
    db_file: Path | None = None
    updated_at: float = 0.0


@dataclass
class InspectState:
    log_dir: Path = Path("./logs/session_runs")
    sessions_dir: Path = Path("./sessions")
    sessions: list[SessionInfo] = field(default_factory=list)
    selected: SessionInfo | None = None
    playback_state: PlaybackState = field(default_factory=PlaybackState)
    playback_moves: list[dict] = field(default_factory=list)
    playback_context: PlaybackContext = field(default_factory=lambda: PlaybackContext(tool_results={}))
    index_tasks: list[RepoMemoryTask] = field(default_factory=list)
    selected_index_task: RepoMemoryTask | None = None


def main() -> None:
    args = _parse_args()
    run_repl(log_dir=Path(args.log_dir), sessions_dir=Path(args.sessions_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect simple-agent session data")
    parser.add_argument("--log-dir", default="./logs/session_runs", help="Directory containing session JSONL logs")
    parser.add_argument("--sessions-dir", default="./sessions", help="Directory containing session DB files")
    return parser.parse_args()


def discover_sessions(*, log_dir: Path, sessions_dir: Path) -> list[SessionInfo]:
    sessions: dict[str, SessionInfo] = {}
    for log_file in log_dir.glob("*.jsonl"):
        session_id = log_file.stem
        info = sessions.setdefault(session_id, SessionInfo(session_id=session_id))
        info.log_file = log_file
        info.updated_at = max(info.updated_at, log_file.stat().st_mtime)
    for db_file in sessions_dir.glob("*.db"):
        session_id = db_file.stem
        info = sessions.setdefault(session_id, SessionInfo(session_id=session_id))
        info.db_file = db_file
        info.updated_at = max(info.updated_at, db_file.stat().st_mtime)
    return sorted(sessions.values(), key=lambda info: info.updated_at, reverse=True)


def run_repl(*, log_dir: Path, sessions_dir: Path) -> None:
    state = InspectState(log_dir=log_dir, sessions_dir=sessions_dir)
    state.sessions = discover_sessions(log_dir=log_dir, sessions_dir=sessions_dir)
    print_repl_help()
    if state.sessions:
        print_sessions(state)
    while True:
        try:
            command = input("session-inspect> ").strip()
        except EOFError:
            print()
            return
        if not command:
            continue
        if command in {"quit", "exit", "q"}:
            return
        handle_repl_command(command, state)


def handle_repl_command(command: str, state: InspectState) -> None:
    parts = command.split()
    name = parts[0]
    args = parts[1:]
    if name in {"help", "?"}:
        print_repl_help()
    elif name in {"sessions", "session"}:
        refresh_sessions(state)
        print_sessions(state)
    elif name == "use":
        select_session(args, state)
    elif name == "tasks":
        print_selected_task_tree(args, state)
    elif name == "index":
        handle_index_command(args, state)
    elif name in {"list", "ls", "show", "move", "tool"}:
        if not ensure_playback_loaded(state):
            return
        handle_playback_command(command, state.playback_moves, state.playback_state, state.playback_context)
    else:
        print(f"[unknown command] {name}")
        print_repl_help()


def refresh_sessions(state: InspectState) -> None:
    selected_id = state.selected.session_id if state.selected else None
    state.sessions = discover_sessions(log_dir=state.log_dir, sessions_dir=state.sessions_dir)
    if selected_id is not None:
        state.selected = next((session for session in state.sessions if session.session_id == selected_id), None)


def print_sessions(state: InspectState) -> None:
    print(f"[sessions] count={len(state.sessions)}")
    for index, session in enumerate(state.sessions, start=1):
        selected = "*" if state.selected and session.session_id == state.selected.session_id else " "
        log = "log" if session.log_file else "no-log"
        db = "db" if session.db_file else "no-db"
        print(f"{selected}{index}. {session.session_id} [{log}, {db}]")


def select_session(args: list[str], state: InspectState) -> None:
    if not args:
        print("[error] use <session-number|session-id>")
        return
    selector = args[0]
    selected = None
    if selector.isdigit():
        index = int(selector)
        if 1 <= index <= len(state.sessions):
            selected = state.sessions[index - 1]
    else:
        selected = next((session for session in state.sessions if session.session_id == selector), None)
    if selected is None:
        print(f"[error] session not found: {selector}")
        return
    state.selected = selected
    state.playback_moves = []
    state.playback_context = PlaybackContext(tool_results={})
    state.index_tasks = []
    state.selected_index_task = None
    print(f"[selected] {state.sessions.index(selected) + 1}. {selected.session_id}")


def ensure_playback_loaded(state: InspectState) -> bool:
    if state.selected is None:
        print("[error] select a session first: use <session-number|session-id>")
        return False
    if state.selected.log_file is None:
        print(f"[error] selected session has no log: {state.selected.session_id}")
        return False
    if state.playback_moves:
        return True
    state.playback_moves = load_moves(state.selected.log_file)
    state.playback_context = build_playback_context(
        state.playback_moves,
        db_path=state.selected.db_file,
        sessions_dir=state.sessions_dir,
        log_file=state.selected.log_file,
    )
    return True


def print_selected_task_tree(args: list[str], state: InspectState) -> None:
    if state.selected is None:
        print("[error] select a session first: use <session-number|session-id>")
        return
    if state.selected.db_file is None:
        print(f"[error] selected session has no database: {state.selected.session_id}")
        return
    format_value = "tree"
    depth = None
    root_id = None
    index = 0
    while index < len(args):
        value = args[index]
        if value == "flat":
            format_value = "flat"
        elif value == "tree":
            format_value = "tree"
        elif value == "--depth" and index + 1 < len(args):
            depth = int(args[index + 1])
            index += 1
        elif value == "--root-id" and index + 1 < len(args):
            root_id = int(args[index + 1])
            index += 1
        else:
            print("[error] use: tasks [tree|flat] [--depth N] [--root-id ID]")
            return
        index += 1

    roots = build_task_tree(Database(str(state.selected.db_file)).list_managed_tasks())
    if root_id is not None:
        roots = [root for root in roots if root.id == root_id]
    if not roots:
        print(f"[error] no task tree found for {state.selected.session_id}")
        return
    for root_index, root in enumerate(roots):
        if root_index:
            print()
        print(TaskTreeRenderer(format=format_value, depth=depth).render(root))


def handle_index_command(args: list[str], state: InspectState) -> None:
    if not args:
        print("[error] use: index list | index use <index-number> | index tree [path] [--depth N] [--entry-limit N] [--db PATH] [--repo PATH]")
        return
    command = args[0]
    rest = args[1:]
    if command == "list":
        print_index_tasks(state)
    elif command == "use":
        select_index_task(rest, state)
    elif command == "tree":
        print_index_tree(rest, state)
    else:
        print(f"[unknown index command] {command}")
        print("[error] use: index list | index use <index-number> | index tree [path] [--depth N] [--entry-limit N] [--db PATH] [--repo PATH]")


def print_index_tasks(state: InspectState) -> None:
    tasks = load_index_tasks(state)
    print(f"[index] count={len(tasks)}")
    for index, task in enumerate(tasks, start=1):
        selected = "*" if state.selected_index_task and task.id == state.selected_index_task.id else " "
        print(f"{selected}{index}. {task.title} db={task.index_db_path} repo={task.repo_path}")


def select_index_task(args: list[str], state: InspectState) -> None:
    tasks = load_index_tasks(state)
    if not args:
        print("[error] index use <index-number>")
        return
    selector = args[0]
    if not selector.isdigit():
        print("[error] index use <index-number>")
        return
    index = int(selector)
    if index < 1 or index > len(tasks):
        print(f"[error] index task out of range: {index}")
        return
    state.selected_index_task = tasks[index - 1]
    print(f"[selected index] {index}. {state.selected_index_task.title}")


def print_index_tree(args: list[str], state: InspectState) -> None:
    try:
        path, depth, entry_limit, db_path, repo_path = parse_index_tree_args(args)
    except ValueError as exc:
        print(f"[error] {exc}")
        print("[error] use: index tree [path] [--depth N] [--entry-limit N] [--db PATH] [--repo PATH]")
        return
    selected_task = state.selected_index_task
    if db_path is None or repo_path is None:
        if selected_task is None:
            tasks = load_index_tasks(state)
            if len(tasks) == 1:
                selected_task = tasks[0]
                state.selected_index_task = selected_task
            elif len(tasks) > 1:
                print("[error] multiple repo indexes found; run index use <index-number> or pass --db and --repo")
                return
            else:
                print("[error] no repo index found; pass --db and --repo")
                return
        db_path = db_path or selected_task.index_db_path
        repo_path = repo_path or selected_task.repo_path
    output = AgentIndex(db_path, base_dir=repo_path).tree(path=path, depth=depth, entry_limit=entry_limit)
    print(output)


def parse_index_tree_args(args: list[str]) -> tuple[str, int | None, int | None, str | None, str | None]:
    path = ""
    depth = None
    entry_limit = None
    db_path = None
    repo_path = None
    index = 0
    while index < len(args):
        value = args[index]
        if value == "--depth" and index + 1 < len(args):
            depth = int(args[index + 1])
            index += 2
        elif value == "--entry-limit" and index + 1 < len(args):
            entry_limit = int(args[index + 1])
            index += 2
        elif value == "--db" and index + 1 < len(args):
            db_path = args[index + 1]
            index += 2
        elif value == "--repo" and index + 1 < len(args):
            repo_path = args[index + 1]
            index += 2
        elif value.startswith("--"):
            raise ValueError(f"Unknown index tree option: {value}")
        elif not path:
            path = value
            index += 1
        else:
            raise ValueError(f"Unexpected index tree argument: {value}")
    return path, depth, entry_limit, db_path, repo_path


def load_index_tasks(state: InspectState) -> list[RepoMemoryTask]:
    if state.selected is None:
        print("[error] select a session first: use <session-number|session-id>")
        return []
    if state.selected.db_file is None:
        print(f"[error] selected session has no database: {state.selected.session_id}")
        return []
    if state.index_tasks:
        return state.index_tasks
    roots = build_task_tree(Database(str(state.selected.db_file)).list_managed_tasks())
    state.index_tasks = [
        task
        for task in flatten_tasks(roots)
        if isinstance(task, RepoMemoryTask)
    ]
    return state.index_tasks


def flatten_tasks(tasks) -> list:
    flattened = []
    stack = list(reversed(tasks))
    while stack:
        task = stack.pop()
        flattened.append(task)
        stack.extend(reversed(task.children))
    return flattened


def load_moves(log_file: Path) -> list[dict[str, Any]]:
    moves: list[dict[str, Any]] = []
    with log_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                moves.append(json.loads(line))
    return moves


def build_playback_context(
    moves: list[dict[str, Any]],
    *,
    db_path: Path | None,
    sessions_dir: Path,
    log_file: Path,
) -> PlaybackContext:
    resolved_db_path = db_path or infer_db_path(moves, sessions_dir=sessions_dir, log_file=log_file)
    if resolved_db_path is None or not resolved_db_path.exists():
        return PlaybackContext(tool_results={})

    session_id = session_id_from_moves(moves, log_file)
    db = Database(str(resolved_db_path))
    return PlaybackContext(
        tool_results={
            record.tool_call_id: extract_tool_result_text(record.tool_result_json)
            for record in db.list_runner_tool_calls(session_id)
        }
    )


def infer_db_path(moves: list[dict[str, Any]], *, sessions_dir: Path, log_file: Path) -> Path | None:
    session_id = session_id_from_moves(moves, log_file)
    if not session_id:
        return None
    return sessions_dir / f"{session_id}.db"


def session_id_from_moves(moves: list[dict[str, Any]], log_file: Path) -> str:
    for move in moves:
        session_id = move.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
    return log_file.stem


def handle_playback_command(
    command: str,
    moves: list[dict[str, Any]],
    state: PlaybackState,
    context: PlaybackContext | None = None,
) -> None:
    context = context or PlaybackContext(tool_results={})
    parts = command.split()
    name = parts[0]
    args = parts[1:]
    if name in {"list", "ls"}:
        print_move_list(moves)
    elif name in {"show", "move"}:
        if not args:
            print("[error] move id required")
            return
        try:
            move_id = int(args[0])
        except ValueError:
            print("[error] move id must be an integer")
            return
        show_state = state
        if len(args) > 1:
            show_state = PlaybackState(
                tool_result_mode=state.tool_result_mode,
                tool_result_limit=state.tool_result_limit,
            )
            if not set_tool_result(args[1:], show_state):
                return
        print_move(moves, move_id, show_state, context)
    elif name == "tool":
        configure_tool_result(args, state)
    else:
        print(f"[unknown command] {name}")


def configure_tool_result(args: list[str], state: PlaybackState) -> None:
    if not args:
        print(f"[tool result] mode={state.tool_result_mode} limit={state.tool_result_limit}")
        return
    if not set_tool_result(args, state):
        return
    print(f"[tool result] mode={state.tool_result_mode} limit={state.tool_result_limit}")


def set_tool_result(args: list[str], state: PlaybackState) -> bool:
    value = args[0]
    if value == "none":
        state.tool_result_mode = "none"
        state.tool_result_limit = None
    elif value == "full":
        state.tool_result_mode = "full"
        state.tool_result_limit = None
    else:
        try:
            limit = int(value)
        except ValueError:
            print("[error] use: tool none | tool full | tool <char-limit>")
            return False
        state.tool_result_mode = "full"
        state.tool_result_limit = limit
    return True


def print_move_list(moves: list[dict[str, Any]]) -> None:
    print(f"[moves] count={len(moves)}")
    for move_id, move in enumerate(moves, start=1):
        print(f"{move_id}. {brief_move(move)}")


def brief_move(move: dict[str, Any]) -> str:
    event = move.get("event", "unknown")
    if event == "handle_running":
        count = len(move.get("messages", []))
        assistant = move.get("assistant_message", {})
        tool_names = tool_call_names(assistant)
        text = first_text(assistant)
        suffix = f" tools: {', '.join(tool_names)}" if tool_names else ""
        return f"running - {count} context messages, assistant: {preview(text, 80)}{suffix}"
    if event == "handle_running_context":
        count = len(move.get("messages", []))
        instruction = first_text(move.get("user_instruction_message", {}))
        return f"running context - {count} messages + instruction: {preview(instruction, 80)}"
    if event == "handle_running_response":
        assistant = move.get("assistant_message", {})
        tool_names = tool_call_names(assistant)
        text = first_text(assistant)
        suffix = f" tools: {', '.join(tool_names)}" if tool_names else ""
        return f"running response - assistant: {preview(text, 80)}{suffix}"
    if event == "handle_compact_result":
        scope = move.get("message_scope", {})
        replacements = len(move.get("replacement_messages", []))
        return (
            "compact result - "
            f"scope {scope.get('start_message_id')}..{scope.get('end_message_id')} "
            f"replacement {replacements} messages"
        )
    return event


def print_move(
    moves: list[dict[str, Any]],
    move_id: int,
    state: PlaybackState,
    context: PlaybackContext | None = None,
) -> None:
    context = context or PlaybackContext(tool_results={})
    if move_id < 1 or move_id > len(moves):
        print(f"[error] move id out of range: {move_id}")
        return
    move = moves[move_id - 1]
    print(f"[move {move_id}] {brief_move(move)}")
    event = move.get("event")
    if event == "handle_running":
        print_messages("messages", move.get("messages", []), state, context)
        print("[user instruction]")
        print_message(move.get("user_instruction_message", {}), state, context)
        print("[assistant response]")
        print_message(move.get("assistant_message", {}), state, context)
        print_tool_results(move.get("tool_results", []), state, context)
    elif event == "handle_running_context":
        print_messages("messages", move.get("messages", []), state, context)
        print("[user instruction]")
        print_message(move.get("user_instruction_message", {}), state, context)
    elif event == "handle_running_response":
        print("[assistant response]")
        print_message(move.get("assistant_message", {}), state, context)
        print_tool_results(move.get("tool_results", []), state, context)
    elif event == "handle_compact_result":
        print(f"[message scope] {move.get('message_scope')}")
        print_messages("compact messages", move.get("compact_messages", []), state, context)
        print_messages("compacted messages", move.get("compacted_messages", []), state, context)
        print_messages("replacement messages", move.get("replacement_messages", []), state, context)
    else:
        print(json.dumps(move, indent=2, ensure_ascii=False))


def print_messages(
    title: str,
    messages: list[dict[str, Any]],
    state: PlaybackState,
    context: PlaybackContext,
) -> None:
    print(f"[{title}] count={len(messages)}")
    for index, item in enumerate(messages, start=1):
        message = item.get("message", item)
        prefix = f"{index}."
        if "id" in item:
            prefix += f" id={item['id']}"
        role = message.get("role", "unknown")
        print(f"{prefix} role={role}")
        print_message(message, state, context, indent="   ")


def print_message(
    message: dict[str, Any],
    state: PlaybackState,
    context: PlaybackContext,
    *,
    indent: str = "  ",
) -> None:
    if "tool_result" in message:
        print_tool_results([message["tool_result"]], state, context, indent=indent)
        return
    for item in message.get("content", []):
        item_type = item.get("type")
        if item_type == "text":
            print(f"{indent}[text] {item.get('text', '')}")
        elif item_type == "thinking":
            print(f"{indent}[thinking] {item.get('thinking', '')}")
        elif item_type == "tool_call":
            print(
                f"{indent}[tool call] {item.get('name')} "
                f"id={item.get('id')} args={json.dumps(item.get('arguments'), ensure_ascii=False)}"
            )


def print_tool_results(
    tool_results: list[dict[str, Any]],
    state: PlaybackState,
    context: PlaybackContext,
    *,
    indent: str = "  ",
) -> None:
    if not tool_results:
        return
    print(f"{indent}[tool results] count={len(tool_results)}")
    for result in tool_results:
        print(
            f"{indent}- tool_call_id={result.get('tool_call_id')} "
            f"tool={result.get('tool_name')}"
        )
        text = tool_result_text(result, context)
        if text and state.tool_result_mode == "full":
            print(f"{indent}  {format_tool_result_text(str(text), state)}")


def tool_result_text(result: dict[str, Any], context: PlaybackContext) -> str | None:
    direct_text = result.get("content") or result.get("text")
    if direct_text is not None:
        return str(direct_text)
    tool_call_id = result.get("tool_call_id")
    if not isinstance(tool_call_id, str):
        return None
    return context.tool_results.get(tool_call_id)


def extract_tool_result_text(payload: str) -> str:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return payload
    text = extract_text(value)
    if text is not None:
        return text
    return json.dumps(value, ensure_ascii=False)


def extract_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [extract_text(item) for item in value]
        return "\n".join(part for part in parts if part) or None
    if not isinstance(value, dict):
        return None

    if isinstance(value.get("text"), str):
        return value["text"]

    content = value.get("content")
    if content is not None:
        return extract_text(content)

    return None


def format_tool_result_text(text: str, state: PlaybackState) -> str:
    if state.tool_result_limit is None:
        return text
    if len(text) <= state.tool_result_limit:
        return text
    return text[: state.tool_result_limit] + f"... [truncated {len(text) - state.tool_result_limit} chars]"


def tool_call_names(message: dict[str, Any]) -> list[str]:
    return [
        item.get("name", "")
        for item in message.get("content", [])
        if item.get("type") == "tool_call"
    ]


def first_text(message: dict[str, Any]) -> str:
    for item in message.get("content", []):
        if item.get("type") == "text":
            return item.get("text", "")
    return ""


def preview(text: str, limit: int) -> str:
    text = text.replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def print_repl_help() -> None:
    print("Commands:")
    print("  sessions")
    print("  use <session-number|session-id>")
    print("  list")
    print("  show <move-id> [none|<char-limit>|full]")
    print("  tool none|<char-limit>|full")
    print("  tasks [tree|flat] [--depth N] [--root-id ID]")
    print("  index list")
    print("  index use <index-number>")
    print("  index tree [path] [--depth N] [--entry-limit N] [--db PATH] [--repo PATH]")
    print("  quit")


if __name__ == "__main__":
    main()
