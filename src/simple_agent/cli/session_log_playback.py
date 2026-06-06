"""Interactive playback for session JSONL run logs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from simple_agent.db.db import Database

ToolResultMode = Literal["none", "full"]


@dataclass
class PlaybackState:
    tool_result_mode: ToolResultMode = "none"
    tool_result_limit: int | None = None


@dataclass
class PlaybackContext:
    tool_results: dict[str, str]


def main() -> None:
    args = _parse_args()
    log_file = Path(args.log_file)
    moves = load_moves(log_file)
    state = PlaybackState(
        tool_result_mode=args.tool_result,
        tool_result_limit=args.tool_result_limit,
    )
    context = build_context(
        moves,
        db_path=Path(args.db) if args.db else None,
        sessions_dir=Path(args.sessions_dir),
        log_file=log_file,
    )
    print_move_list(moves)
    if args.move is not None:
        print()
        print_move(moves, args.move, state, context)
        return
    run_interactive_loop(moves, state, context)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Playback a simple-agent session JSONL run log")
    parser.add_argument("log_file", help="Path to logs/session_runs/<session_id>.jsonl")
    parser.add_argument("--move", type=int, help="Display one move id and exit")
    parser.add_argument(
        "--tool-result",
        choices=["none", "full"],
        default="none",
        help="How much tool result content to show when the log contains it",
    )
    parser.add_argument(
        "--tool-result-limit",
        type=int,
        help="Show at most this many characters of tool result content",
    )
    parser.add_argument("--db", help="Path to the session database used to resolve tool results")
    parser.add_argument("--sessions-dir", default="./sessions", help="Directory containing session DB files")
    return parser.parse_args()


def load_moves(log_file: Path) -> list[dict[str, Any]]:
    moves: list[dict[str, Any]] = []
    with log_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                moves.append(json.loads(line))
    return moves


def build_context(
    moves: list[dict[str, Any]],
    *,
    db_path: Path | None,
    sessions_dir: Path,
    log_file: Path,
) -> PlaybackContext:
    resolved_db_path = db_path or _infer_db_path(moves, sessions_dir=sessions_dir, log_file=log_file)
    if resolved_db_path is None or not resolved_db_path.exists():
        return PlaybackContext(tool_results={})

    session_id = _session_id(moves, log_file)
    db = Database(str(resolved_db_path))
    return PlaybackContext(
        tool_results={
            record.tool_call_id: extract_tool_result_text(record.tool_result_json)
            for record in db.list_runner_tool_calls(session_id)
        }
    )


def _infer_db_path(moves: list[dict[str, Any]], *, sessions_dir: Path, log_file: Path) -> Path | None:
    session_id = _session_id(moves, log_file)
    if not session_id:
        return None
    return sessions_dir / f"{session_id}.db"


def _session_id(moves: list[dict[str, Any]], log_file: Path) -> str:
    for move in moves:
        session_id = move.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
    return log_file.stem


def run_interactive_loop(moves: list[dict[str, Any]], state: PlaybackState, context: PlaybackContext) -> None:
    print()
    print_help()
    while True:
        try:
            command = input("playback> ").strip()
        except EOFError:
            print()
            return
        if not command:
            continue
        if command in {"quit", "exit", "q"}:
            return
        handle_command(command, moves, state, context)


def handle_command(
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
    elif name in {"help", "?"}:
        print_help()
    else:
        print(f"[unknown command] {name}")
        print_help()


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


def print_help() -> None:
    print("Commands:")
    print("  list")
    print("  show <move-id> [none|<char-limit>|full]")
    print("  tool none")
    print("  tool <char-limit>")
    print("  tool full")
    print("  quit")


def print_move_list(moves: list[dict[str, Any]]) -> None:
    print(f"[moves] count={len(moves)}")
    for move_id, move in enumerate(moves, start=1):
        print(f"{move_id}. {brief_move(move)}")


def brief_move(move: dict[str, Any]) -> str:
    event = move.get("event", "unknown")
    if event == "handle_running":
        count = len(move.get("messages", []))
        assistant = move.get("assistant_message", {})
        tool_names = _tool_call_names(assistant)
        text = _first_text(assistant)
        suffix = f" tools: {', '.join(tool_names)}" if tool_names else ""
        return f"running - {count} context messages, assistant: {_preview(text, 80)}{suffix}"
    if event == "handle_running_context":
        count = len(move.get("messages", []))
        instruction = _first_text(move.get("user_instruction_message", {}))
        return f"running context - {count} messages + instruction: {_preview(instruction, 80)}"
    if event == "handle_running_response":
        assistant = move.get("assistant_message", {})
        tool_names = _tool_call_names(assistant)
        text = _first_text(assistant)
        suffix = f" tools: {', '.join(tool_names)}" if tool_names else ""
        return f"running response - assistant: {_preview(text, 80)}{suffix}"
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
        _print_messages("messages", move.get("messages", []), state, context)
        print("[user instruction]")
        _print_message(move.get("user_instruction_message", {}), state, context)
        print("[assistant response]")
        _print_message(move.get("assistant_message", {}), state, context)
        _print_tool_results(move.get("tool_results", []), state, context)
    elif event == "handle_running_context":
        _print_messages("messages", move.get("messages", []), state, context)
        print("[user instruction]")
        _print_message(move.get("user_instruction_message", {}), state, context)
    elif event == "handle_running_response":
        print("[assistant response]")
        _print_message(move.get("assistant_message", {}), state, context)
        _print_tool_results(move.get("tool_results", []), state, context)
    elif event == "handle_compact_result":
        print(f"[message scope] {move.get('message_scope')}")
        _print_messages("compact messages", move.get("compact_messages", []), state, context)
        _print_messages("compacted messages", move.get("compacted_messages", []), state, context)
        _print_messages("replacement messages", move.get("replacement_messages", []), state, context)
    else:
        print(json.dumps(move, indent=2, ensure_ascii=False))


def _print_messages(
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
        _print_message(message, state, context, indent="   ")


def _print_message(
    message: dict[str, Any],
    state: PlaybackState,
    context: PlaybackContext,
    *,
    indent: str = "  ",
) -> None:
    if "tool_result" in message:
        _print_tool_results([message["tool_result"]], state, context, indent=indent)
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


def _print_tool_results(
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
    text = _extract_text(value)
    if text is not None:
        return text
    return json.dumps(value, ensure_ascii=False)


def _extract_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_extract_text(item) for item in value]
        return "\n".join(part for part in parts if part) or None
    if not isinstance(value, dict):
        return None

    if isinstance(value.get("text"), str):
        return value["text"]

    content = value.get("content")
    if content is not None:
        return _extract_text(content)

    return None


def format_tool_result_text(text: str, state: PlaybackState) -> str:
    if state.tool_result_limit is None:
        return text
    if len(text) <= state.tool_result_limit:
        return text
    return text[: state.tool_result_limit] + f"... [truncated {len(text) - state.tool_result_limit} chars]"


def _tool_call_names(message: dict[str, Any]) -> list[str]:
    return [
        item.get("name", "")
        for item in message.get("content", [])
        if item.get("type") == "tool_call"
    ]


def _first_text(message: dict[str, Any]) -> str:
    for item in message.get("content", []):
        if item.get("type") == "text":
            return item.get("text", "")
    return ""


def _preview(text: str, limit: int) -> str:
    text = text.replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


if __name__ == "__main__":
    main()
