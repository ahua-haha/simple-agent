"""SessionRunner owns the persisted Session.run workflow."""

from __future__ import annotations

import asyncio
import logging
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from simple_agent.json_utils import json_safe
from simple_agent.token_estimation import estimate_messages_tokens
from simple_agent.tool.common_tools import create_all_coding_tools

if TYPE_CHECKING:
    from simple_agent.db.db import Database
    from simple_agent.process.agent_process import AgentProcess
    from simple_agent.task_manager import TaskManager
    from pi.agent.types import AgentMessage
    from sqlmodel import Session

_log = logging.getLogger(__name__)

RunnerAction = Literal["normal_run", "compact", "wait_user_input"]

SYSTEM_PROMPT = """You are a helpful coding agent.

Be concise, practical, and honest about uncertainty. Use available tools
when they are needed, and explain outcomes clearly.
"""

DEFAULT_CONTEXT_TOKEN_THRESHOLD = 120_000
DEFAULT_TOOL_CALL_THRESHOLD = 200

COMPACT_SYSTEM_PROMPT = """Compact finished todos into one compacted todo.
Use only the compact tools. Create exactly one compacted todo, record useful
tool-call log IDs, then finish the compacted todo."""


@dataclass(frozen=True)
class PendingToolCallRecord:
    log_id: int
    tool_call: ToolCall | None
    tool_result: ToolResultMessage


class SessionRunner:
    """Persisted runner for one Session.run invocation at a time."""

    _session_id: str
    _db: Database
    _task_manager: TaskManager
    _agent_process: AgentProcess
    _cancel_event: asyncio.Event
    _next_action: RunnerAction
    _active_user_task_id: int | None
    _last_error: str | None
    _messages: list[AgentMessage]
    _context_token_threshold: int
    _tool_call_threshold: int
    _user_paused: bool
    _uncommitted_messages: list[AgentMessage]
    _uncommitted_tool_calls: list[PendingToolCallRecord]
    _next_tool_call_log_id: int

    def __init__(
        self,
        *,
        session_id: str,
        db: Database,
        task_manager: TaskManager,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event,
        context_token_threshold: int = DEFAULT_CONTEXT_TOKEN_THRESHOLD,
        tool_call_threshold: int = DEFAULT_TOOL_CALL_THRESHOLD,
    ):
        self._session_id = session_id
        self._db = db
        self._task_manager = task_manager
        self._agent_process = agent_process
        self._cancel_event = cancel_event
        self._next_action = "wait_user_input"
        self._active_user_task_id = None
        self._last_error = None
        self._messages = []
        self._context_token_threshold = context_token_threshold
        self._tool_call_threshold = tool_call_threshold
        self._user_paused = False
        self._uncommitted_messages = []
        self._uncommitted_tool_calls = []
        self._next_tool_call_log_id = 0

    def subscribe(self, callback: Callable) -> None:
        self._agent_process.subscribe(callback)

    def unsubscribe(self, callback: Callable) -> None:
        self._agent_process.unsubscribe(callback)

    def pause(self) -> None:
        self._user_paused = True
        self._cancel_event.set()

    def load(self) -> None:
        with self._db.create_session() as session:
            metadata = self._db.get_runner_state_metadata(self._session_id, session=session)
            self._messages = self._db.list_runner_messages(self._session_id, session=session)
            self._uncommitted_messages = []
            self._uncommitted_tool_calls = []
            self._next_tool_call_log_id = self._db.next_runner_tool_call_id(self._session_id, session=session)
            if metadata is None:
                self._next_action = "wait_user_input"
                self._active_user_task_id = None
                self._last_error = None
                self._task_manager.load(None, session=session)
                return
            self._next_action = metadata.next_action
            self._active_user_task_id = metadata.active_user_task_id
            self._last_error = metadata.last_error
            self._task_manager.load(metadata.active_user_task_id, session=session)

    def sync_metadata(self, *, session: Session) -> None:
        self._db.upsert_runner_state_metadata(
            self._session_id,
            next_action=self._next_action,
            active_user_task_id=self._active_user_task_id,
            last_error=self._last_error,
            session=session,
        )

    def append_messages(self, messages: list[AgentMessage]) -> None:
        self._messages.extend(messages)
        self._uncommitted_messages.extend(messages)

    def sync_messages(self, *, session: Session) -> None:
        for message in self._uncommitted_messages:
            self._db.insert_runner_message(self._session_id, message, session=session)

    def sync_tool_calls(self, *, session: Session) -> None:
        for pending in self._uncommitted_tool_calls:
            self._db.insert_runner_tool_call(
                id=pending.log_id,
                session_id=self._session_id,
                tool_call_id=pending.tool_result.tool_call_id,
                tool_name=pending.tool_result.tool_name,
                tool_call_json=self._tool_call_json(pending.tool_call),
                tool_result_json=self._tool_result_json(pending.tool_result),
                session=session,
            )

    def record_tool_calls(
        self,
        assistant_message: AssistantMessage,
        tool_results: list[ToolResultMessage],
    ) -> None:
        tool_calls = {
            content.id: content
            for content in assistant_message.content
            if isinstance(content, ToolCall)
        }
        for tool_result in tool_results:
            log_id = self._next_tool_call_log_id
            self._next_tool_call_log_id = log_id + 1
            self._task_manager.record_tool_call(log_id)
            self._uncommitted_tool_calls.append(
                PendingToolCallRecord(
                    log_id=log_id,
                    tool_call=tool_calls.get(tool_result.tool_call_id),
                    tool_result=tool_result,
                )
            )

    def save_current_data(
        self,
        *,
        save_tasks: bool = True,
    ) -> None:
        with self._db.create_session() as session:
            self.sync_tool_calls(session=session)
            self.sync_messages(session=session)
            if save_tasks:
                self._task_manager.save(session=session)
            self.sync_metadata(session=session)
            session.commit()
        self._uncommitted_messages.clear()
        self._uncommitted_tool_calls.clear()

    def _create_tools(self):
        return [
            self._task_manager.create_create_todo_tool(),
            self._task_manager.create_finish_todo_tool(),
            self._task_manager.create_error_todo_tool(),
            *create_all_coding_tools("."),
        ]

    async def run(self, user_input: str | None):
        self._user_paused = False
        self._cancel_event.clear()
        self.load()
        next_action = self.handle_input(user_input)
        try:
            while next_action != "wait_user_input":
                if self._user_paused:
                    break
                next_action = await self.route_next_action(next_action, user_input)
        except Exception as exc:
            self.handle_error(exc)
            raise

        if self._active_user_task_id is None:
            return None
        with self._db.create_session() as session:
            return self._db.get_managed_task(self._active_user_task_id, session=session)

    async def route_next_action(self, next_action: RunnerAction, user_input: str | None) -> RunnerAction:
        if next_action == "normal_run":
            return await self.handle_running(user_input)
        if next_action == "compact":
            return await self.handle_compact(user_input)
        if next_action == "wait_user_input":
            return next_action
        raise RuntimeError(f"Unknown runner action: {next_action}")

    def handle_input(self, user_input: str | None) -> RunnerAction:
        if user_input is None:
            return self._next_action

        self.finish_previous_user_task()
        with self._db.create_session() as session:
            self._task_manager.load(None, session=session)
        user_task = self._task_manager.create_user_task(user_input)
        self._active_user_task_id = user_task.id
        self._next_action = "normal_run"
        self._last_error = None
        self._cancel_event.clear()
        self.append_messages(
            [
                UserMessage(
                    content=[TextContent(text=user_input)],
                    timestamp=int(time.time() * 1000),
                )
            ]
        )
        self.save_current_data()
        return self._next_action

    def finish_previous_user_task(self) -> None:
        previous_user_task = self._task_manager.active_user_task
        if previous_user_task is None:
            return

        if previous_user_task.status != "done":
            if self._task_manager.active_todo_id is not None:
                self._task_manager.error_task("Interrupted by new user input")
            self._task_manager.finish_user_task()
            self._next_action = "wait_user_input"
            self.save_current_data()
        self._active_user_task_id = None

    async def handle_running(self, user_input: str | None) -> RunnerAction:
        if self._next_action != "normal_run":
            return self._next_action

        tools = self._create_tools()
        llm_messages = [
            *self._messages,
            UserMessage(
                content=[TextContent(text=self._task_manager.user_instruction_text())],
                timestamp=int(time.time() * 1000),
            ),
        ]
        assistant_message = await self._agent_process.call_llm_step(
            system_prompt=SYSTEM_PROMPT,
            messages=llm_messages,
            tools=tools,
            cancel_event=self._cancel_event,
        )
        tool_results: list[ToolResultMessage] = []
        has_tool_calls = self._assistant_has_tool_calls(assistant_message)
        if has_tool_calls:
            tool_results = await self._agent_process.run_tool_calls_step(
                tools=tools,
                assistant_message=assistant_message,
                cancel_event=self._cancel_event,
            )
            self.record_tool_calls(assistant_message, tool_results)

        self.append_messages([assistant_message, *tool_results])

        if not has_tool_calls:
            self._task_manager.finish_user_task()
            self._next_action = "wait_user_input"
        else:
            self.pause_for_compaction_if_needed()

        self.save_current_data()
        return self._next_action

    def pause_for_compaction_if_needed(self) -> None:
        context_tokens = estimate_messages_tokens(list(self._messages))
        with self._db.create_session() as session:
            tool_calls = len(self._db.list_runner_tool_calls(self._session_id, session=session))
        tool_calls += len(self._uncommitted_tool_calls)
        if (
            context_tokens <= self._context_token_threshold
            and tool_calls <= self._tool_call_threshold
        ):
            return

        self._next_action = "compact"
        self._cancel_event.set()

    async def handle_compact(self, user_input: str | None) -> RunnerAction:
        scope = self._task_manager.compact_scope()
        if scope is None:
            self._next_action = "normal_run"
            self._cancel_event.clear()
            self.save_current_data(save_tasks=False)
            return self._next_action

        start_tool_call_id = scope.compact_todos[0].create_tool_call_id
        if start_tool_call_id is None:
            raise RuntimeError("Compact start todo is missing create_tool_call_id")
        start_index = self.find_assistant_message_index_for_tool_call(start_tool_call_id)

        end_tool_call_id = scope.compact_todos[-1].end_tool_call_id
        if end_tool_call_id is None:
            raise RuntimeError("Compact end todo is missing end_tool_call_id")
        end_index = self.find_tool_result_message_index_for_tool_call(end_tool_call_id)
        if end_index < start_index:
            raise RuntimeError("Compact end message is before start message")

        messages_before_start = self._messages[:start_index]
        preserved_messages = self._messages[end_index + 1:]

        self._task_manager.begin_compact_buffer()
        compact_messages = list(self._messages[start_index:end_index + 1])
        compact_tools = self._task_manager.create_compact_tools()
        while True:
            assistant_message = await self._agent_process.call_llm_step(
                system_prompt=COMPACT_SYSTEM_PROMPT,
                messages=compact_messages,
                tools=compact_tools,
                cancel_event=self._cancel_event,
            )
            compact_messages.append(assistant_message)
            if not self._assistant_has_tool_calls(assistant_message):
                break
            tool_results = await self._agent_process.run_tool_calls_step(
                tools=compact_tools,
                assistant_message=assistant_message,
                cancel_event=self._cancel_event,
            )
            compact_messages.extend(tool_results)

        with self._db.create_session() as session:
            compacted_todo = self._task_manager.replace_compact_scope(session=session)
            compact_messages = [self.format_compacted_todo_message(compacted_todo)]
            replacement_messages = [*messages_before_start, *compact_messages, *preserved_messages]
            self._db.replace_runner_messages(self._session_id, replacement_messages, session=session)
            self._next_action = "normal_run"
            self.sync_metadata(session=session)
            session.commit()

        self._messages = replacement_messages
        self._uncommitted_messages.clear()
        self._cancel_event.clear()
        return self._next_action

    def find_assistant_message_index_for_tool_call(self, tool_call_id: str) -> int:
        for index, message in enumerate(self._messages):
            if not isinstance(message, AssistantMessage):
                continue
            for content in message.content:
                if isinstance(content, ToolCall) and content.id == tool_call_id:
                    return index
        raise RuntimeError(f"Could not find assistant message for tool call {tool_call_id}")

    def find_tool_result_message_index_for_tool_call(self, tool_call_id: str) -> int:
        for index, message in enumerate(self._messages):
            if isinstance(message, ToolResultMessage) and message.tool_call_id == tool_call_id:
                return index
        raise RuntimeError(f"Could not find tool result message for tool call {tool_call_id}")

    def _assistant_has_tool_calls(self, message: AssistantMessage) -> bool:
        return any(isinstance(content, ToolCall) for content in message.content)

    def _tool_call_json(self, tool_call: ToolCall | None) -> str:
        return json.dumps(json_safe(tool_call), sort_keys=True, separators=(",", ":"))

    def _tool_result_json(self, tool_result: ToolResultMessage) -> str:
        return json.dumps(json_safe(tool_result), sort_keys=True, separators=(",", ":"))

    def format_compacted_todo_message(self, compacted_todo) -> AgentMessage:
        tool_refs = [
            child.tool_call_log_id
            for child in compacted_todo.children
            if child.kind == "tool_call" and child.tool_call_log_id is not None
        ]
        text = (
            f"Compacted todo: {compacted_todo.result or compacted_todo.title}\n"
            f"Useful tool calls: {tool_refs}"
        )
        return AssistantMessage(role="assistant", content=[TextContent(text=text)])

    def handle_error(self, exc: Exception) -> None:
        _log.exception("session runner failed: session=%s", self._session_id)
        self._next_action = "wait_user_input"
        self._last_error = str(exc)
        with self._db.create_session() as session:
            self.sync_metadata(session=session)
            session.commit()
