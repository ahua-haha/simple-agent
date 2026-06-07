"""SessionRunner owns the persisted Session.run workflow."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Callable, Literal, TypeAlias

from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from simple_agent.message_store import MessageEntry
from simple_agent.run_log import runtime_logger
from simple_agent.token_estimation import estimate_messages_tokens
from simple_agent.tool.common_tools import create_all_coding_tools

if TYPE_CHECKING:
    from simple_agent.db.db import Database
    from simple_agent.process.agent_process import AgentProcess
    from simple_agent.task_manager import TaskManager
    from sqlmodel import Session

_log = logging.getLogger(__name__)

RunnerAction = Literal["normal_run", "compact", "handle_error", "wait_user_input"]

SYSTEM_PROMPT = """You are a helpful coding agent.

Be concise, practical, and honest about uncertainty. Use available tools
when they are needed, and explain outcomes clearly.
"""

DEFAULT_CONTEXT_TOKEN_THRESHOLD = 120_000
DEFAULT_TOOL_CALL_THRESHOLD = 200

COMPACT_SYSTEM_PROMPT = """Compact the finished user task into one compacted user task.
Use only the compact tools. Set the compacted user task result, record useful
tool-call log IDs, then finish the compacted user task."""


ToolCallLogRecord: TypeAlias = tuple[int, ToolCall | None, ToolResultMessage]


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
    _messages: list[MessageEntry]
    _context_token_threshold: int
    _tool_call_threshold: int
    _user_paused: bool
    _next_message_id: int
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
        self._next_message_id = 1
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
            self._messages = [
                MessageEntry(id=message_id, message=message)
                for message_id, message in self._db.list_runner_message_entries(self._session_id, session=session)
            ]
            self._next_message_id = self._db.next_runner_message_id(session=session)
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

    def append_messages(self, messages: list[MessageEntry]) -> None:
        self._messages.extend(messages)

    def sync_messages(self, messages: list[MessageEntry], *, session: Session) -> None:
        for pending in messages:
            self._db.insert_runner_message(
                self._session_id,
                pending.message,
                id=pending.id,
                session=session,
            )

    def sync_replaced_messages(
        self,
        *,
        messages: list[MessageEntry],
        session: Session,
    ) -> None:
        self._db.replace_runner_messages(
            self._session_id,
            [entry.message for entry in messages],
            ids=[entry.id for entry in messages],
            session=session,
        )

    def sync_tool_calls(self, tool_calls: list[ToolCallLogRecord], *, session: Session) -> None:
        for log_id, tool_call, tool_result in tool_calls:
            self._db.insert_runner_tool_call(
                id=log_id,
                session_id=self._session_id,
                tool_call_id=tool_result.tool_call_id,
                tool_name=tool_result.tool_name,
                tool_call_json=tool_call.model_dump_json() if tool_call is not None else "null",
                tool_result_json=tool_result.model_dump_json(),
                session=session,
            )

    def tool_call_log_records(
        self,
        assistant_message: AssistantMessage,
        tool_results: list[ToolResultMessage],
    ) -> list[ToolCallLogRecord]:
        tool_calls = {
            content.id: content
            for content in assistant_message.content
            if isinstance(content, ToolCall)
        }
        records: list[ToolCallLogRecord] = []
        for tool_result in tool_results:
            log_id = self._next_tool_call_log_id
            self._next_tool_call_log_id = log_id + 1
            records.append((log_id, tool_calls.get(tool_result.tool_call_id), tool_result))
        return records

    def sync_current_data(
        self,
        *,
        messages: list[MessageEntry] | None = None,
        tool_calls: list[ToolCallLogRecord] | None = None,
    ) -> None:
        messages = messages or []
        tool_calls = tool_calls or []
        with self._db.create_session() as session:
            self.sync_tool_calls(tool_calls, session=session)
            self.sync_messages(messages, session=session)
            self._task_manager.save(session=session)
            self.sync_metadata(session=session)
            session.commit()

    def _create_tools(self):
        task_tools = self._task_manager.create_tools()
        return [
            *task_tools,
            *create_all_coding_tools("."),
        ]

    async def run(self, user_input: str | None):
        self._user_paused = False
        self._cancel_event.clear()
        self.load()
        try:
            next_action = self.handle_input(user_input)
        except Exception as exc:
            next_action = self.handle_error(exc)

        while next_action != "wait_user_input":
            if self._user_paused:
                break
            try:
                next_action = await self.route_next_action(next_action, user_input)
            except Exception as exc:
                next_action = self.handle_error(exc)

        if self._active_user_task_id is None:
            return None
        with self._db.create_session() as session:
            return self._db.get_managed_task(self._active_user_task_id, session=session)

    async def route_next_action(self, next_action: RunnerAction, user_input: str | None) -> RunnerAction:
        if next_action == "normal_run":
            return await self.handle_running(user_input)
        if next_action == "compact":
            return await self.handle_compact(user_input, run_done=self._user_task_is_done())
        if next_action == "handle_error":
            return self.handle_error()
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
        user_message = UserMessage(
            content=[TextContent(text=user_input)],
            timestamp=int(time.time() * 1000),
        )
        pending_messages = [
            MessageEntry(id=self._allocate_message_id(), message=user_message)
        ]
        self.append_messages(pending_messages)
        user_task.start_message_id = pending_messages[0].id
        self.sync_current_data(messages=pending_messages)
        return self._next_action

    def finish_previous_user_task(self) -> None:
        previous_user_task = self._task_manager.user_task
        if previous_user_task is None:
            return

        if previous_user_task.status != "done":
            active_task = self._task_manager.active_lifecycle_for_tools().task
            if active_task.kind == "todo":
                self._task_manager.active_lifecycle_for_tools().error_task(error="Interrupted by new user input")
                self._task_manager.refresh_active_task()
            self._task_manager.finish_user_task()
            self._next_action = "wait_user_input"
            self.sync_current_data()
        self._active_user_task_id = None

    async def handle_running(self, user_input: str | None) -> RunnerAction:
        if self._next_action != "normal_run":
            return self._next_action

        tools = self._create_tools()
        user_instruction_message = UserMessage(
            content=[TextContent(text=self._task_manager.user_instruction_text())],
            timestamp=int(time.time() * 1000),
        )
        message_context = list(self._messages)
        llm_messages = [
            *self._message_values(),
            user_instruction_message,
        ]
        assistant_message = await self._agent_process.call_llm_step(
            system_prompt=SYSTEM_PROMPT,
            messages=llm_messages,
            tools=tools,
            cancel_event=self._cancel_event,
        )
        if self._assistant_is_error(assistant_message):
            return self.exit(next_action="handle_error", error=assistant_message)

        tool_results: list[ToolResultMessage] = []
        tool_call_records: list[ToolCallLogRecord] = []
        has_tool_calls = self._assistant_has_tool_calls(assistant_message)
        assistant_message_id = self._allocate_message_id()
        if has_tool_calls:
            active_lifecycle = self._task_manager.active_lifecycle_for_tools()
            active_lifecycle.current_assistant_message_id = assistant_message_id
            try:
                tool_results = await self._agent_process.run_tool_calls_step(
                    tools=tools,
                    assistant_message=assistant_message,
                    cancel_event=self._cancel_event,
                )
            finally:
                active_lifecycle.current_assistant_message_id = None
            active_lifecycle.load_tool_call_log_id(self._next_tool_call_log_id)
            tool_call_records, tool_call_tasks = active_lifecycle.create_tool_call_record_task_entries(
                assistant_message=assistant_message,
                tool_result_messages=tool_results,
            )
            self._next_tool_call_log_id = active_lifecycle.next_tool_call_log_id
            active_lifecycle.task.children.extend(tool_call_tasks)
            if tool_call_tasks:
                active_lifecycle.task.touch()
            self._task_manager.refresh_active_task()

        assistant_entry = MessageEntry(id=assistant_message_id, message=assistant_message)
        tool_result_entries = [
            MessageEntry(id=self._allocate_message_id(), message=tool_result)
            for tool_result in tool_results
        ]
        pending_messages = [assistant_entry, *tool_result_entries]
        self.append_messages(pending_messages)

        if self._user_task_is_done():
            self._next_action = "compact"
        elif not has_tool_calls:
            self._task_manager.finish_user_task(end_message_id=assistant_message_id)
            self._next_action = "compact"
        else:
            self.pause_for_compaction_if_needed(tool_call_records=tool_call_records)

        runtime_logger.log_handle_running(
            session_id=self._session_id,
            assistant_message_id=assistant_message_id,
            assistant_message=assistant_message,
            messages=message_context,
            user_instruction_message=user_instruction_message,
            tool_result_entries=tool_result_entries,
            next_action=self._next_action,
        )
        self.sync_current_data(
            messages=pending_messages,
            tool_calls=tool_call_records,
        )
        return self._next_action

    def pause_for_compaction_if_needed(self, *, tool_call_records: list[ToolCallLogRecord] | None = None) -> None:
        tool_call_records = tool_call_records or []
        context_tokens = estimate_messages_tokens(self._message_values())
        with self._db.create_session() as session:
            tool_calls = len(self._db.list_runner_tool_calls(self._session_id, session=session))
        tool_calls += len(tool_call_records)
        if (
            context_tokens <= self._context_token_threshold
            and tool_calls <= self._tool_call_threshold
        ):
            return

        self._next_action = "compact"
        self._cancel_event.set()

    async def handle_compact(self, user_input: str | None, *, run_done: bool = False) -> RunnerAction:
        if not self._task_manager.begin_compact(run_done=run_done):
            return self.exit(
                next_action="wait_user_input" if run_done else "normal_run",
                clear_cancel=True,
            )

        compact_instruction = self._task_manager.compact_instruction_text(
            session_id=self._session_id,
        )
        compact_messages = [
            *self._message_values(),
            UserMessage(
                content=[TextContent(text=compact_instruction)],
                timestamp=int(time.time() * 1000),
            ),
        ]
        compact_tools = self._task_manager.create_compact_tools()
        while True:
            assistant_message = await self._agent_process.call_llm_step(
                system_prompt=COMPACT_SYSTEM_PROMPT,
                messages=compact_messages,
                tools=compact_tools,
                cancel_event=self._cancel_event,
            )
            compact_messages.append(assistant_message)
            if self._assistant_is_error(assistant_message):
                return self.exit(next_action="handle_error", error=assistant_message)
            if not self._assistant_has_tool_calls(assistant_message):
                break
            tool_results = await self._agent_process.run_tool_calls_step(
                tools=compact_tools,
                assistant_message=assistant_message,
                cancel_event=self._cancel_event,
            )
            compact_messages.extend(tool_results)

        start_message_id, end_message_id, compacted_messages = self._task_manager.compacted_messages()
        replacement_messages = self._replace_message_range(
            messages=list(self._messages),
            start_message_id=start_message_id,
            end_message_id=end_message_id,
            replacement_messages=[
                MessageEntry(id=self._allocate_message_id(), message=message)
                for message in compacted_messages
            ],
        )
        self._messages = replacement_messages

        with self._db.create_session() as session:
            self._next_action = "wait_user_input" if run_done else "normal_run"
            runtime_logger.log_handle_compact_result(
                session_id=self._session_id,
                compact_messages=compact_messages,
                start_message_id=start_message_id,
                end_message_id=end_message_id,
                compacted_messages=compacted_messages,
                replacement_messages=replacement_messages,
                next_action=self._next_action,
            )
            self.sync_replaced_messages(
                messages=replacement_messages,
                session=session,
            )
            self._task_manager.sync_compaction(session=session)
            self.sync_metadata(session=session)
            session.commit()

        self._cancel_event.clear()
        return self._next_action

    def _assistant_has_tool_calls(self, message: AssistantMessage) -> bool:
        return any(isinstance(content, ToolCall) for content in message.content)

    def _assistant_is_error(self, message: AssistantMessage) -> bool:
        return message.stop_reason == "error" or bool(message.error_message)

    def _allocate_message_id(self) -> int:
        message_id = self._next_message_id
        self._next_message_id += 1
        return message_id

    def _message_values(self) -> list[AgentMessage]:
        return [entry.message for entry in self._messages]

    def _replace_message_range(
        self,
        *,
        messages: list[MessageEntry],
        start_message_id: int,
        end_message_id: int,
        replacement_messages: list[MessageEntry],
    ) -> list[MessageEntry]:
        start_index, end_index = self._message_range_indices(
            messages=messages,
            start_message_id=start_message_id,
            end_message_id=end_message_id,
        )
        return messages[:start_index] + replacement_messages + messages[end_index + 1:]

    def _message_range_indices(
        self,
        *,
        messages: list[MessageEntry],
        start_message_id: int,
        end_message_id: int,
    ) -> tuple[int, int]:
        message_ids = [entry.id for entry in messages]
        try:
            start_index = message_ids.index(start_message_id)
        except ValueError as exc:
            raise RuntimeError(f"Could not find compact start message id {start_message_id}") from exc
        try:
            end_index = message_ids.index(end_message_id)
        except ValueError as exc:
            raise RuntimeError(f"Could not find compact end message id {end_message_id}") from exc
        if end_index < start_index:
            raise RuntimeError("Compact end message is before compact start message")

        end_message = messages[end_index].message
        if isinstance(end_message, AssistantMessage):
            tool_call_ids = {
                content.id
                for content in end_message.content
                if isinstance(content, ToolCall)
            }
            while (
                tool_call_ids
                and end_index + 1 < len(messages)
                and isinstance(messages[end_index + 1].message, ToolResultMessage)
                and messages[end_index + 1].message.tool_call_id in tool_call_ids
            ):
                end_index += 1
        return start_index, end_index

    def _user_task_is_done(self) -> bool:
        user_task = self._task_manager.user_task
        return user_task is not None and user_task.status == "done"

    def exit(
        self,
        *,
        next_action: RunnerAction,
        error: Exception | AssistantMessage | str | None = None,
        clear_cancel: bool = False,
    ) -> RunnerAction:
        if error is not None:
            self._last_error = self._error_message(error)
        self._next_action = next_action
        if clear_cancel:
            self._cancel_event.clear()
        with self._db.create_session() as session:
            self.sync_metadata(session=session)
            session.commit()
        return self._next_action

    def handle_error(self, error: Exception | AssistantMessage | str | None = None) -> RunnerAction:
        if error is not None:
            self._last_error = self._error_message(error)
            if isinstance(error, Exception):
                _log.error(
                    "session runner failed: session=%s",
                    self._session_id,
                    exc_info=(type(error), error, error.__traceback__),
                )
            self._next_action = "handle_error"
            return self._next_action

        self._next_action = "wait_user_input"
        self._cancel_event.clear()
        with self._db.create_session() as session:
            self.sync_metadata(session=session)
            session.commit()
        return self._next_action

    def _error_message(self, error: Exception | AssistantMessage | str) -> str:
        if isinstance(error, AssistantMessage):
            return error.error_message or "assistant response stopped with error"
        return str(error)
