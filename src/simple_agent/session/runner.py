"""SessionRunner owns the persisted Session.run workflow."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Callable

from pi.ai.types import TextContent, UserMessage

from simple_agent.message_store import MessageEntry
from simple_agent.task_manager.base_lifecycle import SessionState
from simple_agent.task_manager.repo_memory_lifecycle import RepoMemoryLifecycle
from simple_agent.task_manager.task_lifecycle import CommonTaskLifecycle
from simple_agent.task_manager.orchestrator import OrchestratorLifecycle
from simple_agent.task_manager.models import UserTask

if TYPE_CHECKING:
    from simple_agent.db.db import Database
    from simple_agent.process.agent_process import AgentProcess
    from sqlmodel import Session


class SessionRunner:
    """Persisted runner for one Session.run invocation at a time."""

    _session_id: str
    _db: Database
    _workspace_dir: str
    _agent_process: AgentProcess
    _cancel_event: asyncio.Event
    _last_error: str | None
    _user_task: UserTask | None
    _common_task: CommonTaskLifecycle
    _orchestrator: OrchestratorLifecycle
    _repo_memory: RepoMemoryLifecycle
    _session_state: SessionState
    _user_paused: bool

    def __init__(
        self,
        *,
        session_id: str,
        db: Database,
        agent_process: AgentProcess,
        cancel_event: asyncio.Event,
        workspace_dir: str,
    ):
        self._session_id = session_id
        self._db = db
        self._workspace_dir = workspace_dir
        self._agent_process = agent_process
        self._cancel_event = cancel_event
        self._last_error = None
        self._user_task = None
        self._common_task = CommonTaskLifecycle()
        self._orchestrator = OrchestratorLifecycle()
        self._repo_memory = RepoMemoryLifecycle()
        self._session_state = SessionState(
            messages=[],
            session_id=self._session_id,
            database=self._db,
            workspace_dir=self._workspace_dir,
        )
        self._user_paused = False

    def subscribe(self, callback: Callable) -> None:
        self._agent_process.subscribe(callback)

    def unsubscribe(self, callback: Callable) -> None:
        self._agent_process.unsubscribe(callback)

    def pause(self) -> None:
        self._user_paused = True
        self._cancel_event.set()

    def load(self) -> None:
        with self._db.create_session() as session:
            self._load_session_state(session=session)
            self._user_task = None
            metadata = self._db.get_runner_state_metadata(self._session_id, session=session)
            if metadata is None:
                self._last_error = None
                return
            self._last_error = metadata.last_error
            # TODO: reconstruct the task tree and active lifecycle from the
            # stored runner state.

    def sync_metadata(self, *, session: Session) -> None:
        self._db.upsert_runner_state_metadata(
            self._session_id,
            active_user_task_id=self._current_active_user_task_id(),
            last_error=self._last_error,
            session=session,
        )

    async def run(self, user_input: str | None):
        self._user_paused = False
        self._cancel_event.clear()
        self.load()
        self.run_input_transition(user_input)

        while True:
            if self._user_paused:
                break

            phase = self._session_state.next_phase
            if phase is None:
                break
            if phase == "done":
                self._session_state.set_current_task(None, None)
                break

            if phase == "orchestrator":
                lifecycle = self._orchestrator
            elif phase == "common_task":
                lifecycle = self._common_task
            elif phase == "repo_memory":
                lifecycle = self._repo_memory
            else:
                break

            lifecycle.set_data(self._session_state)
            try:
                await lifecycle.run(
                    agent_process=self._agent_process,
                    cancel_event=self._cancel_event,
                )
            finally:
                lifecycle.clear_data()

            with self._db.create_session() as session:
                self.sync_metadata(session=session)
                session.commit()

        return self._current_user_task_from_database()

    def run_input_transition(self, user_input: str | None) -> None:
        if user_input is None:
            return
        if self._session_state.current_task_id is not None or self._session_state.current_task is not None:
            # TODO: finish or interrupt existing active tasks before accepting
            # a new user task.
            return

        user_message = UserMessage(
            content=[TextContent(text=user_input)],
            timestamp=int(time.time() * 1000),
        )
        message_entry = self._session_state.append_message(user_message)

        task = UserTask(
            id=self._session_state.allocate_task_id(),
            title=user_input,
            start_message_id=message_entry.id,
        )
        self._user_task = task
        self._session_state.set_current_task(task.id, task)
        self._session_state.next_phase = "orchestrator"
        self._last_error = None

        with self._db.create_session() as session:
            self._session_state.append_messages_to_database(
                messages=[message_entry],
                session=session,
            )
            self._session_state.append_tasks_to_database(
                tasks=[task],
                session=session,
            )
            self.sync_metadata(session=session)
            session.commit()

    def _current_user_task_from_database(self) -> Any | None:
        if self._user_task is None:
            return None
        with self._db.create_session() as session:
            return self._db.get_managed_task(self._user_task.id, session=session)

    def _load_session_state(self, *, session: Session) -> None:
        self._session_state = SessionState(
            messages=[
                MessageEntry(id=message_id, message=message)
                for message_id, message in self._db.list_runner_message_entries(self._session_id, session=session)
            ],
            session_id=self._session_id,
            database=self._db,
            workspace_dir=self._workspace_dir,
            next_message_id=self._db.next_runner_message_id(session=session),
            next_tool_call_log_id=self._db.next_runner_tool_call_id(self._session_id, session=session),
            next_task_id_to_allocate=self._db.next_managed_task_id(session=session),
        )

    @property
    def user_task(self) -> UserTask | None:
        return self._user_task

    def _current_active_user_task_id(self) -> int | None:
        if self._user_task is not None and self._user_task.status == "active":
            return self._user_task.id
        return None
