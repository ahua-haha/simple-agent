"""SessionManager — manages multiple Session instances with background task tracking."""

from __future__ import annotations

import asyncio
import logging
import os
import time

from simple_agent.log import logged
from simple_agent.session.session import Session

_log = logging.getLogger(__name__)


DEFAULT_COOLDOWN_SECONDS = 300


class SessionBusyError(Exception):
    """Raised when run() is called on a session that is already executing."""
    pass


class SessionManager:
    """Manages multiple Session instances with lifecycle operations.

    Tracks background asyncio tasks per session and implements an idle
    cooldown mechanism that parks sessions to disk after inactivity.

    Usage::

        sm = SessionManager(sessions_dir="./sessions")
        s = sm.create()
        queue = sm.run(s.id, "build a test suite")
        # read events from queue for SSE streaming
    """

    def __init__(self, sessions_dir: str = "./sessions",
                 cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS):
        self._sessions_dir = sessions_dir
        self._cooldown_seconds = cooldown_seconds
        self._sessions: dict[str, Session] = {}
        self._run_tasks: dict[str, asyncio.Task] = {}
        self._cooldown_timers: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # create / get / list / remove
    # ------------------------------------------------------------------

    def create(self) -> Session:
        """Create a new session, persist it, and register in memory."""
        session = Session(base_dir=self._sessions_dir)
        session._checkpoint()
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Return a session by ID, reloading from disk if parked.

        Returns None if the session doesn't exist (no in-memory entry
        and no DB file on disk).
        """
        if session_id in self._sessions:
            return self._sessions[session_id]

        db_path = os.path.join(self._sessions_dir, f"{session_id}.db")
        if os.path.isfile(db_path):
            session = Session(session_id=session_id, base_dir=self._sessions_dir)
            self._sessions[session_id] = session
            return session

        return None

    def list(self) -> list[dict]:
        """Return all known sessions with derived status.

        Status is derived, not stored:
        - "running": active asyncio task in _run_tasks
        - "idle": in _sessions but not running
        - "parked": on disk but not in _sessions
        """
        seen: set[str] = set()
        result: list[dict] = []

        for sid, session in self._sessions.items():
            seen.add(sid)
            result.append({
                "id": sid,
                "status": "running" if sid in self._run_tasks else "idle",
                "created_at": session._created_at,
                "updated_at": session._updated_at,
            })

        # Add parked sessions (on disk, not in memory)
        if os.path.isdir(self._sessions_dir):
            for fname in os.listdir(self._sessions_dir):
                if not fname.endswith(".db"):
                    continue
                sid = fname[:-3]
                if sid not in seen:
                    result.append({
                        "id": sid,
                        "status": "parked",
                        "created_at": None,
                        "updated_at": None,
                    })

        return result

    def remove(self, session_id: str) -> None:
        """Pause if running, remove from memory, and delete the DB file."""
        # Pause if running
        if session_id in self._run_tasks:
            session = self._sessions.get(session_id)
            if session is not None:
                session.pause()
            self._run_tasks[session_id].cancel()
            del self._run_tasks[session_id]

        # Cancel any pending cooldown timer
        if session_id in self._cooldown_timers:
            self._cooldown_timers[session_id].cancel()
            del self._cooldown_timers[session_id]

        # Park the session (release cursor) if in memory
        if session_id in self._sessions:
            session = self._sessions.pop(session_id)
            if not session._running:
                session.park()

        # Delete DB file
        db_path = os.path.join(self._sessions_dir, f"{session_id}.db")
        if os.path.isfile(db_path):
            os.remove(db_path)

    # ------------------------------------------------------------------
    # run / pause / cooldown
    # ------------------------------------------------------------------

    @logged(_log)
    def run(self, session_id: str, user_input: str) -> asyncio.Queue:
        """Start a background run of *session_id* with *user_input*.

        Returns the session's event queue for SSE streaming.
        Raises SessionBusyError if the session is already running.
        """
        if session_id in self._run_tasks:
            raise SessionBusyError(f"Session {session_id} is already running")

        session = self.get(session_id)
        if session is None:
            raise LookupError(f"Session {session_id} not found")

        # Cancel cooldown if one is active
        if session_id in self._cooldown_timers:
            self._cooldown_timers[session_id].cancel()
            del self._cooldown_timers[session_id]

        # Create queue before spawning task so the caller can read from it immediately
        queue: asyncio.Queue = asyncio.Queue()
        session.event_queue = queue

        task = asyncio.create_task(
            self._run_task_wrapper(session_id, session, user_input)
        )
        self._run_tasks[session_id] = task

        return queue

    def pause(self, session_id: str) -> None:
        """Signal the session's run loop to stop at the next safe point."""
        session = self._sessions.get(session_id)
        if session is not None:
            session.pause()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    async def _run_task_wrapper(self, session_id: str, session: Session,
                                  user_input: str) -> None:
        """Wrap session.run() with cleanup on completion."""
        try:
            await session.run(user_input)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            # Clean up run task tracking
            self._run_tasks.pop(session_id, None)

            # Start cooldown timer if session is still in memory
            if session_id in self._sessions and self._cooldown_seconds > 0:
                self._start_cooldown(session_id, session)

    def _start_cooldown(self, session_id: str, session: Session) -> None:
        """Start an idle cooldown timer that parks the session on expiry."""

        async def _cooldown():
            await asyncio.sleep(self._cooldown_seconds)
            if session_id in self._sessions:
                s = self._sessions.pop(session_id)
                s.park()
            self._cooldown_timers.pop(session_id, None)

        self._cooldown_timers[session_id] = asyncio.create_task(_cooldown())
