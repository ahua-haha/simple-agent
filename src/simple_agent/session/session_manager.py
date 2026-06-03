"""SessionManager — manages multiple Session instances."""

from __future__ import annotations

import logging
import os

from simple_agent.log import logged
from simple_agent.session.session import Session

_log = logging.getLogger(__name__)


class SessionBusyError(Exception):
    """Raised when run() is called on a session that is already executing."""
    pass


class SessionManager:
    """Manages multiple Session instances with lifecycle operations.

    Usage::

        sm = SessionManager(sessions_dir="./sessions")
        s = sm.create()
        queue = sm.run(s.id, "build a test suite")
        # read events from queue for SSE streaming
    """

    def __init__(self, sessions_dir: str = "./sessions"):
        self._sessions_dir = sessions_dir
        self._sessions: dict[str, Session] = {}

    # ------------------------------------------------------------------
    # create / get / list / remove
    # ------------------------------------------------------------------

    def create(self) -> Session:
        """Create a new session and register it in memory."""
        session = Session(base_dir=self._sessions_dir)
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
        - "running": session.is_running is true
        - "idle": in _sessions but not running
        - "parked": on disk but not in _sessions
        """
        seen: set[str] = set()
        result: list[dict] = []

        for sid, session in self._sessions.items():
            seen.add(sid)
            result.append({
                "id": sid,
                "status": "running" if session.is_running else "idle",
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
                    })

        return result

    def remove(self, session_id: str) -> None:
        """Pause if running, remove from memory, and delete the DB file."""
        session = self._sessions.get(session_id)
        if session is not None and session.is_running:
            session.pause()

        if session_id in self._sessions:
            self._sessions.pop(session_id)

        # Delete DB file
        db_path = os.path.join(self._sessions_dir, f"{session_id}.db")
        if os.path.isfile(db_path):
            os.remove(db_path)

    # ------------------------------------------------------------------
    # run / pause
    # ------------------------------------------------------------------

    @logged(_log)
    def run(self, session_id: str, user_input: str | None) -> asyncio.Queue:
        """Start a background run of *session_id* with *user_input*.

        Returns the session's event queue for SSE streaming.
        Raises SessionBusyError if the session is already running.
        """
        session = self.get(session_id)
        if session is None:
            raise LookupError(f"Session {session_id} not found")
        if session.is_running:
            raise SessionBusyError(f"Session {session_id} is already running")

        return session.run(user_input)

    def pause(self, session_id: str) -> None:
        """Signal the session's run loop to stop at the next safe point."""
        session = self._sessions.get(session_id)
        if session is not None:
            session.pause()
