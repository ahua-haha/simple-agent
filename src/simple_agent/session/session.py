"""Session — user-facing interaction wrapper for a persisted runner."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

from pi.ai import get_model

from simple_agent.log import logged
from simple_agent.process.agent_process import AgentProcess
from simple_agent.session.runner import SessionRunner
from simple_agent.db.db import Database
from simple_agent.models import register_custom_models

_log = logging.getLogger(__name__)


class Session:
    """A user-facing session wrapper.

    Each session is identified by a unique ID.  The DB file is
    ``{session_id}.db`` inside *base_dir*. Runtime state is owned by
    ``SessionRunner``.

    Usage::

        session = Session()                         # new session, auto ID
        queue = session.run("build a test suite")

        session2 = Session(session_id=s._id)        # reload existing
    """

    def __init__(self, session_id: str | None = None,
                 base_dir: str = "./sessions"):
        self._id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        self._base_dir = base_dir
        self._db_path = os.path.join(base_dir, f"{self._id}.db")
        self._db = Database(self._db_path)
        register_custom_models()
        self._agent_process = AgentProcess(get_model("deepseek", "deepseek-v4-pro"))
        self._runner = SessionRunner(
            session_id=self._id,
            db=self._db,
            agent_process=self._agent_process,
            cancel_event=asyncio.Event(),
        )
        self._running = False
        self._run_task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    @logged(_log)
    def run(self, user_input: str | None) -> asyncio.Queue:
        """Start the persisted runner and return the event queue."""
        if self._running:
            raise RuntimeError("Session is already running")
        self._running = True
        queue: asyncio.Queue = asyncio.Queue()

        def on_agent_event(event) -> None:
            _log.debug("agent event: %s", type(event).__name__)
            queue.put_nowait(event)

        self._runner.subscribe(on_agent_event)
        self._run_task = asyncio.create_task(self._run(user_input, queue, on_agent_event))
        return queue

    async def _run(self, user_input: str | None, queue: asyncio.Queue, on_agent_event) -> None:
        """Execute the runner and close the event queue when complete."""
        user_task = None
        try:
            user_task = await self._runner.run(user_input)
        except Exception:
            _log.exception("run: session=%s failed", self._id)
            queue.put_nowait({"type": "error"})
            raise
        finally:
            self._running = False
            self._run_task = None
            queue.put_nowait(None)
            self._runner.unsubscribe(on_agent_event)

        _log.info("run: session=%s done, result=%s", self._id, user_task.id if user_task else None)

    def pause(self) -> None:
        """Signal the run loop to stop at the next safe point.

        The current transition completes, then the loop exits.
        Safe to call from any task / thread.
        """
        self._runner.pause()

    @property
    def id(self) -> str:
        return self._id
