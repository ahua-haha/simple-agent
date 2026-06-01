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
from simple_agent.task_manager import TaskManager
from simple_agent.tool.execution_logger import ToolExecutionLogger
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
        task = await session.run("build a test suite")

        session2 = Session(session_id=s._id)        # reload existing
    """

    def __init__(self, session_id: str | None = None,
                 base_dir: str = "./sessions"):
        self._id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        self._base_dir = base_dir
        self._db_path = os.path.join(base_dir, f"{self._id}.db")
        self._db = Database(self._db_path)
        self._task_manager = TaskManager(self._db)
        self._execution_logger = ToolExecutionLogger(
            self._db,
            task_manager=self._task_manager,
            session_id=self._id,
        )
        register_custom_models()
        self._agent_process = AgentProcess(get_model("deepseek", "deepseek-v4-pro"))
        self._runner = SessionRunner(
            session_id=self._id,
            db=self._db,
            task_manager=self._task_manager,
            execution_logger=self._execution_logger,
            agent_process=self._agent_process,
            cancel_event=asyncio.Event(),
        )
        self._running = False
        self.event_queue: asyncio.Queue | None = None

        self._runner.subscribe(self._on_agent_event)

    def _on_agent_event(self, event) -> None:
        """Push agent events into the event queue if one is active."""
        _log.debug("agent event: %s", type(event).__name__)
        if self.event_queue is not None:
            self.event_queue.put_nowait(event)

    @property
    def is_running(self) -> bool:
        return self._running

    @logged(_log)
    async def run(self, user_input: str):
        """Run the persisted session runner for one user task."""
        self._running = True
        if self.event_queue is None:
            self.event_queue = asyncio.Queue()

        try:
            user_task = await self._runner.run(user_input)
        except Exception:
            _log.exception("run: session=%s failed", self._id)
            if self.event_queue is not None:
                self.event_queue.put_nowait({"type": "error"})
            raise
        finally:
            self._running = False
            if self.event_queue is not None:
                self.event_queue.put_nowait(None)
                self.event_queue = None

        _log.info("run: session=%s done, result=%s", self._id, user_task.id if user_task else None)
        return user_task

    def pause(self) -> None:
        """Signal the run loop to stop at the next safe point.

        The current transition completes, then the loop exits.
        Safe to call from any task / thread.
        """
        self._runner.pause()

    @property
    def id(self) -> str:
        return self._id
