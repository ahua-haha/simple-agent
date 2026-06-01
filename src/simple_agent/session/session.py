"""Session — stores a task tree and session metadata in SQLite."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid

from pi.ai import get_model

from simple_agent.log import logged
from simple_agent.process.agent_process import AgentProcess, AgentState
from simple_agent.task_manager import TaskManager
from simple_agent.tool.common_tools import create_all_coding_tools
from simple_agent.tool.execution_logger import ToolExecutionLogger
from simple_agent.db.db import Database
from simple_agent.models import register_custom_models

_log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful coding agent.

Use create_todo before starting a coherent unit of work.
Call finish_todo when the active todo is complete.
Call error_todo if the active todo cannot be completed.
Keep responses concise and use available tools to do the work.
"""


class Session:
    """A session that stores a task tree and metadata in SQLite.

    Each session is identified by a unique ID.  The DB file is
    ``{session_id}.db`` inside *base_dir*.

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
        self._execution_logger = ToolExecutionLogger(self._db, task_manager=self._task_manager)
        register_custom_models()
        self._agent_process = AgentProcess(get_model("deepseek", "deepseek-v4-pro"))
        self._cancel_event = asyncio.Event()
        self._running = False
        self.event_queue: asyncio.Queue | None = None

        self._agent_process.subscribe(self._on_agent_event)
        self._load_session()

    def _on_agent_event(self, event) -> None:
        """Push agent events into the event queue if one is active."""
        _log.debug("agent event: %s", type(event).__name__)
        if self.event_queue is not None:
            self.event_queue.put_nowait(event)

    def _load_session(self) -> None:
        """Load session metadata from DB by session ID, or init defaults."""
        data = self._db.get_session(self._id)
        if data is not None:
            self._cursor_id = data.get("cursor_id")
            self._created_at = data.get("created_at", time.time())
            self._updated_at = data.get("updated_at", time.time())
        else:
            self._cursor_id: int | None = None
            self._created_at = time.time()
            self._updated_at = self._created_at

    @property
    def is_running(self) -> bool:
        return self._running

    @logged(_log)
    async def run(self, user_input: str):
        """Run the generic agent runtime for one user task."""
        self._running = True
        if self.event_queue is None:
            self.event_queue = asyncio.Queue()

        user_task = self._task_manager.create_user_task(user_input)
        self._cursor_id = user_task.id
        self._checkpoint()

        state = AgentState()
        tools = [
            self._task_manager.create_create_todo_tool(),
            self._task_manager.create_finish_todo_tool(),
            self._task_manager.create_error_todo_tool(),
            *create_all_coding_tools("."),
        ]
        tools = self._execution_logger.wrap_tools(tools)

        try:
            await self._agent_process.run(
                system_prompt=SYSTEM_PROMPT,
                messages=[],
                tools=tools,
                state=state,
                user_prompt=user_input,
            )
            if self._task_manager.active_todo_id is None:
                user_task = self._task_manager.finish_user_task()
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
        self._cancel_event.set()

    def resume(self) -> None:
        """Clear the pause signal so ``run()`` can proceed again."""
        self._cancel_event.clear()

    def park(self) -> None:
        """Clear in-memory signals and keep persisted session metadata."""
        if self._running:
            raise RuntimeError("Cannot park while session is running")
        self._cancel_event.clear()
        self._checkpoint()

    @property
    def id(self) -> str:
        return self._id

    def _checkpoint(self, updates=None, inserts=None) -> None:
        """Atomically persist tasks and session metadata in one transaction."""
        self._updated_at = time.time()
        all_tasks = (updates or []) + (inserts or [])
        with self._db._get_session() as s:
            for task in all_tasks:
                self._db.upsert_task(task, session=s)
            self._db.upsert_session(self._id, "", self._cursor_id, session=s)
            s.commit()
