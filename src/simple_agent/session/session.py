"""Session — stores a task tree and session metadata in SQLite."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid

from pi.ai import get_model

from simple_agent.log import logged
from simple_agent.process.central_control import CentralControl
from simple_agent.process.agent_process import AgentProcess
from simple_agent.process.runners import CollectRunner, SingleRunRunner
from simple_agent.process.explore_runner import ExploreRunner
from simple_agent.process.plan_runner import PlanRunner
from simple_agent.snapshot.ghost_indexer import RepoWatcher
from simple_agent.state.state import Task
from simple_agent.task_manager import TaskManager
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.db.db import Database
from simple_agent.models import register_custom_models

_log = logging.getLogger(__name__)


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
        self._tools_mgr = ToolMgr(self._db, task_manager=self._task_manager)
        register_custom_models()
        self._agent_process = AgentProcess(get_model("deepseek", "deepseek-v4-pro"))

        runners = {
            "plan": PlanRunner(self._db, self._tools_mgr, self._agent_process),
            "explore": ExploreRunner(self._db, self._tools_mgr, self._agent_process),
            "collect": CollectRunner(self._db, self._tools_mgr, self._agent_process),
            "single_run": SingleRunRunner(self._db, self._tools_mgr, self._agent_process),
        }
        self._cc = CentralControl(self._db, runners)

        self._cursor: Task | None = None
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
    def root(self) -> Task | None:
        rows = self._db.load_all_tasks()
        if rows:
            tasks = Task.from_db_rows(rows)
            for task in tasks.values():
                if task.parent_id is None:
                    return task
        return None

    def _load_cursor(self) -> Task | None:
        """Load the cursor task from DB by cursor_id."""
        if self._cursor_id is None:
            return None
        record = self._db.get_task(self._cursor_id)
        if record is None:
            return None
        tasks = Task.from_db_rows([record])
        return tasks.get(self._cursor_id)

    def _ensure_task_metadata(self, task: Task) -> None:
        """Pre-load all runtime metadata for *task* so runners don't need to.

        Populates ``task.metadata["context_msgs"]`` (ancestor message chain)
        and, for explore tasks, ``task.metadata["repo_watcher"]``.
        """
        if "context_msgs" not in task.metadata:
            current_id = task.parent_id
            ancestor_rows: list = []
            while current_id is not None:
                record = self._db.get_task(current_id)
                if record is None:
                    break
                ancestor_rows.append(record)
                current_id = record.parent_id
            ancestor_rows.reverse()
            tasks_by_id = Task.from_db_rows(ancestor_rows) if ancestor_rows else {}
            task.metadata["context_msgs"] = (
                task.context(tasks_by_id) if tasks_by_id else list(task.messages)
            )

        if task.type == "explore" and "repo_watcher" not in task.metadata:
            task.metadata["repo_watcher"] = RepoWatcher(
                task.repo_path, "./data/snapshots"
            )

    @property
    def is_running(self) -> bool:
        return self._running

    @logged(_log)
    async def run(self, user_input: str) -> Task | None:
        """Run the task tree until finished, paused, or cancelled.

        Returns the root Task on completion, or None if paused.
        *step_timeout* caps each ``_cc.run()`` call; None disables the cap.
        """
        if self._cursor is None:
            self._cursor = self._load_cursor()

        if self._cursor is None:
            self._cursor = Task(input=user_input, state="PENDING", type="explore")
            self._cursor_id = self._db.upsert_task(self._cursor)
            self._cursor.id = self._cursor_id
            self._checkpoint()

        self._running = True
        if self.event_queue is None:
            self.event_queue = asyncio.Queue()

        try:
            while self._cursor is not None:
                if self._cancel_event.is_set():
                    break

                self._ensure_task_metadata(self._cursor)
                new_cursor, updates, inserts = await self._cc.run(self._cursor)

                self._cursor = new_cursor
                self._cursor_id = self._cursor.id if self._cursor else None
                self._checkpoint(updates=updates, inserts=inserts)

                if self._cancel_event.is_set():
                    break
        except Exception:
            _log.exception("run: session=%s failed", self._id)
            import traceback
            traceback.print_exc()
            if self.event_queue is not None:
                self.event_queue.put_nowait({"type": "error"})
            raise
        finally:
            self._running = False
            if self.event_queue is not None:
                self.event_queue.put_nowait(None)
                self.event_queue = None

        if self._cursor is None:
            result = self.root
            _log.info("run: session=%s done, result=%s", self._id, result.id if result else None)
            return result

        _log.info("run: session=%s paused", self._id)
        return None  # paused

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
        """Free the cursor from memory, keep only persisted metadata.

        Sets ``self._cursor = None`` to release the Task object.  The
        *cursor_id* stays in the DB checkpoint so ``run()`` can
        reload it on next call.

        Raises RuntimeError if the session is currently running.
        """
        if self._running:
            raise RuntimeError("Cannot park while session is running")
        self._cursor = None
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
