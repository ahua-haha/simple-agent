"""Session — stores a task tree in SQLite, runs via CentralControl."""

from __future__ import annotations

import asyncio
import json
import os

from pi.ai import get_model

from simple_agent.process.central_control import CentralControl
from simple_agent.process.agent_process import AgentProcess
from simple_agent.process.runners import CollectRunner, SingleRunRunner
from simple_agent.process.explore_runner import ExploreRunner
from simple_agent.process.plan_runner import PlanRunner
from simple_agent.snapshot.ghost_indexer import RepoWatcher
from simple_agent.state.state import Task
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.db.db import Database
from simple_agent.models import register_custom_models


class Session:
    """A session that stores a task tree in SQLite and runs via CentralControl.

    Session state (cursor_id) is persisted to a JSON file.  Task data is
    stored in the SQLite DB.

    Usage::

        session = Session("my-task")
        task = await session.run("build a test suite")
    """

    def __init__(self, name: str, base_dir: str = "./sessions"):
        self._name = name
        self._base_dir = base_dir
        self._db_path = os.path.join(base_dir, f"{name}.db")
        self._session_path = os.path.join(base_dir, f"{name}.json")
        self._db = Database(self._db_path)
        self._tools_mgr = ToolMgr(self._db)
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

        if os.path.exists(self._session_path):
            self._load_session()
        else:
            import time
            self._cursor_id: int | None = None
            self._created_at = time.time()
            self._updated_at = self._created_at

    def _load_session(self) -> None:
        """Load session metadata from file."""
        import time
        with open(self._session_path) as f:
            data = json.load(f)
        self._cursor_id = data.get("cursor_id")
        self._created_at = data.get("created_at", time.time())
        self._updated_at = data.get("updated_at", time.time())

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
        row = self._db.get_task(self._cursor_id)
        if row is None:
            return None
        tasks = Task.from_db_rows([row])
        return tasks.get(self._cursor_id)

    def _ensure_task_metadata(self, task: Task) -> None:
        """Pre-load all runtime metadata for *task* so runners don't need to.

        Populates ``task.metadata["context_msgs"]`` (ancestor message chain)
        and, for explore tasks, ``task.metadata["repo_watcher"]``.
        """
        if "context_msgs" not in task.metadata:
            current_id = task.parent_id
            ancestor_rows: list[dict] = []
            while current_id is not None:
                row = self._db.get_task(current_id)
                if row is None:
                    break
                ancestor_rows.append(row)
                current_id = row.get("parent_id")
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

    async def run(self, user_input: str) -> Task | None:
        """Run the task tree until finished, paused, or cancelled.

        Returns the root Task on completion, or None if paused.
        """
        if self._cursor is None:
            self._cursor = self._load_cursor()

        if self._cursor is None:
            self._cursor = Task(input=user_input, state="PENDING")
            self._cursor_id = self._db.upsert_task(self._cursor)
            self._cursor.id = self._cursor_id
            self._checkpoint()

        register_custom_models()
        self._running = True

        try:
            while self._cursor is not None:
                if self._cancel_event.is_set():
                    break

                self._ensure_task_metadata(self._cursor)
                new_cursor, updates, inserts = await self._cc.run(self._cursor)

                for t in updates:
                    self._db.upsert_task(t)
                for t in inserts:
                    self._db.upsert_task(t)

                self._cursor = new_cursor
                self._cursor_id = self._cursor.id if self._cursor else None
                self._checkpoint()

                if self._cancel_event.is_set():
                    break
        finally:
            self._running = False

        if self._cursor is None:
            return self.root

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
        *cursor_id* stays in the JSON checkpoint so ``run()`` can
        reload it from DB on next call.

        Raises RuntimeError if the session is currently running.
        """
        if self._running:
            raise RuntimeError("Cannot park while session is running")
        self._cursor = None
        self._cancel_event.clear()
        self.save()

    def save(self) -> str:
        """Persist session metadata to file.  Returns filepath."""
        import time
        self._updated_at = time.time()
        os.makedirs(os.path.dirname(self._session_path) or ".", exist_ok=True)
        data = {
            "name": self._name,
            "cursor_id": self._cursor_id,
            "created_at": self._created_at,
            "updated_at": self._updated_at,
        }
        with open(self._session_path, "w") as f:
            json.dump(data, f, indent=2)
        return self._session_path

    def _checkpoint(self) -> None:
        """Alias for save, used internally after each transition."""
        self.save()

    @staticmethod
    def list_sessions(base_dir: str = "./sessions") -> list[str]:
        if not os.path.isdir(base_dir):
            return []
        return sorted(
            f[:-3] for f in os.listdir(base_dir) if f.endswith(".db")
        )
