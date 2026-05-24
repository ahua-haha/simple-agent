"""Session — stores a task tree in SQLite, runs via CentralControl."""

from __future__ import annotations

import json
import os

from pi.ai import get_model

from simple_agent.process.central_control import CentralControl
from simple_agent.process.runners import PlanRunner, ExploreRunner, CollectRunner, SingleRunRunner
from simple_agent.state.state import Task
from simple_agent.db.db import Database
from simple_agent.models import register_custom_models

RUNNERS = {
    "plan": PlanRunner(),
    "explore": ExploreRunner(),
    "collect": CollectRunner(),
    "single_run": SingleRunRunner(),
}


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
        self._cc = CentralControl(self._db, RUNNERS)

        # Load cursor from session file, or default to None
        self._cursor_id: int | None = None
        if os.path.exists(self._session_path):
            with open(self._session_path) as f:
                data = json.load(f)
                self._cursor_id = data.get("cursor_id")

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

    async def run(self, user_input: str) -> Task:
        cursor = self._load_cursor()

        if cursor is None:
            cursor = Task(input=user_input, state="PENDING")
            self._cursor_id = self._db.upsert_task(cursor)
            cursor.id = self._cursor_id
            self._save_session()

        register_custom_models()

        while cursor is not None:
            new_cursor, updates, inserts = await self._cc.run(cursor)

            for t in updates:
                self._db.upsert_task(t)
            for t in inserts:
                self._db.upsert_task(t)

            cursor = new_cursor
            self._cursor_id = cursor.id if cursor else None
            self._save_session()

        return self.root

    def _save_session(self) -> None:
        """Persist session state (cursor_id) as JSON."""
        os.makedirs(os.path.dirname(self._session_path) or ".", exist_ok=True)
        with open(self._session_path, "w") as f:
            json.dump({"cursor_id": self._cursor_id}, f)

    @staticmethod
    def list_sessions(base_dir: str = "./sessions") -> list[str]:
        if not os.path.isdir(base_dir):
            return []
        return sorted(
            f[:-3] for f in os.listdir(base_dir) if f.endswith(".db")
        )
