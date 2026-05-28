"""CentralControl — single-transition state machine for task tree execution."""

from __future__ import annotations

from typing import TYPE_CHECKING


from simple_agent.process.runners import BaseRunner, RunnerResult
from simple_agent.state.state import Task
from simple_agent.db.db import Database

if TYPE_CHECKING:
    from collections.abc import Callable


class CentralControl:
    """Single-transition state machine.

    ``run(cursor)`` executes one agent cycle and one state transition.
    Returns ``(new_cursor, updates, inserts)`` — the exact tasks to
    persist.  Session owns the loop and cursor.

    Usage::

        cursor = db.load_active()
        while cursor is not None:
            new_cursor, updates, inserts = await cc.run(cursor)
            for t in updates: db.upsert_task(t)
            for t in inserts: db.upsert_task(t)
            cursor = new_cursor
    """

    def __init__(self, db: Database, runners: dict[str, BaseRunner]):
        self._db = db
        self._runners = runners

    async def run(self, cursor: Task) -> tuple[Task | None, list[Task], list[Task]]:
        """Execute one transition.

        Returns:
            (new_cursor, updates, inserts)
            new_cursor: the task to focus next, or None if root finished
            updates: tasks modified in this transition
            inserts: tasks newly created in this transition
        """
        runner = self._runners[cursor.type]
        result = await runner.run(cursor)

        if result.kind == "continue":
            return cursor, [cursor], []

        if result.kind == "finished":
            return self._handle_finished(cursor)

        if result.kind == "sub_task":
            return self._handle_sub_task(cursor, result.child)

        return cursor, [cursor], []

    # ------------------------------------------------------------------
    # move handlers
    # ------------------------------------------------------------------

    def _handle_finished(self, cursor: Task) -> tuple[Task | None, list[Task], list[Task]]:
        """Absorb finished cursor into parent, return parent as new cursor."""
        cursor.state = "FINISHED"
        cursor.metadata.clear()

        if cursor.parent_id is None:
            return None, [cursor], []

        record = self._db.get_task(cursor.parent_id)
        if record is None:
            return None, [cursor], []

        parent = Task.from_db_rows([record])[cursor.parent_id]

        parent.messages.extend(cursor.result_msg or [])
        parent.finished_task_ids.append(cursor.id)
        parent.running_task_id = None
        parent.running_task = None

        return parent, [cursor, parent], []

    def _handle_sub_task(self, cursor: Task, child: Task | None
                         ) -> tuple[Task | None, list[Task], list[Task]]:
        """Wire child into tree, return child as new cursor."""
        if child is None:
            return cursor, [cursor], []

        child.parent_id = cursor.id
        child.state = child.state or "PENDING"
        cursor.running_task = child
        cursor.running_task_id = child.id
        cursor.state = "WAITING"

        return child, [cursor], [child]
