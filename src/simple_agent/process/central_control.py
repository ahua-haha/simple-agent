"""CentralControl — single-transition state machine for task tree execution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from simple_agent.log import logged
from simple_agent.process.runners import BaseRunner, RunnerResult
from simple_agent.state.state import Task

if TYPE_CHECKING:
    from collections.abc import Callable

_log = logging.getLogger(__name__)


class CentralControl:
    """Single-transition state machine.

    ``run(cursor)`` executes one agent cycle and one state transition.
    This legacy runner no longer persists ``Task`` rows; session running is
    owned by ``SessionRunner`` and ``TaskManager``.
    """

    def __init__(self, db, runners: dict[str, BaseRunner]):
        self._db = db
        self._runners = runners

    @logged(_log)
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

        raise RuntimeError("CentralControl no longer supports persisted parent loading")

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
