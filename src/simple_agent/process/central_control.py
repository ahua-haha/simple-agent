"""CentralControl — cursor-based state machine for task tree execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pi.ai.types import UserMessage, TextContent

from simple_agent.process.runners import BaseRunner, RunnerResult
from simple_agent.state.state import Task

if TYPE_CHECKING:
    from collections.abc import Callable


def _format_child_result(child: Task) -> list:
    """Format a finished child's results as messages for the parent."""
    msgs: list = []
    for r in child.result or []:
        msgs.append(
            UserMessage(
                content=[TextContent(text=f"[sub-task result] {r.desc}")],
                timestamp=0,
            )
        )
    return msgs


class CentralControl:
    """Cursor-based state machine for the task tree.

    Uses *tasks_by_id* for tree navigation (parent and child lookup by
    ID) and ``running_task`` object ref for fast cursor movement.

    Usage::

        cc = CentralControl(root, tasks_by_id, runners, checkpoint_fn)
        await cc.run()
    """

    def __init__(
        self,
        root: Task,
        tasks_by_id: dict[int, Task],
        runners: dict[str, BaseRunner],
        checkpoint_fn: Callable[[], None],
    ):
        self.cursor = root
        self._tasks = tasks_by_id
        self._runners = runners
        self._checkpoint = checkpoint_fn

    async def run(self) -> None:
        """Run the state machine to completion."""
        while True:
            runner = self._runners[self.cursor.type]
            result = await runner.run(self.cursor)
            self._checkpoint()

            if result.kind == "continue":
                continue

            elif result.kind == "finished":
                self._handle_finished()
                if self.cursor is None:
                    break

            elif result.kind == "sub_task":
                self._handle_sub_task(result)

    # ------------------------------------------------------------------
    # move handlers
    # ------------------------------------------------------------------

    def _handle_finished(self) -> None:
        """Absorb cursor into parent and move cursor up."""
        if self.cursor.parent_id is None:
            self.cursor = None
            return

        parent = self._tasks.get(self.cursor.parent_id)
        if parent is None:
            self.cursor = None
            return

        parent.messages.extend(_format_child_result(self.cursor))
        parent.finished_task_ids.append(self.cursor.id)
        parent.running_task_id = None
        parent.running_task = None
        self.cursor = parent

    def _handle_sub_task(self, result: RunnerResult) -> None:
        """Wire child into tree and move cursor down."""
        child = result.child
        if child is None:
            return

        child.parent_id = self.cursor.id
        self.cursor.running_task = child
        self.cursor.running_task_id = child.id
        self.cursor.state = "WAITING"
        self._tasks[child.id] = child
        self.cursor = child
