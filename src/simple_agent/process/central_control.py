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

    Owns a ``cursor`` pointing to the currently-executing task.
    Dispatches to the runner for ``cursor.type``, processes the signal,
    checkpoints on every move.

    Usage::

        cc = CentralControl(root, runners, checkpoint_fn)
        await cc.run()
    """

    def __init__(
        self,
        root: Task,
        runners: dict[str, BaseRunner],
        checkpoint_fn: Callable[[], None],
    ):
        self.cursor = root
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
        parent = self.cursor.parent
        if parent is None:
            # Root finished — signal completion.
            self.cursor = None
            return

        parent.messages.extend(_format_child_result(self.cursor))
        parent.finished_tasks.append(self.cursor)
        parent.running_task = None
        self.cursor = parent

    def _handle_sub_task(self, result: RunnerResult) -> None:
        """Wire child into tree and move cursor down."""
        child = result.child
        if child is None:
            return

        child.parent = self.cursor
        self.cursor.running_task = child
        self.cursor.state = "WAITING"
        self.cursor = child
