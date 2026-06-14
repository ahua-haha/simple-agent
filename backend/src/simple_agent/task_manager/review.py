"""Task tree rendering utilities."""

from __future__ import annotations

from typing import Any, Literal


TaskTreeRenderFormat = Literal["tree", "flat"]


class TaskTreeRenderer:
    def __init__(
        self,
        *,
        format: TaskTreeRenderFormat,
        depth: int | None,
    ):
        self._format = format
        self._depth = depth
        self._lines: list[str] = ["Task tree:"]
        self._next_tool_call_seq = 1

    def render(self, root_task: Any) -> str:
        self._render_task(root_task, depth=0)
        return "\n".join(self._lines)

    def _render_task(self, task: Any, *, depth: int) -> None:
        self._append_task(task, depth=depth)
        if self._format == "flat":
            for tool_call in _flat_tool_calls(task):
                self._append_task(tool_call, depth=depth + 1)
            return

        if self._depth is not None and depth >= self._depth:
            return

        for child in getattr(task, "children", []):
            self._render_task(child, depth=depth + 1)

    def _append_task(self, task: Any, *, depth: int) -> None:
        sequence = None
        if getattr(task, "kind", None) == "tool_call":
            sequence = self._next_tool_call_seq
            self._next_tool_call_seq += 1

        self._lines.append(
            f"{self._indent(depth)}- {task.format_for_render(tool_call=None, sequence=sequence)}"
        )
        if getattr(task, "kind", None) == "tool_call":
            return
        result = getattr(task, "result", None)
        error = getattr(task, "error", None)
        if result:
            self._lines.append(f"{self._indent(depth + 1)}result: {result}")
        if error:
            self._lines.append(f"{self._indent(depth + 1)}error: {error}")

    def _indent(self, depth: int) -> str:
        return "  " * depth


def _flat_tool_calls(task: Any) -> list[Any]:
    tool_calls: list[Any] = []
    stack = list(reversed(getattr(task, "children", [])))
    while stack:
        child = stack.pop()
        if getattr(child, "kind", None) == "tool_call":
            tool_calls.append(child)
        else:
            stack.extend(reversed(getattr(child, "children", [])))
    return tool_calls


def build_task_tree(tasks: list[Any]) -> list[Any]:
    """Build in-memory task trees from flat persisted tasks without changing order."""
    by_id = {
        task.id: task
        for task in tasks
        if getattr(task, "id", None) is not None
    }
    roots: list[Any] = []
    for task in tasks:
        if hasattr(task, "children"):
            task.children = []

    for task in tasks:
        parent_id = getattr(task, "parent_id", None)
        if parent_id is not None and parent_id in by_id:
            by_id[parent_id].children.append(task)
        else:
            roots.append(task)
    return roots
