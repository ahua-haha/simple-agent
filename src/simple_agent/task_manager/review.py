"""Task tree rendering utilities."""

from __future__ import annotations

from typing import Literal

from simple_agent.task_manager.models import ManagedTask, ToolCallTask


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

    def render(self, root_task: ManagedTask) -> str:
        self._render_task(root_task, depth=0)
        return "\n".join(self._lines)

    def _render_task(self, task: ManagedTask, *, depth: int) -> None:
        self._append_task(task, depth=depth)
        if self._format == "flat":
            for tool_call in _flat_tool_calls(task):
                self._append_task(tool_call, depth=depth + 1)
            return

        if self._depth is not None and depth >= self._depth:
            return

        for child in task.children:
            self._render_task(child, depth=depth + 1)

    def _append_task(self, task: ManagedTask, *, depth: int) -> None:
        sequence = None
        tool_call = None
        if isinstance(task, ToolCallTask):
            sequence = self._next_tool_call_seq
            self._next_tool_call_seq += 1

        self._lines.append(
            f"{self._indent(depth)}- {task.format_for_render(tool_call=tool_call, sequence=sequence)}"
        )
        if task.kind == "tool_call":
            return
        result = getattr(task, "result", None)
        error = getattr(task, "error", None)
        if result:
            self._lines.append(f"{self._indent(depth + 1)}result: {result}")
        if error:
            self._lines.append(f"{self._indent(depth + 1)}error: {error}")

    def _indent(self, depth: int) -> str:
        return "  " * depth


def _flat_tool_calls(task: ManagedTask) -> list[ToolCallTask]:
    tool_calls: list[ToolCallTask] = []
    stack = list(reversed(task.children))
    while stack:
        child = stack.pop()
        if isinstance(child, ToolCallTask):
            tool_calls.append(child)
        else:
            stack.extend(reversed(child.children))
    return tool_calls

