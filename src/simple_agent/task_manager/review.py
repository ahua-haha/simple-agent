"""Task tree review rendering utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from simple_agent.task_manager.models import ManagedTask


TaskTreeReviewFormat = Literal["tree", "flat"]


@dataclass(frozen=True)
class ToolCallReview:
    name: str
    arguments: Any | None = None


@dataclass(frozen=True)
class TaskTreeReview:
    text: str
    tool_call_log_ids: dict[int, int]


class TaskTreeReviewRenderer:
    def __init__(
        self,
        *,
        format: TaskTreeReviewFormat,
        depth: int | None,
        tool_calls: Mapping[int, ToolCallReview],
    ):
        self._format = format
        self._depth = depth
        self._tool_calls = tool_calls
        self._lines: list[str] = ["Task tree:"]
        self._tool_call_log_ids: dict[int, int] = {}
        self._next_tool_call_seq = 1

    def render(self, user_task: ManagedTask) -> TaskTreeReview:
        self._render_task(user_task, depth=0)
        return TaskTreeReview(
            text="\n".join(self._lines),
            tool_call_log_ids=self._tool_call_log_ids,
        )

    def _render_task(self, task: ManagedTask, *, depth: int) -> None:
        self._append_task(task, depth=depth)
        if self._format == "flat":
            for tool_call in self._tool_calls_in_tree(task):
                self._append_tool_call(tool_call, depth=depth + 1)
            return

        if self._depth is not None and depth >= self._depth:
            return

        for child in task.children:
            if child.kind == "tool_call":
                self._append_tool_call(child, depth=depth + 1)
                continue
            self._render_task(child, depth=depth + 1)

    def _append_task(self, task: ManagedTask, *, depth: int) -> None:
        self._lines.append(f"{self._indent(depth)}- {task.kind} [{task.status}] {task.title}")
        if task.result:
            self._lines.append(f"{self._indent(depth + 1)}result: {task.result}")
        if task.error:
            self._lines.append(f"{self._indent(depth + 1)}error: {task.error}")

    def _append_tool_call(self, task: ManagedTask, *, depth: int) -> None:
        seq = self._next_tool_call_seq
        self._next_tool_call_seq += 1
        if task.tool_call_log_id is not None:
            self._tool_call_log_ids[seq] = task.tool_call_log_id

        details = self._tool_call_details(task)
        line = f"{self._indent(depth)}- tool_call {seq}. {details.name}"
        if details.arguments is not None:
            line += f" args: {self._format_arguments(details.arguments)}"
        self._lines.append(line)

    def _tool_call_details(self, task: ManagedTask) -> ToolCallReview:
        if task.tool_call_log_id is None:
            return ToolCallReview(name="unknown_tool")
        return self._tool_calls.get(task.tool_call_log_id, ToolCallReview(name="unknown_tool"))

    def _tool_calls_in_tree(self, task: ManagedTask) -> list[ManagedTask]:
        tool_calls: list[ManagedTask] = []
        stack = list(reversed(task.children))
        while stack:
            child = stack.pop()
            if child.kind == "tool_call":
                tool_calls.append(child)
            else:
                stack.extend(reversed(child.children))
        return tool_calls

    def _format_arguments(self, arguments: Any) -> str:
        if isinstance(arguments, str):
            return arguments
        if hasattr(arguments, "model_dump_json"):
            return arguments.model_dump_json()
        return json.dumps(arguments, separators=(",", ":"))

    def _indent(self, depth: int) -> str:
        return "  " * depth
