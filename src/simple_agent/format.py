"""Shared format_result_message — consistent output across all process types."""

from pi.agent.types import AgentMessage
from pi.ai.types import AssistantMessage, TextContent

from simple_agent.db.db import Database
from simple_agent.state.state import Task


def format_results(
    db: Database,
    task: Task,
    status: str = "finished",
    label: str | None = None,
) -> list[AgentMessage]:
    text_results = task.result or []
    task_label = label or f"the task: {task.input}"

    result: list[AgentMessage] = []

    status_text = "successfully completed" if status == "finished" else "failed to complete"
    desc_lines = [f"- {tr.desc}" for tr in text_results]
    desc_text = "\n".join(desc_lines) if desc_lines else "(no results)"
    result.append(AssistantMessage(
        content=[TextContent(text=f"{status_text} {task_label}\nthe result of the task are as follows:\n{desc_text}")]
    ))

    return result
