"""Shared format_result_message — consistent output across all process types."""

from pi.agent.types import AgentMessage
from pi.ai.types import AssistantMessage, TextContent
from pi.ai import ToolResultMessage

from simple_agent.state.state import Task, ToolExecMessage
from simple_agent.tool.execution_logger import ToolExecutionLogger


def format_results(
    execution_logger: ToolExecutionLogger,
    task: Task,
    status: str = "finished",
    label: str | None = None,
) -> list[AgentMessage]:
    text_results = task.result or []
    task_label = label or f"the task: {task.input}"

    result: list[AgentMessage] = []

    # Collect recorded tool call records
    tool_log_ids: list[int] = []
    for tr in text_results:
        tool_log_ids.extend(tr.toolCallLogID)
    records: list[ToolExecMessage] = execution_logger.get_all_messages(tool_log_ids)

    # 1. One single AssistantMessage with all recorded tool calls
    tool_calls = [r.tool_call for r in records]
    if tool_calls:
        result.append(AssistantMessage(
            content=list(tool_calls),
            stop_reason="tool_use",
        ))

    # 2. All tool call results
    for r in records:
        result.append(ToolResultMessage(
            tool_call_id=r.tool_call.id,
            tool_name=r.tool_call.name,
            content=r.tool_result.content,
            details=r.tool_result.details,
            is_error=False,
        ))

    # 3. One single AssistantMessage combining goal + all TextResult descriptions
    status_text = "successfully completed" if status == "finished" else "failed to complete"
    desc_lines = [f"- {tr.desc}" for tr in text_results]
    desc_text = "\n".join(desc_lines) if desc_lines else "(no results)"
    result.append(AssistantMessage(
        content=[TextContent(text=f"{status_text} {task_label}\nthe result of the task are as follows:\n{desc_text}")]
    ))

    return result
