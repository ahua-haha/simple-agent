"""Shared format_result_message — consistent output across all process types."""

from pi.agent.types import AgentMessage
from pi.ai.types import AssistantMessage, TextContent

from simple_agent.state.state import Task
from simple_agent.tool.tool_mgr import ToolMgr


def format_results(
    tools_mgr: ToolMgr,
    task: Task,
    status: str = "finished",
    label: str | None = None,
) -> list[AgentMessage]:
    text_results = task.result or []
    task_label = label or f"the task: {task.input}"

    result: list[AgentMessage] = []

    # 1. Recorded tool calls
    tool_log_ids: list[int] = []
    for tr in text_results:
        tool_log_ids.extend(tr.toolCallLogID)
    result.extend(tools_mgr.get_all_messages(tool_log_ids))

    # 2. Status AssistantMessage
    status_text = "successfully completed" if status == "finished" else "failed to complete"
    result.append(AssistantMessage(
        content=[TextContent(text=f"{status_text} {task_label}\nthe result of the task are as follows")]
    ))

    # 3. Each TextResult as individual AssistantMessage
    for tr in text_results:
        ids = ", ".join(str(i) for i in tr.toolCallLogID) if tr.toolCallLogID else "none"
        result.append(AssistantMessage(
            content=[TextContent(text=f"{tr.desc} [toolCallLogID: {ids}]")]
        ))

    return result
