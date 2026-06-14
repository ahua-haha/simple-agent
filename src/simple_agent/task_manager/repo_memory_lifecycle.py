"""Lifecycle for writing durable repo memory through AgentIndex tools."""

from __future__ import annotations

import asyncio
import time
from typing import cast

from pi.agent import AgentTool
from pi.ai.types import AssistantMessage, TextContent, UserMessage

from simple_agent.index.indexer import AgentIndex
from simple_agent.run_log import runtime_logger
from simple_agent.task_manager.base_lifecycle import (
    BaseTaskLifecycle,
    SessionState,
    TaskLifecycleError,
    USER_TASK_SYSTEM_PROMPT,
    render_prompt_template,
)
from simple_agent.task_manager.models import RepoMemoryTask
from simple_agent.tool.common_tools import create_all_coding_tools


REPO_MEMORY_INSTRUCTION_TEMPLATE = """\
Runtime instruction for repo memory:
- Write durable repo memory for the current repository.
- Repo path: {{ repo_path }}
- AgentIndex database: {{ index_db_path }}
- Inspect files before writing memory.
- Use index_tree to review existing repo memory.
- Use index_upsert to record a short and concise description for each inspected entry.
- Each description should say what each entry does, not how you found it.
- Keep descriptions factual, specific, and brief enough to scan in the tree view.
- When enough useful repo memory is written, respond without tool calls with a concise summary.
"""


class RepoMemoryLifecycle(BaseTaskLifecycle):
    """Lifecycle that lets the agent inspect a repo and update AgentIndex."""

    task: RepoMemoryTask | None
    _agent_index: AgentIndex | None

    def __init__(self) -> None:
        self._agent_index = None

    def set_data(self, session_state: SessionState) -> None:
        self._session_state = session_state
        self.finished_task = None
        task = self._session_state.current_task
        if task is None:
            raise TaskLifecycleError("Session state has no current task")
        if task.kind != "repo_memory":
            raise TaskLifecycleError("Active lifecycle task is not a repo memory task")
        self.task = cast(RepoMemoryTask, task)
        self._agent_index = AgentIndex(
            db_path=self.task.index_db_path,
            base_dir=self.task.repo_path,
        )

    def clear_data(self) -> None:
        super().clear_data()
        self.task = None

    def instruction_text(self) -> str:
        return render_prompt_template(
            REPO_MEMORY_INSTRUCTION_TEMPLATE,
            repo_path=self.task.repo_path,
            index_db_path=self.task.index_db_path,
        )

    def create_tools(self) -> list[AgentTool]:
        return [
            *self._agent_index.create_tools(),
            *create_all_coding_tools(self.task.repo_path),
        ]

    async def run(
        self,
        *,
        agent_process,
        cancel_event: asyncio.Event | None = None,
    ) -> SessionState:
        task = self.task
        if task.status != "active":
            self._session_state.current_task_id = None
            self._session_state.current_task = None
            return self._session_state
        return await self.run_one_turn(
            agent_process=agent_process,
            cancel_event=cancel_event,
        )

    async def run_one_turn(
        self,
        *,
        agent_process,
        cancel_event: asyncio.Event | None = None,
    ) -> SessionState:
        task = self.task
        tools = self.create_tools()
        user_instruction_message = UserMessage(
            content=[TextContent(text=self.instruction_text())],
            timestamp=int(time.time() * 1000),
        )
        context_messages = list(self._session_state.messages)
        run_messages = [*self._session_state.message_values(), user_instruction_message]
        turn_result = await self.run_agent_turn(
            agent_process=agent_process,
            system_prompt=USER_TASK_SYSTEM_PROMPT,
            messages=run_messages,
            tools=tools,
            cancel_event=cancel_event,
        )
        assistant_message = turn_result.assistant_message
        assistant_entry = turn_result.assistant_entry
        tool_result_entries = turn_result.tool_result_entries
        tool_call_records = turn_result.tool_call_records
        tool_call_log_ids = turn_result.tool_call_log_ids
        if turn_result.has_tool_call:
            if tool_call_log_ids:
                task.touch()

        new_messages = [assistant_entry, *tool_result_entries]
        self._session_state.append_messages(new_messages)

        if not turn_result.has_tool_call:
            task.status = "done"
            task.result = _assistant_text(assistant_message)
            task.touch()
            self._session_state.current_task_id = None
            self._session_state.current_task = None
        elif task.status == "active":
            self.set_current_task(task.id, task)

        runtime_logger.log_handle_running(
            session_id=self._session_state._require_session_id(),
            messages=context_messages,
            user_instruction_message=user_instruction_message,
            assistant_message_id=assistant_entry.id,
            assistant_message=assistant_message,
            tool_result_entries=tool_result_entries,
        )

        tasks_to_sync = [task, *task.children]
        with self._session_state.create_database_session() as session:
            self._session_state.append_messages_to_database(
                messages=new_messages,
                session=session,
            )
            self._session_state.append_tool_calls_to_database(
                tool_calls=tool_call_records,
                session=session,
            )
            self._session_state.append_tasks_to_database(
                tasks=tasks_to_sync,
                session=session,
            )
            session.commit()
        return self._session_state


def _assistant_text(message: AssistantMessage) -> str | None:
    texts = [content.text for content in message.content if isinstance(content, TextContent)]
    return "\n".join(texts) if texts else None
