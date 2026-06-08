"""Lifecycle for writing durable repo memory through AgentIndex tools."""

from __future__ import annotations

import asyncio
import time
from typing import cast

from pi.agent import AgentTool
from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage

from simple_agent.index.indexer import AgentIndex
from simple_agent.message_store import MessageEntry
from simple_agent.run_log import runtime_logger
from simple_agent.task_manager.lifecycle import (
    BaseTaskLifecycle,
    SessionState,
    TaskLifecycleError,
    USER_TASK_SYSTEM_PROMPT,
    _assistant_has_tool_calls,
    _assistant_is_error,
    _next_task_action_text,
)
from simple_agent.task_manager.models import ManagedTask, RepoMemoryTask, ToolCallTask
from simple_agent.tool.common_tools import create_all_coding_tools


class RepoMemoryLifecycle(BaseTaskLifecycle):
    """Lifecycle that lets the agent inspect a repo and update AgentIndex."""

    def set_data(self, session_state: SessionState) -> None:
        self._session_state = session_state
        self.current_assistant_message_id = None
        task = self._session_state.next_task
        if task is None:
            raise TaskLifecycleError("Session state has no next task")
        if task.kind != "repo_memory":
            raise TaskLifecycleError("Active lifecycle task is not a repo memory task")
        self.task = cast(RepoMemoryTask, task)

    def clear_data(self) -> None:
        super().clear_data()
        self.task = None

    def instruction_text(self) -> str:
        return (
            "Runtime instruction for repo memory:\n"
            "- Write durable repo memory for the current repository.\n"
            f"- Repo path: {self.task.repo_path}\n"
            f"- AgentIndex database: {self.task.index_db_path}\n"
            "- Inspect files before writing memory.\n"
            "- Use index_tree to review existing repo memory.\n"
            "- Use index_upsert to record a short and concise description for each inspected entry.\n"
            "- Each description should say what each entry does, not how you found it.\n"
            "- Keep descriptions factual, specific, and brief enough to scan in the tree view.\n"
            "- When enough useful repo memory is written, respond without tool calls with a concise summary."
        )

    def create_tools(self) -> list[AgentTool]:
        index = AgentIndex(
            db_path=self.task.index_db_path,
            base_dir=self.task.repo_path,
        )
        return [
            *index.create_tools(),
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
            self._session_state.next_task_id_to_run = task.parent_id
            self._session_state.next_task = None
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
        assistant_message = await agent_process.call_llm_step(
            system_prompt=USER_TASK_SYSTEM_PROMPT,
            messages=[*self._session_state.message_values(), user_instruction_message],
            tools=tools,
            cancel_event=cancel_event,
        )
        if _assistant_is_error(assistant_message):
            raise TaskLifecycleError(
                assistant_message.error_message or "assistant response stopped with error"
            )

        assistant_entry = MessageEntry(
            id=self._session_state.allocate_message_id(),
            message=assistant_message,
        )
        tool_results: list[ToolResultMessage] = []
        tool_call_records: list[tuple[int, ToolCall | None, ToolResultMessage]] = []
        tool_call_tasks: list[ToolCallTask] = []
        if _assistant_has_tool_calls(assistant_message):
            self.current_assistant_message_id = assistant_entry.id
            try:
                tool_results = await agent_process.run_tool_calls_step(
                    tools=tools,
                    assistant_message=assistant_message,
                    cancel_event=cancel_event,
                )
            finally:
                self.current_assistant_message_id = None
            tool_call_records, tool_call_tasks = self._session_state.create_tool_call_record_task_entries(
                assistant_message=assistant_message,
                tool_result_messages=tool_results,
                parent_task=task,
            )
            task.children.extend(tool_call_tasks)
            if tool_call_tasks:
                task.touch()

        tool_result_entries = [
            MessageEntry(id=self._session_state.allocate_message_id(), message=tool_result)
            for tool_result in tool_results
        ]
        new_messages = [assistant_entry, *tool_result_entries]
        self._session_state.append_messages(new_messages)

        if not _assistant_has_tool_calls(assistant_message):
            task.status = "done"
            task.result = _assistant_text(assistant_message)
            task.touch()
            self._session_state.next_task_id_to_run = task.parent_id
            self._session_state.next_task = None
        elif task.status == "active":
            self._session_state.set_next_task(task, keep_instance=True)

        runtime_logger.log_handle_running(
            session_id=self._session_state._require_session_id(),
            messages=context_messages,
            user_instruction_message=user_instruction_message,
            assistant_message_id=assistant_entry.id,
            assistant_message=assistant_message,
            tool_result_entries=tool_result_entries,
            next_action=_next_task_action_text(self._session_state),
        )

        tasks_to_sync: list[ManagedTask] = [task, *task.children]
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
