"""SingleRunProcess - directly execute tools, then collect results."""

from __future__ import annotations

from pi.ai import get_model

from simple_agent.process.collect_result_process import CollectResultProcess
from simple_agent.process.agent_process import AgentProcess
from simple_agent.snapshot.ghost_indexer import RepoWatcher
from simple_agent.state.state import Task, StateClarification, SessionState
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.db.db import Database
from simple_agent.stream import stream_event


SYSTEM_PROMPT = """You are a helpful assistant. Use the available tools to directly accomplish the user's task.
<important>
When the task is complete and no further tool calls are required, you MUST use 'determine_state' tool to determine the state BEFORE your final response.
</important>

<example>
tool call 1 ...
tool call 1 result ...
tool call 2 ...
tool call 2 result ...

Now the context information is complete. use 'determine_state' tool call to determine the state
Final response: ...
</example>
"""


class SingleRunProcess:

    proc: AgentProcess

    def __init__(self, tools_mgr: ToolMgr | None = None, db: Database | None = None,
                 agent_process: AgentProcess | None = None):
        self.tools_mgr = tools_mgr or ToolMgr()
        self._db = db or Database()
        self._agent_process = agent_process

        determine_state_tool = self.tools_mgr.create_determine_state_tool()

        proc = agent_process or AgentProcess(get_model("deepseek", "deepseek-v4-pro"))
        proc.subscribe(stream_event)
        proc.add_tool(determine_state_tool, on_call=lambda self: self.stop_agent("determine_state"), store=True)
        proc.add_tool(self.tools_mgr.create_all_tools("."))
        self.proc = proc

    def _append_format_results(self, task: Task, state: SessionState, status: str = "finished") -> None:
        from simple_agent.format import format_results
        state.messages.extend(format_results(self.tools_mgr, task, status=status))

    async def _try_run(self, task: Task, state: SessionState) -> StateClarification | None:
        new_messages, _, results = await self.proc.step(SYSTEM_PROMPT, task.messages or [], task.input)
        new_messages = self.proc.prune_messages(new_messages, "determine_state")
        task.messages.extend(new_messages)

        items = results.get("determine_state", [])
        if items and isinstance(items[-1], StateClarification):
            return items[-1]
        return None

    async def process(self, task: Task, state: SessionState) -> None:
        self.proc.reset()

        index = len(state.messages)

        if task.result is None:
            task.result = []

        if task.repo_watcher is None:
            task.repo_watcher = RepoWatcher(".", "./data/snapshots")
        task.start_snapshot = task.repo_watcher.take_snapshot()

        state_result = await self._try_run(task, state)

        task.end_snapshot = task.repo_watcher.take_snapshot()

        collectProc = CollectResultProcess(tools_mgr=self.tools_mgr, db=self._db, agent_process=self._agent_process)
        await collectProc.process(task, state)

        status = "finished"
        if state_result is not None:
            status = state_result.state

        self._db.save_task(
            task_type="single_run",
            task_input=task.input,
            messages=task.messages,
            results=task.result,
            status=status,
        )

        self._append_format_results(task, state, status=status)
        task.messages.clear()
