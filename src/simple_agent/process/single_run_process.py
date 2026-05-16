"""SingleRunProcess - directly execute tools, then collect results."""

from __future__ import annotations

from pi.agent.types import AgentMessage

from simple_agent.process.collect_result_process import CollectResultProcess
from simple_agent.process.agent_process import AgentProcess
from simple_agent.snapshot.ghost_indexer import RepoWatcher
from simple_agent.state.state import Task, StateClarification
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.db.db import Database
from simple_agent.stream import stream_event
from pi.ai import get_model


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

    def __init__(self, tools_mgr: ToolMgr | None = None, db: Database | None = None):
        self.tools_mgr = tools_mgr or ToolMgr()
        self._db = db or Database()
        self.message: list[AgentMessage] = []

        determine_state_tool = self.tools_mgr.create_record_tool(
            model_class=StateClarification,
            name="determine_state",
            description="Determine the current state based on context. States: finished (task complete), error (task failed)",
            parameters={
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "Available states:\n- finished: task complete\n- error: task failed",
                        "enum": ["finished", "error"],
                    },
                    "reason": {"type": "string", "description": "Reason for choosing this state"},
                },
                "required": ["state", "reason"],
            },
        )

        proc = AgentProcess(get_model("deepseek", "deepseek-v4-pro"))
        proc.agent.subscribe(stream_event)
        proc.add_tool(determine_state_tool, on_call=lambda self: self.stop_agent("determine_state"), store=True)
        proc.add_tool(self.tools_mgr.create_all_tools("."))
        self.proc = proc

    def format_result_message(self, task: Task, state: str = "finished") -> list[AgentMessage]:
        from simple_agent.format import format_results
        return format_results(self.tools_mgr, task, status=state)

    async def try_run(self, task: Task) -> StateClarification | None:
        await self.proc.step(SYSTEM_PROMPT, self.message, task.input)
        new_messages, _, results = self.proc.prune("determine_state").result()
        self.message = new_messages

        items = results.get("determine_state", [])
        if items and isinstance(items[-1], StateClarification):
            return items[-1]
        return None

    async def process(self, task: Task, context: list[AgentMessage] = []) -> list[AgentMessage]:
        self.proc.agent.reset()

        index = len(context)
        self.message = context

        if task.result is None:
            task.result = []

        if task.repo_watcher is None:
            task.repo_watcher = RepoWatcher(".", "./data/snapshots")
        task.start_snapshot = task.repo_watcher.take_snapshot()

        state_result = await self.try_run(task)

        task.end_snapshot = task.repo_watcher.take_snapshot()

        collectProc = CollectResultProcess(tools_mgr=self.tools_mgr, db=self._db)
        await collectProc.process(task, self.message[index:])

        state = "finished"
        if state_result is not None:
            state = state_result.state

        self._db.save_task(
            task_type="single_run",
            task_input=task.input,
            messages=self.message,
            results=task.result,
            status=state,
        )

        return self.format_result_message(task, state=state)
