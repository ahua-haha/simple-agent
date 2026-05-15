"""SubTaskProcess - planner agent that decomposes tasks into sub-tasks."""

from __future__ import annotations

from pi.ai import UserMessage, TextContent, get_model
from pi.agent.types import AgentMessage

from simple_agent.process.explore_process import ExploreProcess
from simple_agent.process.collect_result_process import CollectResultProcess
from simple_agent.process.agent_process import AgentProcess
from simple_agent.state.state import Task, StateClarification
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.db.db import Database
from simple_agent.stream import stream_event


SYSTEM_PROMPT = """You are a planner. Your job is to break down the user's task into sub-tasks
and execute them one at a time.

IMPORTANT: Each sub-task agent can ONLY see the input you give it — it does NOT see
the original user query or previous sub-task results. You MUST include sufficient
context in each sub-task's input so it can work independently. Include:
- The specific goal for this sub-task
- Relevant environment info (working directory, file paths mentioned in results)
- Key findings from previous sub-tasks that this sub-task needs to know
- Any constraints or preferences from the original user request

To define a sub-task: call 'define_task' with the task input (including context).
To finish: call 'determine_state' with state='finished' and a reason.

Example:
1. User asks: "build a test suite for this project"
2. define_task(input="Explore the project at /workspace/project: find all existing test files, identify the test framework used, and check which source modules have no test coverage.")
3. Sub-task executes, results come back showing pytest, 3 modules uncovered
4. define_task(input="Write unit tests for src/utils.py and src/models.py using pytest. These modules were identified as uncovered by the previous exploration. The project uses pytest with fixtures in conftest.py.")
5. Sub-task executes, results come back
6. determine_state(state="finished", reason="Test suite complete for uncovered modules")
"""


class SubTaskProcess:

    proc: AgentProcess

    def __init__(self, tools_mgr: ToolMgr | None = None, db: Database | None = None):
        self.tools_mgr = tools_mgr or ToolMgr()
        self._db = db or Database()
        self.message: list[AgentMessage] = []

        define_task_tool = self.tools_mgr.create_record_tool(
            model_class=Task,
            name="define_task",
            description="Define a sub-task to be executed. Include all necessary context: goal, environment info, relevant prior findings, and constraints.",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "The full input for this sub-task, including goal, environment context, relevant prior results, and constraints"},
                },
                "required": ["input"],
            },
        )
        determine_state_tool = self.tools_mgr.create_record_tool(
            model_class=StateClarification,
            name="determine_state",
            description="Determine the current state. States: finished (task complete), error (task failed)",
            parameters={
                "type": "object",
                "properties": {
                    "state": {"type": "string", "description": "Available states:\n- finished: task complete\n- error: task failed", "enum": ["finished", "error"]},
                    "reason": {"type": "string", "description": "Reason for choosing this state"},
                },
                "required": ["state", "reason"],
            },
        )

        proc = AgentProcess(get_model("deepseek", "deepseek-v4-pro"))
        proc.add_tool(define_task_tool, on_call=lambda self: self.stop_agent("define_task"), store=True)
        proc.add_tool(determine_state_tool, on_call=lambda self: self.stop_agent("determine_state"), store=True)
        proc.agent.subscribe(stream_event)
        self.proc = proc

    def format_result_message(self, task: Task, state: str = "finished") -> list[AgentMessage]:
        from simple_agent.format import format_results
        return format_results(self.tools_mgr, task, status=state)

    async def try_sub_task(self) -> Task | StateClarification | None:
        await self.proc.step(
            SYSTEM_PROMPT, self.message,
            "now based on the history, determine whether to define a sub task or this task is completed",
        )
        new_messages, finish_reason, results = self.proc.result()

        items = results.get("define_task", [])
        if items and isinstance(items[-1], Task):
            return items[-1]

        items = results.get("determine_state", [])
        if items and isinstance(items[-1], StateClarification):
            return items[-1]

        return None

    async def process(self, task: Task, context: list[AgentMessage] = []) -> list[AgentMessage]:

        index = len(context)
        self.message = context
        self.message.append(UserMessage(content=[TextContent(text=task.input)], timestamp=0))

        if task.result is None:
            task.result = []
        if task.subTasks is None:
            task.subTasks = []

        while True:
            res = await self.try_sub_task()

            if isinstance(res, Task):
                res.result = []
                task.subTasks.append(res)
                explore_proc = ExploreProcess(tools_mgr=self.tools_mgr, db=self._db)
                sub_msgs = await explore_proc.process(res, self.message[index+1:])
                self.message.extend(sub_msgs)
                continue

            if isinstance(res, StateClarification):
                collectProc = CollectResultProcess(tools_mgr=self.tools_mgr, db=self._db)
                await collectProc.process(task, self.message[index:])
                break

            break

        self._db.save_task(
            task_type="sub_task",
            task_input=task.input,
            messages=self.message,
            results=task.result,
            status="finished",
        )

        return self.format_result_message(task, state="finished")
