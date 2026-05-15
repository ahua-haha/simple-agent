"""CollectResultProcess - synthesize TextResults from exploration history."""

from __future__ import annotations

from pi.ai import get_model
from pi.agent.types import AgentMessage

from simple_agent.process.agent_process import AgentProcess
from simple_agent.db.db import Database
from simple_agent.models import register_custom_models, get_api_key
from simple_agent.state.state import TEXT_RESULT_JSON_SCHEMA, Task, TextResult
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.stream import stream_event


SYSTEM_PROMPT = """You are a result synthesizer. Review the conversation history,
use bash commands like tool-inspect, grep, sed, head to inspect
tool results, and record each useful outcome as TextResult.

Focus on WHAT was accomplished, not HOW. Each TextResult description MUST:
- Be a single concise sentence stating the outcome
- Mention specific artifacts by name (files, modules, functions, classes)
- Use past tense declarative form: "Found X", "Created Y", "Identified Z"
- Be self-contained — readable without seeing the tool calls

When done, respond with only FINISH. Do NOT generate verbose output.

Examples:
- record_textresult(desc="Found main entry point at src/main.py with FastAPI app", toolCallLogID=[3])
- record_textresult(desc="Identified 3 core modules: process, state, and tool", toolCallLogID=[1,2])
- record_textresult(desc="Created test suite covering 12 functions across 3 modules", toolCallLogID=[5,6,7])
"""


class CollectResultProcess:
    
    proc: AgentProcess

    def __init__(self, tools_mgr: ToolMgr | None = None, db: Database | None = None):
        self.tools_mgr = tools_mgr or ToolMgr()
        self._db = db or Database()
        self.message: list[AgentMessage] = []

        record_tool = self.tools_mgr.create_record_tool(
            model_class=TextResult,
            name="record_textresult",
            description="Record a TextResult instance with the tool call log ID referencing related tool executions",
            parameters=TEXT_RESULT_JSON_SCHEMA,
        )

        proc = AgentProcess(get_model("deepseek", "deepseek-v4-pro"))
        proc.agent.subscribe(stream_event)
        proc.add_tool(record_tool, store=True)
        for tool in self.tools_mgr.create_all_tools("."):
            proc.add_tool(tool)
        self.proc = proc

    async def process(self, task: Task, context: list[AgentMessage]) -> list[AgentMessage]:
        index = len(context)
        self.message = context

        await self.proc.step(SYSTEM_PROMPT, self.message,
            "Please review the conversation history and record all useful results as TextResult using the record_textresult tool. When done, respond with only FINISH.")
        new_messages, _, results = self.proc.result()
        self.message = new_messages

        items = results.get("record_textresult", [])
        if items:
            task.result = [i for i in items if isinstance(i, TextResult)]

        self._db.save_task(
            task_type="collect result",
            task_input=task.input,
            messages=self.message,
            results=task.result,
            status="finished",
        )

        return self.message[index:]
