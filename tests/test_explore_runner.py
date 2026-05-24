"""Integration test for ExploreRunner — runs against a real LLM."""

from __future__ import annotations

import tempfile

import pytest

from pi.ai import get_model

from simple_agent.process.agent_process import AgentProcess
from simple_agent.process.explore_runner import ExploreRunner
from simple_agent.state.state import Task
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.db.db import Database
from simple_agent.models import register_custom_models
from simple_agent.stream import stream_event


@pytest.mark.asyncio
async def test_explore_runner_integration():
    """Run ExploreRunner end-to-end with a real LLM.

    Creates a task, executes Phase 1 (explore), then Phase 2 (collect),
    and verifies results are accumulated.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    register_custom_models()
    model = get_model("deepseek", "deepseek-v4-pro")
    agent_process = AgentProcess(model)
    agent_process.subscribe(stream_event)
    db = Database(db_path)
    tools_mgr = ToolMgr(db)

    runner = ExploreRunner(db, tools_mgr, agent_process)

    task = Task(
        input="list the files in the current directory",
        type="explore",
        state="RUNNING",
        messages=[],
        result=[],
    )
    task.id = db.upsert_task(task)
    assert task.id is not None

    # Phase 1: execute
    result = await runner.run(task)
    assert result.kind == "continue"
    assert task.state == "context_complete"
    assert len(task.messages) > 0
    print("\n===== Phase 1 messages =====")
    for m in task.messages:
        print(f"  [{m.role}] {m}")

    # Phase 2: collect
    result = await runner.run(task)
    assert result.kind == "finished"
    assert task.state == "FINISHED"
    print("\n===== Phase 2 messages =====")
    for m in task.messages:
        print(f"  [{m.role}] {m}")
    print(f"\n===== Results: {task.result} =====")
