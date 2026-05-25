"""Integration test for PlanRunner — runs against a real LLM."""

from __future__ import annotations

import tempfile

import pytest

from pi.ai import get_model

from simple_agent.process.agent_process import AgentProcess
from simple_agent.process.explore_runner import ExploreRunner
from simple_agent.process.plan_runner import PlanRunner
from simple_agent.state.state import Task
from simple_agent.tool.tool_mgr import ToolMgr
from simple_agent.db.db import Database
from simple_agent.models import register_custom_models
from simple_agent.stream import stream_event


@pytest.mark.asyncio
async def test_plan_runner_integration():
    """Run PlanRunner end-to-end with a real LLM.

    The plan runner examines context and either creates a sub_task
    or calls determine_state to finish.  If it creates a sub_task,
    we run it through the ExploreRunner, then come back to the plan
    runner to simulate the CentralControl cursor loop.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    register_custom_models()
    model = get_model("deepseek", "deepseek-v4-pro")
    agent_process = AgentProcess(model)
    agent_process.subscribe(stream_event)
    db = Database(db_path)
    tools_mgr = ToolMgr(db)

    plan_runner = PlanRunner(db, tools_mgr, agent_process)
    explore_runner = ExploreRunner(db, tools_mgr, agent_process)

    plan_task = Task(
        input="there is a project at /root/workspace/simple-agent, explore its structure and summarize what you find",
        type="plan",
        state="RUNNING",
        messages=[],
        result=[],
    )
    plan_task.id = db.upsert_task(plan_task)

    max_iterations = 5
    for i in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"Plan iteration {i + 1}")
        print(f"{'='*60}")

        result = await plan_runner.run(plan_task)
        print(f"\nPlan result: kind={result.kind}")

        if result.kind == "finished":
            print("Plan complete!")
            break

        elif result.kind == "sub_task":
            child = result.child
            child.id = db.upsert_task(child)
            child.parent_id = plan_task.id
            plan_task.running_task = child
            plan_task.running_task_id = child.id
            plan_task.state = "WAITING"
            db.upsert_task(plan_task)

            print(f"\n--- Running sub-task: {child.input[:80]}... ---")
            # Phase 1: execute
            r = await explore_runner.run(child)
            print(f"Explore phase 1: kind={r.kind} state={child.state}")
            assert r.kind == "continue"

            # Phase 2: collect
            r = await explore_runner.run(child)
            print(f"Explore phase 2: kind={r.kind} state={child.state}")
            assert r.kind == "finished"

            print(f"Sub-task results: {[x.desc for x in (child.result or [])]}")

            # Absorb child into parent (simulating CentralControl._handle_finished)
            from simple_agent.process.central_control import _format_child_result
            plan_task.messages.extend(_format_child_result(child))
            plan_task.finished_task_ids.append(child.id)
            plan_task.running_task = None
            plan_task.running_task_id = None
            plan_task.state = "RUNNING"
            db.upsert_task(plan_task)
            db.upsert_task(child)

        else:
            print(f"Unexpected result: {result.kind}")
            break

    print(f"\n===== Final plan task state: {plan_task.state} =====")
    print(f"Messages count: {len(plan_task.messages)}")
    print(f"Finished sub-tasks: {plan_task.finished_task_ids}")
