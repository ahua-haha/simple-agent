# Runner Compact Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add runner threshold checks that persist a `compact` phase and stop the current agent loop when context tokens or runner tool-call records exceed configured limits.

**Architecture:** `SessionRunner` owns the thresholds and phase routing. Token counts use the project-local estimator over runner messages. Tool-call counts use the runner tool-call log table. `handle_compact` is intentionally left as a TODO placeholder; the task manager is unchanged.

**Tech Stack:** Python 3.14, SQLModel, pytest, pytest-asyncio, project-local token estimation helpers.

---

## File Structure

- Modify `src/simple_agent/session/runner.py`: add `compact` phase, threshold fields, turn-end threshold checks, and TODO `handle_compact`.
- Modify `tests/test_session_runner.py`: cover token threshold, tool-call threshold, and TODO compact handler behavior.
- Modify `docs/superpowers/specs/2026-06-02-runner-compact-threshold-design.md`: align the design with the reduced scope.

---

### Task 1: Add Compact Phase Threshold Routing

**Files:**
- Modify: `src/simple_agent/session/runner.py`
- Modify: `tests/test_session_runner.py`

- [ ] **Step 1: Write threshold tests**

Add tests that call `handle_running()` directly with low thresholds and assert:

```python
metadata.phase == "compact"
metadata.status == "compact"
cancel_event.is_set() is True
runner._phase == "compact"
```

For the tool-call threshold test, seed the runner DB log with
`db.insert_runner_tool_call(...)` before calling `handle_running()`.

- [ ] **Step 2: Implement threshold routing**

Add `RunnerPhase = Literal["idle", "running", "compact", "done", "error"]`,
default threshold constants, constructor fields, and a
`pause_for_compaction_if_needed()` helper that:

```python
context_tokens = estimate_messages_tokens(self._messages)
tool_calls = len(self._db.list_runner_tool_calls(self._session_id))
```

If either value exceeds its threshold, set `_phase = "compact"`, persist
metadata with `status="compact"`, and set `_cancel_event`.

- [ ] **Step 3: Add TODO compact handler**

Add compact phase routing in `run()` and implement:

```python
async def handle_compact(self, user_input: str) -> None:
    # TODO: implement compact handling.
    raise NotImplementedError("compact handling is not implemented yet")
```

- [ ] **Step 4: Run focused runner tests**

Run: `uv run pytest tests/test_session_runner.py -q`

Expected: PASS.

---

### Task 2: Verification

**Files:**
- No additional code changes expected.

- [ ] **Step 1: Run focused baseline**

Run:

```bash
uv run pytest tests/test_agent_process.py tests/test_session_runner.py tests/test_session.py tests/test_task_tools.py tests/test_execution_logger.py tests/test_session_manager.py tests/test_runner_storage.py tests/test_task_manager.py tests/test_token_estimation.py external/pi-agent/tests/test_agent.py external/pi-coding-agent/tests/test_tools.py -q
```

Expected: PASS.

- [ ] **Step 2: Run whitespace check**

Run: `git diff --check`

Expected: no output.

- [ ] **Step 3: Commit**

Run:

```bash
git add docs/superpowers/plans/2026-06-02-runner-compact-threshold.md docs/superpowers/specs/2026-06-02-runner-compact-threshold-design.md src/simple_agent/session/runner.py tests/test_session_runner.py
git commit -m "Add runner compact threshold routing"
```

Expected: commit succeeds.
