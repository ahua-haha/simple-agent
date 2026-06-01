# Session Runner Refactor Design

Date: 2026-06-01

## Scope

Refactor only the `Session.run` execution path. Existing `ExploreRunner`,
`PlanRunner`, and older process flows are out of scope for this change unless a
small compatibility edit is required by the `AgentProcess` contract.

The goal is to simplify the agent runtime and move session-run state management
into a dedicated runner. `AgentProcess` should be a simple executor. The runner
should own persistence, checkpointing, messages, and task lifecycle decisions.

## Architecture

Introduce a `SessionRunner` for the `Session.run` workflow.

`Session` becomes a thin entry point:

1. Open or create the session database file.
2. Construct `TaskManager` on that database.
3. Construct `ToolExecutionLogger` on that database and task manager.
4. Construct `SessionRunner`.
5. Delegate `run(user_input)` to the runner.

`SessionRunner` owns:

- Loading and saving runner state.
- The session run state machine.
- The runtime message list.
- Tool assembly and central wrapping.
- User-task creation and finalization through `TaskManager`.
- Checkpointing after each atomic handler.

`AgentProcess` owns only agent-loop execution. It receives prompts, messages,
tools, and an optional cancel event, then returns the new messages produced by
the agent loop.

## Persistence

All data for a session run is stored in one session database file.

Existing `TaskManager` task data stays in that same database and remains the
source of truth for user tasks, todos, aggregate tasks, and task-owned ordered
items.

Add three runner-owned tables:

- `runner_state_metadata`: one row for runner lifecycle metadata.
- `runner_messages`: ordered message history used to rebuild runtime context.
- `runner_tool_calls`: structured execution log rows written by the tool
  execution wrapper.

`runner_state_metadata` should include:

- `session_id`
- `phase`
- `status`
- `active_user_task_id`
- `last_error`
- `created_at`
- `updated_at`
- `version`

`runner_messages` should include:

- `id`
- `session_id`
- `seq`
- `role`
- `content_json`
- `timestamp_ms`

Do not add extra message columns in the first implementation. Store message
payloads in `content_json` so they can round-trip without depending on display
formatting.

`runner_tool_calls` should include:

- `id`
- `session_id`
- `tool_call_id`
- `tool_name`
- `params_json`
- `result_json`
- `status`
- `started_at`
- `finished_at`
- `error`

Keep the existing generic `ToolCallRecord` for current task timeline and
tool-inspection behavior. `runner_tool_calls` is the runner-level execution log.

## State Machine

Keep the session runner state machine intentionally small.

Phases:

- `idle`: ready to start a user input.
- `running`: a session run is in progress.
- `done`: the run completed successfully.
- `error`: the run failed and the last error was persisted.

Handlers:

- `handle_idle(user_input)`: create the user task, initialize metadata, set the
  phase to `running`, and checkpoint.
- `handle_running(user_input)`: load messages, build tools, call
  `AgentProcess.run`, append returned messages, finalize the user task if there
  is no active todo, set the phase to `done`, and checkpoint.
- `handle_error(exc)`: store `last_error`, set the phase/status to `error`,
  checkpoint, and re-raise.

The main control loop routes the current phase to the appropriate handler until
it reaches `done` or raises from `error`.

Resume behavior is scoped to run boundaries. The runner can load metadata,
messages, task data, and tool logs from the same database. It does not need to
resume from the middle of a single `agent_loop` call in this refactor.

## AgentProcess Contract

Change `AgentProcess.run` to remove `AgentState`.

Target shape:

```python
async def run(
    system_prompt: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    user_prompt: str = "",
    cancel_event: asyncio.Event | None = None,
) -> list[AgentMessage]:
    ...
```

`AgentProcess.run` should:

- Build the `AgentContext`.
- Pass `cancel_event` through to `agent_loop`.
- Emit existing stream events to subscribers.
- Collect messages from `AgentEndEvent`.
- Return only the new messages.

`AgentProcess.run` should not:

- Accept or create `AgentState`.
- Return tool-call records.
- Mutate runner, task, or database state directly.
- Know about checkpoints or session phases.

Caller-owned state is modified through tools bound to domain objects. For
example, task tools created by `TaskManager` mutate the task manager during tool
execution.

## Tool Logging

`ToolExecutionLogger` remains the central wrapper for tool execution.

When a tool is wrapped, execution should:

1. Record start time.
2. Run the original tool.
3. Record output, finish time, and success status in `runner_tool_calls`.
4. Preserve the existing tool-call record behavior used by tool inspection.
5. Notify `TaskManager.record_tool_call(log_id)` when a task manager is present.
6. Record failed executions with status `error` and the error string, then
   re-raise.

Coding tools should be created raw. `SessionRunner` assembles the full tool list
and wraps all tools in one central place.

## SessionRunner Tool Set

For the session-run path, the runner should assemble:

- `TaskManager.create_create_todo_tool()`
- `TaskManager.create_finish_todo_tool()`
- `TaskManager.create_error_todo_tool()`
- raw coding tools from `create_all_coding_tools(".")`

The runner then wraps the entire list with `ToolExecutionLogger.wrap_tools`.

## Error Handling

Errors from the agent loop or tool execution should be stored on
`runner_state_metadata.last_error` before being re-raised to the caller.

Tool execution errors should also create a `runner_tool_calls` row with
`status="error"` when enough information is available to identify the tool call.

The runner should always clear its in-memory running flag and notify the event
queue lifecycle in the surrounding `Session` path consistently with current
behavior.

## Testing

Focused tests should cover:

- `AgentProcess.run` no longer requires `AgentState` and returns messages only.
- `Session.run` delegates the workflow to `SessionRunner`.
- `SessionRunner` creates a user task, calls the agent once, persists runner
  metadata and messages, and finalizes the user task.
- `ToolExecutionLogger` writes `runner_tool_calls` for successful executions.
- `ToolExecutionLogger` writes `runner_tool_calls` for failed executions and
  re-raises.
- Runner load/resume behavior reads metadata and messages from the same session
  database.
- Existing task-manager, task-tool, session-manager, and diff-tool tests remain
  green.

## Non-Goals

- Do not refactor `ExploreRunner` or `PlanRunner` beyond compatibility changes
  required by the new `AgentProcess.run` signature.
- Do not introduce nested tasks or multiple active todos.
- Do not add automatic task compaction triggers.
- Do not make `AgentProcess` responsible for persistence or state routing.
