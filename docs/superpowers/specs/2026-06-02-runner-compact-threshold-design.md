# Runner Compact Threshold Design

## Goal

Add run logic that monitors the active user task after each agent turn. If the
current user-task context grows past a configured token threshold, or the loaded
task tree grows past a configured tool-call threshold, the runner pauses normal
execution, persists a `compact` phase, and routes into compact handling.

The compact handler replaces the current runner message history with one compact
summary message, then returns the runner to `running` so the agent can continue
with smaller context.

## Architecture

`SessionRunner` remains the owner of session run state and control flow.
`AgentProcess` remains a message/tool executor. The task manager remains an
in-memory task tree that saves only at runner checkpoints.

`RunnerPhase` gains one new value: `compact`.

The phase router becomes:

- `idle`: create a user task, persist metadata, enter `running`.
- `running`: execute one or more agent turns using `AgentProcess.run`.
- `compact`: replace the active context with a compact summary, persist it, and
  return to `running`.
- `done`: stop and return the completed user task.
- `error`: persist failure state.

The synchronous `turn_end` hook remains the bridge from `AgentProcess` into the
runner. It saves new messages and task-manager state first. Then it checks the
thresholds using the runner's in-memory data. If either threshold is exceeded,
it sets `_phase = "compact"`, persists that phase, and sets the runner
`cancel_event` so the current agent loop stops cleanly.

## Threshold Inputs

The runner owns two threshold fields:

- `context_token_threshold`
- `tool_call_threshold`

Token counts use the project-local token estimation helpers against the
runner's `_messages` list. The current system has one active user task at a
time, so `_messages` is the current user-task message history.

Tool-call counts come from the loaded task manager, not from a fresh database
query. `TaskManager` exposes a simple in-memory count helper that walks the
active user task tree and counts `TaskItem(kind="tool_call")` entries.

## Compaction

`handle_compact` creates one compact summary message that preserves the useful
state needed for continuation:

- original user task title/request
- current user task status
- active todo, if any
- visible todos and their statuses/results/errors
- compact tool-call history from the task tree

The summary replaces the previous runner message rows for the session. This
requires a database helper such as `replace_runner_messages(session_id,
messages, session=...)` that deletes existing message rows for that session and
inserts the replacement list with fresh sequence numbers.

The compact save uses the existing composed transaction style:

1. open one database session
2. replace runner messages
3. save task-manager data
4. save runner metadata
5. commit once

After a successful compact save, the runner updates `_messages` to the compact
message list, clears the cancel event, sets `_phase = "running"`, and persists
the updated phase.

## Error Handling

Threshold checks happen only after a completed turn is saved. Partial stream
events do not trigger compaction.

If compaction fails, the runner uses the existing error path: set `_phase =
"error"`, persist `last_error`, and re-raise the exception.

## Testing

Focused tests should cover:

- token threshold exceeded at `turn_end` persists `compact` and sets the cancel
  event
- tool-call threshold exceeded at `turn_end` persists `compact` and sets the
  cancel event
- below-threshold runs keep the existing normal flow
- `handle_compact` replaces stored runner messages rather than appending
- compact handling clears the cancel event and routes back to `running`
- task manager tool-call counting uses loaded in-memory task data

## Out Of Scope

This design does not introduce provider-specific tokenization. The existing
heuristic estimator remains the threshold signal.

This design does not add external scheduling for compaction. The trigger is the
runner's turn-end threshold check.
