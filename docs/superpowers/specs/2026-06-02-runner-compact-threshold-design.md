# Runner Compact Threshold Design

## Goal

Add run logic that monitors the active user task after each agent turn. If the
current user-task context grows past a configured token threshold, or the runner
tool-call log grows past a configured tool-call threshold, the runner pauses normal
execution, persists a `compact` phase, and routes into compact handling.

The compact handler is intentionally left as a TODO in this change. This keeps
the first implementation focused on detecting the threshold crossing and routing
to the compact phase without changing task-manager behavior.

## Architecture

`SessionRunner` remains the owner of session run state and control flow.
`AgentProcess` remains a message/tool executor. The task manager remains an
in-memory task tree that saves only at runner checkpoints.

`RunnerPhase` gains one new value: `compact`.

The phase router becomes:

- `idle`: create a user task, persist metadata, enter `running`.
- `running`: execute one or more agent turns using `AgentProcess.run`.
- `compact`: call the compact handler placeholder.
- `done`: stop and return the completed user task.
- `error`: persist failure state.

The synchronous `turn_end` hook remains the bridge from `AgentProcess` into the
runner. It saves new messages and task-manager state first. Then it checks the
thresholds. If either threshold is exceeded,
it sets `_phase = "compact"`, persists that phase, and sets the runner
`cancel_event` so the current agent loop stops cleanly.

## Threshold Inputs

The runner owns two threshold fields:

- `context_token_threshold`
- `tool_call_threshold`

Token counts use the project-local token estimation helpers against the
runner's `_messages` list. The current system has one active user task at a
time, so `_messages` is the current user-task message history.

Tool-call counts come from the runner tool-call log table for the current
session. This avoids adding task-manager APIs for the first threshold-routing
implementation.

## Compaction

`handle_compact` is a TODO placeholder in this change and raises
`NotImplementedError`. The future compaction implementation can decide how to
replace messages, aggregate task context, and return to `running`.

## Error Handling

Threshold checks happen only after a completed turn is saved. Partial stream
events do not trigger compaction.

Because `handle_compact` is currently a TODO, calling it raises
`NotImplementedError`. The runner's existing top-level error path persists that
as an `error` phase if it is reached through `run()`.

## Testing

Focused tests should cover:

- token threshold exceeded at `turn_end` persists `compact` and sets the cancel
  event
- tool-call threshold exceeded at `turn_end` persists `compact` and sets the
  cancel event
- below-threshold runs keep the existing normal flow
- `handle_compact` is a TODO placeholder

## Out Of Scope

This design does not introduce provider-specific tokenization. The existing
heuristic estimator remains the threshold signal.

This design does not add external scheduling for compaction. The trigger is the
runner's turn-end threshold check.

This design does not change the task manager.
