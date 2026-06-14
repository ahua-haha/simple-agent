# Simple Agent

A task-driven AI coding agent with an orchestrator/task-agent lifecycle.

## Structure

```
├── backend/      # Python agent (task manager, session runner, web API)
├── frontend/     # Next.js web UI
├── logs/         # Session run logs (JSONL)
└── sessions/     # Session databases (SQLite)
```

## Backend

See [backend/README.md](backend/README.md) for setup and usage.

## Frontend

See [frontend/package.json](frontend/package.json) for scripts and dependencies.
