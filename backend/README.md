# Simple Agent — Backend

A task-driven AI coding agent with an orchestrator/task-agent lifecycle, durable
repository memory, and a FastAPI web interface.

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────┐
│                SessionRunner                     │
│  phase-driven loop: orchestrator ⇄ common_task  │
└─────────────────────────────────────────────────┘
    │                    │
    ▼                    ▼
┌──────────────┐  ┌──────────────────┐
│ Orchestrator  │  │  CommonTask      │
│ Lifecycle     │  │  Lifecycle       │
│              │  │                  │
│ • task_plan   │  │ • instruction    │
│ • instruction │  │ • response       │
│ • finish_task │  │ • coding tools   │
│              │  │ • index_tree      │
└──────────────┘  └──────────────────┘
```

## Orchestrator / Task-Agent Workflow

The agent uses a ping-pong lifecycle between two phases:

### Orchestrator Phase (`orchestrator.py`)
Inspects the task state and decides next steps. Has three tools:

| Tool | Purpose |
|---|---|
| `set_instruction` | Assign one atomic task to the agent |
| `update_task_plan` | Maintain a markdown task list (`- [x]` done, `- [ ]` pending) |
| `finish_task` | Mark the entire task as complete |

**Rules:** Each instruction must contain exactly **one atomic task**. The
orchestrator decomposes complex work into single-step instructions and guides
the agent through them one by one.

### Common Task Phase (`task_lifecycle.py`)
Executes the orchestrator's instruction. Has:

- **Coding tools** — read, write, edit, bash, grep, find, ls
- **`index_tree`** — inspect the repository structure and existing index memory
- **`response_instruction`** — report back when the instruction is complete or errored

**Rules:** The agent must never work outside the scope of the current
instruction. It calls `response_instruction` immediately when done or on error.

### Routing
The runner's `while` loop reads `next_phase` from session state and dispatches:
`"orchestrator"` → `"common_task"` → `"orchestrator"` → … → `"done"` (exit).

## Index Memory System

A durable, repository-scoped memory store backed by SQLite (`.index.db`).

### What it stores
Concise descriptions of files, modules, classes, and markdown sections — indexed
by path. Each entry records what an item **does**, not how it was found.

### How it works
1. **`index_tree`** — renders a filtered, depth-limited tree view of the repository.
   Shows directory structure, Python symbols (classes/functions), markdown headings,
   and existing index descriptions beside each entry.
2. **`index_upsert`** — creates or updates a memory entry for a specific path. Accepts
   `kind` and `description` metadata.

### File Walkers
The index tree is built by walking the filesystem and parsing source files:

| Walker | Purpose |
|---|---|
| `dir_walker.py` | Walk a directory tree, respecting `.gitignore` |
| `python_walker.py` | Parse Python AST to extract classes and functions as tree nodes |
| `markdown_walker.py` | Parse markdown headings into a hierarchical tree |
| `tree.py` | Assemble and render the full ASCII tree with `#` descriptions |

### Auto-Commit
When the repo changes (detected via git diff), the index can automatically
expire entries for modified paths. The `commit(target_commit)` method expires
stale entries and tracks the current commit so the agent knows which memory
needs review.

## Repo Watcher

Tracks file changes between git commits. Used by the index memory system to
detect which index entries are stale after a code change.

- `parse_diff(from_commit, to_commit)` — returns changed file paths
- Handles renames via `git diff --name-status`

When the index commits to a new target, expired entries are marked and can be
reviewed by the agent on the next task.

## Session Persistence

Sessions are stored as SQLite databases in `./sessions/`. Each session DB stores:

| Table | Purpose |
|---|---|
| `TaskRecord` | User task state (plan, instruction, response, tool call logs) |
| `RunnerMessageRecord` | All agent/user messages in order |
| `RunnerToolCallRecord` | Tool execution logs (call + result) |
| `RunnerStateMetadataRecord` | Runner lifecycle metadata |

Runtime events are also logged as JSONL to `./logs/session_runs/`.

## Web API

FastAPI server with a session management API:

```bash
# From the backend/ directory:
python -m simple_agent.web.app --port 8080
```

Endpoints:
- `POST /api/sessions` — create a new session
- `GET /api/sessions` — list all sessions
- `DELETE /api/sessions/{id}` — delete a session

## Project Structure

```
backend/
├── src/simple_agent/
│   ├── task_manager/     # Orchestrator + task lifecycles, models
│   ├── session/          # Session, SessionRunner, SessionManager
│   ├── index/            # AgentIndex, tree builder, file walkers
│   ├── db/               # Database layer (SQLite via SQLModel)
│   ├── state/            # DB record classes, serialization
│   ├── process/          # AgentProcess (LLM/tool execution loop)
│   ├── tool/             # Coding tools (bash, read, write, edit, etc.)
│   ├── web/              # FastAPI web app + session API
│   ├── cli/              # CLI tools (session_inspect)
│   └── snapshot/         # Ghost indexer for commit-based index diffs
├── tests/
├── sessions/             # Session SQLite databases
├── logs/                 # Runtime JSONL logs
├── data/                 # Index databases and tool logs
└── external/             # Vendored pi-* packages
```

## Setup

```bash
cd backend
uv sync
cp .env.example .env  # add your API keys
python -m simple_agent.web.app --port 8080
```
