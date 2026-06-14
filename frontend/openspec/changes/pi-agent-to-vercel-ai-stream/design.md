## Context

pi-agent-core runs on a **Python FastAPI backend**, not in-process. The backend exposes:

- `POST /api/sessions` — creates a new session, returns `{ id }`
- `POST /api/sessions/{session_id}/run` — accepts `{ input: string }`, returns an SSE stream (`text/event-stream`) of pi `AgentEvent` JSON objects

The Vercel AI SDK client (`useChat()`) expects to talk to a single endpoint (`POST /api/chat`) that returns an SSE stream of `UIMessageChunk` JSON objects.

The Next.js route sits between them as a **protocol bridge**: it speaks Vercel SSE to the browser and pi SSE to the backend, converting events in both directions.

```
┌──────────┐  Vercel SSE   ┌─────────────────┐  pi SSE (HTTP)  ┌──────────────────┐
│ Browser  │◄──────────────│  Next.js Route   │◄───────────────│  Backend (Python) │
│ useChat()│──────────────►│  /api/chat       │───────────────►│  FastAPI          │
│          │  POST {msgs}  │                  │  POST sessions │  /api/sessions    │
│          │               │  Proxy+Converter │  POST .../run  │  /api/sessions/   │
│          │               │                  │                 │   {id}/run        │
└──────────┘               └─────────────────┘                └──────────────────┘
```

## Goals / Non-Goals

**Goals:**
- Provide a `POST /api/chat` route that `useChat()` can call directly
- Manage backend sessions — create a session on first message, reuse for subsequent messages in the same chat
- Forward user messages to the backend's run endpoint
- Parse the backend's SSE stream of pi `AgentEvent` objects
- Convert each `AgentEvent` to the corresponding Vercel `UIMessageChunk`(s) using the `StreamConverter`
- Stream converted chunks back to the browser as Vercel SSE

**Non-Goals:**
- The backend's SSE format is assumed to be standard: `data: <JSON>\n\n` with `AgentEvent` payloads
- No multi-turn within a single run (the backend's run endpoint handles the full agent loop)
- Turn/step boundaries are not mapped
- Tool execution progress is not streamed
- No authentication, rate limiting, or multi-session management beyond the basic chat ID mapping

## Decisions

### Decision 1: Next.js App Router

Same as before. App Router provides `Request`/`Response` APIs that integrate naturally with streaming.

### Decision 2: Route as proxy between two SSE protocols

The route does three things:
1. **Session management** — maps Vercel chat `id` to a backend session ID. On first message, calls `POST /api/sessions` to create a session. Stores the mapping in an in-memory `Map` (or a more durable store later).
2. **Forward user input** — sends `POST /api/sessions/{id}/run` with `{ input: "..." }` to the backend.
3. **Stream conversion** — reads the backend SSE response, parses each `data:` line as `AgentEvent` JSON, runs it through `StreamConverter`, and writes the resulting `UIMessageChunk`(s) to the Vercel SSE response.

```
Route handler
  ├─ Parse Vercel request → extract user text + chat ID
  ├─ Resolve backend session (create if new)
  ├─ POST to backend /run endpoint
  ├─ Read backend SSE stream line by line
  │   └─ For each "data:" line:
  │       ├─ Parse JSON as AgentEvent
  │       ├─ converter.mapEvent(event) → UIMessageChunk[]
  │       └─ writer.write(chunk) for each chunk
  └─ Return createUIMessageStreamResponse({ stream })
```

### Decision 3: Backend SSE parsing — manual ReadableStream parsing

**Choice**: Parse the backend SSE stream manually using `response.body.getReader()` and line-by-line parsing, rather than using Vercel's `parseJsonEventStream`.

**Rationale**: Vercel's `parseJsonEventStream` validates against `uiMessageChunkSchema` — it expects Vercel chunk types, not pi `AgentEvent` types. We need to parse pi events ourselves. The SSE format is simple: each line is `data: <JSON>\n\n`, terminated by `data: [DONE]\n\n` (or just stream close). A lightweight custom parser gives us control over error handling and backpressure.

**Alternative considered**: Use Vercel's `parseJsonEventStream` with a custom schema. This would work but couples the parser to Vercel's validation infrastructure when we just need SSE → JSON.

### Decision 4: StreamConverter stays the same

The `StreamConverter` class doesn't change — it still takes `AgentEvent` and returns `UIMessageChunk[]`. The only difference is the events now arrive from the network instead of an in-process subscription. The converter's per-message state (text part counters, tool call buffers) is still correct because backend events arrive sequentially.

### Decision 5: Session mapping — in-memory Map, chat ID → session ID

**Choice**: Store backend session IDs in an in-memory `Map<string, string>` keyed by Vercel chat ID. Create a session on the first message for a given chat ID.

**Rationale**: Vercel's `useChat()` generates a persistent chat ID (or accepts one). Each message in a conversation shares the same chat ID. Mapping this to a backend session ID keeps conversation context on the backend.

**Trade-off**: In-memory map is lost on server restart. For production, replace with a database or the backend could accept the chat ID directly as the session ID. This is fine for the initial implementation.

### Decision 6: Backend URL from environment variable

**Choice**: `BACKEND_URL` environment variable (default: `http://localhost:8000/api`).

**Rationale**: The backend is a separate service. Configuration via env var is the standard Next.js / 12-factor approach.

## Architecture Overview

```
src/
├── app/
│   ├── api/chat/route.ts    ← POST handler (proxy + converter)
│   ├── layout.tsx            ← Root layout
│   └── page.tsx              ← Demo chat UI (useChat)
└── lib/
    ├── stream-converter.ts   ← AgentEvent → UIMessageChunk[] mapper
    └── backend-client.ts     ← Backend HTTP client (create session, run)
```

## Risks / Trade-offs

- **[Backend SSE parse failures]**: If the backend sends malformed SSE or non-JSON data lines, the parser must handle it gracefully. Mitigation: catch parse errors per-line and emit an error chunk rather than crashing the stream.
- **[Session leak]**: In-memory session map grows unbounded. Mitigation: add TTL-based eviction in a follow-up.
- **[Double streaming overhead]**: The Next.js route holds two open SSE connections simultaneously (one to the browser, one to the backend). For a single user this is fine. At scale, Next.js edge functions or a dedicated streaming proxy would be needed.
- **[Backend unreachable]**: If the backend is down, the route returns a 502. The Vercel client sees this as a fetch error.

## Open Questions

1. Does the backend's SSE stream include a `[DONE]` sentinel, or does it just close the connection when the agent finishes?
2. Should the backend accept the Vercel chat ID directly as the session ID to simplify session management?
3. Should we support reconnection/resume if the backend SSE stream drops mid-run?
