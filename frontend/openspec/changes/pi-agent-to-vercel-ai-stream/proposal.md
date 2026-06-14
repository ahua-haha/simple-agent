## Why

pi-agent-core and Vercel AI SDK use incompatible streaming protocols. pi-agent-core emits coarse-grained agent lifecycle events (`message_start`, `message_update` with sub-events, `message_end`, `tool_execution_start/end`), while Vercel AI SDK expects a flat SSE stream of fine-grained UI message chunks (`start`, `text-delta`, `tool-input-delta`, `tool-output-available`, `finish`). Without a conversion layer, the `useChat()` hook cannot consume pi-agent-core agent output, blocking the ability to build streaming chat UIs on top of pi-agent-core backends.

## What Changes

- Add a Next.js API route (`POST /api/chat`) that acts as a protocol bridge between pi-agent-core agent events and Vercel AI SDK data stream format
- Implement an event-to-chunk mapper that converts pi `AgentEvent` and `AssistantMessageEvent` types to Vercel `UIMessageChunk` types
- Route receives `{ messages: UIMessage[] }` from `useChat()`, creates a pi `Agent`, subscribes to events, and returns an SSE stream via `createUIMessageStreamResponse()`
- Add Next.js scaffolding (App Router) to the project
- Each assistant message gets its own `start`/`finish` chunk pair; text, reasoning, and tool call content maps to the corresponding part chunks

## Capabilities

### New Capabilities

- `stream-conversion`: Protocol bridge that maps pi-agent-core `AgentEvent` stream to Vercel AI SDK `UIMessageChunk` stream, enabling the `useChat()` hook to consume pi-agent-core agent output as standard `UIMessage` objects

### Modified Capabilities

<!-- No existing capabilities to modify -->

## Impact

- **Dependencies**: Add `next`, `react`, `react-dom` (Next.js App Router scaffolding)
- **New files**: `src/app/api/chat/route.ts` (API route), `src/app/page.tsx` (demo chat UI), `src/lib/stream-converter.ts` (event-to-chunk mapper)
- **Existing files**: `package.json` (add Next.js dependencies and scripts)
- **No breaking changes**: project is pre-implementation, no existing code to break
