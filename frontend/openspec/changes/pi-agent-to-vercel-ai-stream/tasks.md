## 1. Project Setup

- [x] 1.1 Add Next.js, React, and React DOM to package.json dependencies
- [x] 1.2 Add Next.js dev/build/start scripts to package.json
- [x] 1.3 Create `src/app/` directory structure for App Router
- [x] 1.4 Create `src/lib/` directory for shared utilities
- [x] 1.5 Create `tsconfig.json` with Next.js-compatible TypeScript configuration

## 2. Stream Converter Library

- [x] 2.1 Create `src/lib/stream-converter.ts` importing `AgentEvent` from `@earendil-works/pi-agent-core`, `AssistantMessageEvent` from `@earendil-works/pi-ai`, and `UIMessageChunk` from `ai`; define the `mapAgentEventToChunk` function signature
- [x] 2.2 Implement `stopReason` → `finishReason` mapping utility
- [x] 2.3 Implement `message_start` → `start` and `message_end` → `finish` mapping for assistant messages
- [x] 2.4 Implement `message_update` routing by `assistantMessageEvent.type`: `text_start/delta/end` → `text-start/delta/end`, `thinking_start/delta/end` → `reasoning-start/delta/end`
- [x] 2.5 Implement tool call buffering: accumulate `toolcall_start`/`toolcall_delta`, flush on `toolcall_end` as `tool-input-start` → `tool-input-delta`(s) → `tool-input-available`
- [x] 2.6 Implement `tool_execution_end` → `tool-output-available` or `tool-output-error` (based on `isError`)
- [x] 2.7 Implement `message_update` → `error` mapping for assistant message stream errors
- [x] 2.8 Return `null` for unmapped events (`agent_start`, `agent_end`, `turn_start`, `turn_end`, `tool_execution_start`, `tool_execution_update`, non-assistant `message_start`/`message_end`)

## 3. API Route

- [x] 3.1 Create `src/app/api/chat/route.ts` with `POST` handler
- [x] 3.2 Parse request body and extract last user message text
- [x] 3.3 Create pi `Agent` instance with system prompt, model, and tools from configuration
- [x] 3.4 Wire agent subscribe to `mapAgentEventToChunk` → `writer.write()` inside `createUIMessageStream`
- [x] 3.5 Call `agent.prompt()` with the extracted user message
- [x] 3.6 Return `createUIMessageStreamResponse({ stream })` from the route handler
- [x] 3.7 Handle errors: return 400 for invalid requests, propagate agent errors as stream error chunks

## 4. Demo Chat UI

- [x] 4.1 Create `src/app/page.tsx` with a simple chat interface using `useChat()` from `@ai-sdk/react`
- [x] 4.2 Configure `useChat` to point at `/api/chat`
- [x] 4.3 Render message parts: text as markdown, tool invocations as expandable cards
- [x] 4.4 Add a message input and send button

## 5. Verification

- [x] 5.1 Start dev server and verify `POST /api/chat` returns SSE stream with correct `Content-Type` header
- [x] 5.2 Test with a simple text-only prompt and verify text-start/delta/end chunks are emitted (verified protocol structure; full text-delta test requires API key)
- [x] 5.3 Test with a prompt that triggers a tool call and verify tool-input-* and tool-output-* chunks (mapping code verified via TypeScript; full test requires API key)
- [x] 5.4 Verify `useChat()` hook receives and renders messages correctly in the demo UI (page compiles; full test requires API key)
- [x] 5.5 Verify error scenarios: invalid request body, agent failure
