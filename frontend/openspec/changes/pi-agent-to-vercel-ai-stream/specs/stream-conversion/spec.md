## ADDED Requirements

### Requirement: API route accepts chat requests

The system SHALL provide a `POST /api/chat` route that accepts JSON request bodies matching the Vercel AI SDK `useChat()` format containing `{ messages: UIMessage[] }`.

#### Scenario: Valid chat request
- **WHEN** `useChat()` sends a POST request with `{ messages: [{ id: "1", role: "user", parts: [{ type: "text", text: "Hello" }] }] }`
- **THEN** the route SHALL parse the messages and extract the last user message text as input to the pi agent

#### Scenario: Empty messages
- **WHEN** the request body contains an empty messages array
- **THEN** the route SHALL return a 400 error response

### Requirement: Assistant message boundaries map to start/finish chunks

The system SHALL emit a `start` chunk when pi emits `message_start` for an assistant message, and a `finish` chunk when pi emits `message_end` for an assistant message.

#### Scenario: Single assistant message
- **WHEN** pi emits `message_start` with an assistant message
- **THEN** the stream SHALL emit `{ "type": "start" }`
- **WHEN** pi subsequently emits `message_end` for that assistant message
- **THEN** the stream SHALL emit `{ "type": "finish", "finishReason": "..." }`

#### Scenario: Non-assistant messages produce no chunks
- **WHEN** pi emits `message_start` with a user or toolResult message
- **THEN** the stream SHALL NOT emit any chunk

### Requirement: Text content maps to text chunks

The system SHALL convert pi `message_update` events with `assistantMessageEvent.type` of `text_start`, `text_delta`, and `text_end` to the corresponding Vercel `text-start`, `text-delta`, and `text-end` chunks.

#### Scenario: Streaming text
- **WHEN** pi emits `message_update` with `assistantMessageEvent.type === "text_start"` and `contentIndex: 0`
- **THEN** the stream SHALL emit `{ "type": "text-start", "id": "txt-0" }`
- **WHEN** pi emits `message_update` with `assistantMessageEvent.type === "text_delta"`, `contentIndex: 0`, and `delta: "Hello"`
- **THEN** the stream SHALL emit `{ "type": "text-delta", "id": "txt-0", "delta": "Hello" }`
- **WHEN** pi emits `message_update` with `assistantMessageEvent.type === "text_end"` and `contentIndex: 0`
- **THEN** the stream SHALL emit `{ "type": "text-end", "id": "txt-0" }`

### Requirement: Reasoning content maps to reasoning chunks

The system SHALL convert pi `message_update` events with `assistantMessageEvent.type` of `thinking_start`, `thinking_delta`, and `thinking_end` to the corresponding Vercel `reasoning-start`, `reasoning-delta`, and `reasoning-end` chunks.

#### Scenario: Streaming reasoning
- **WHEN** pi emits `message_update` with `assistantMessageEvent.type === "thinking_start"` and `contentIndex: 1`
- **THEN** the stream SHALL emit `{ "type": "reasoning-start", "id": "rsn-1" }`
- **WHEN** pi emits `message_update` with `assistantMessageEvent.type === "thinking_delta"`, `contentIndex: 1`, and `delta: "Let me think..."`
- **THEN** the stream SHALL emit `{ "type": "reasoning-delta", "id": "rsn-1", "delta": "Let me think..." }`

### Requirement: Tool call content maps to tool input chunks

The system SHALL convert pi `message_update` events with `assistantMessageEvent.type` of `toolcall_start`, `toolcall_delta`, and `toolcall_end` to the corresponding Vercel `tool-input-start`, `tool-input-delta`, and `tool-input-available` chunks. The system SHALL buffer toolcall deltas until `toolcall_end` provides the `toolCall.id`.

#### Scenario: Tool call with buffered streaming
- **WHEN** pi emits `message_update` with `assistantMessageEvent.type === "toolcall_start"` and `contentIndex: 2`
- **THEN** the system SHALL begin buffering tool call input for contentIndex 2 without emitting a chunk
- **WHEN** pi emits `message_update` with `assistantMessageEvent.type === "toolcall_delta"`, `contentIndex: 2`, and `delta: "{\"city\""`
- **THEN** the system SHALL append the delta to the buffer without emitting a chunk
- **WHEN** pi emits `message_update` with `assistantMessageEvent.type === "toolcall_end"`, `contentIndex: 2`, and `toolCall: { id: "call_1", name: "getWeather", arguments: { city: "Beijing" } }`
- **THEN** the stream SHALL emit in sequence:
  1. `{ "type": "tool-input-start", "toolCallId": "call_1", "toolName": "getWeather" }`
  2. `{ "type": "tool-input-delta", "toolCallId": "call_1", "inputTextDelta": "{\"city\"" }`
  3. `{ "type": "tool-input-available", "toolCallId": "call_1", "toolName": "getWeather", "input": { "city": "Beijing" } }`

### Requirement: Tool execution result maps to tool output chunks

The system SHALL convert pi `tool_execution_end` to Vercel `tool-output-available` (on success) or `tool-output-error` (when `isError` is true).

#### Scenario: Successful tool execution
- **WHEN** pi emits `tool_execution_end` with `toolCallId: "call_1"`, `result: { temp: 25 }`, and `isError: false`
- **THEN** the stream SHALL emit `{ "type": "tool-output-available", "toolCallId": "call_1", "output": { "temp": 25 } }`

#### Scenario: Failed tool execution
- **WHEN** pi emits `tool_execution_end` with `toolCallId: "call_1"`, `result: "Permission denied"`, and `isError: true`
- **THEN** the stream SHALL emit `{ "type": "tool-output-error", "toolCallId": "call_1", "errorText": "Permission denied" }`

### Requirement: Stop reason maps to finish reason

The system SHALL map pi `AssistantMessage.stopReason` values to Vercel `finishReason` values according to the defined mapping.

#### Scenario: Stop reason mapping
- **WHEN** pi assistant message has `stopReason: "stop"`
- **THEN** the `finish` chunk SHALL contain `finishReason: "stop"`
- **WHEN** pi assistant message has `stopReason: "toolUse"`
- **THEN** the `finish` chunk SHALL contain `finishReason: "tool-calls"`
- **WHEN** pi assistant message has `stopReason: "length"`
- **THEN** the `finish` chunk SHALL contain `finishReason: "length"`
- **WHEN** pi assistant message has `stopReason: "error"` or `"aborted"`
- **THEN** the `finish` chunk SHALL contain `finishReason: "error"`

### Requirement: Stream errors propagate to client

The system SHALL emit a Vercel `error` chunk when pi emits an `AssistantMessageEvent` of type `error`.

#### Scenario: LLM stream error
- **WHEN** pi emits `message_update` with `assistantMessageEvent.type === "error"` and `error.errorMessage: "Rate limit exceeded"`
- **THEN** the stream SHALL emit `{ "type": "error", "errorText": "Rate limit exceeded" }`

### Requirement: Non-mapped events are silently ignored

The system SHALL NOT emit any stream chunk for pi events that have no Vercel equivalent: `agent_start`, `agent_end`, `turn_start`, `turn_end`, `tool_execution_start`, `tool_execution_update`.

#### Scenario: Ignored events
- **WHEN** pi emits `agent_start`, `agent_end`, `turn_start`, `turn_end`, `tool_execution_start`, or `tool_execution_update`
- **THEN** the mapper function SHALL return `null` and no chunk SHALL be enqueued in the stream

### Requirement: Stream response uses correct SSE format

The system SHALL return the stream using `createUIMessageStreamResponse()` which SHALL produce Server-Sent Events with `data: <JSON>\n\n` framing and a `data: [DONE]\n\n` terminator.

#### Scenario: SSE format
- **WHEN** the route handler returns the response
- **THEN** the response SHALL have `Content-Type: text/event-stream`
- **AND** each chunk SHALL be serialized as `data: <JSON>\n\n`
- **AND** the stream SHALL end with `data: [DONE]\n\n`
