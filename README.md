# Simple Agent

CLI application for building and running agents with pi-agent runtime.

## Web API

Start the server:

```bash
simple-agent-web --model-provider deepseek --model-name deepseek-v4-pro
```

Or without the chat API (task-tree debug UI only):

```bash
simple-agent-web
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Agent chat (streaming, Vercel AI SDK v6 protocol) — only when `--model-provider` is set |
| `GET` | `/` | Task tree debug UI |
| `GET` | `/task/{id}` | Task detail debug UI |

### Chat API: Vercel AI SDK Data Stream Protocol (v6)

The `POST /api/chat` endpoint follows the [Vercel AI SDK data stream protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol#data-stream-protocol). The response carries the header `x-vercel-ai-data-stream: v1`.

**Wire format:** SSE-like — each frame is a `data: {json}\n` line, terminated by `data: [DONE]\n`.

#### Data frame reference

Each frame is serialized as `data: <json>\n`. All frames emitted by this server are marked ✓.

| # | type | Category | JSON format | Emitted |
|---|------|----------|-------------|---------|
| 1 | `start` | Lifecycle | `{"type":"start","messageId":"msg_abc"}` | ✓ |
| 2 | `text-start` | Text | `{"type":"text-start","id":"txt_1"}` | ✓ |
| 3 | `text-delta` | Text | `{"type":"text-delta","id":"txt_1","delta":"Hello"}` | ✓ |
| 4 | `text-end` | Text | `{"type":"text-end","id":"txt_1"}` | ✓ |
| 5 | `reasoning-start` | Reasoning | `{"type":"reasoning-start","id":"msg_abc"}` | |
| 6 | `reasoning-delta` | Reasoning | `{"type":"reasoning-delta","id":"msg_abc","delta":"Let me think..."}` | |
| 7 | `reasoning-end` | Reasoning | `{"type":"reasoning-end","id":"msg_abc"}` | |
| 8 | `tool-input-start` | Tool | `{"type":"tool-input-start","toolCallId":"tc1","toolName":"read_file"}` | ✓ |
| 9 | `tool-input-delta` | Tool | `{"type":"tool-input-delta","toolCallId":"tc1","inputTextDelta":"{\\"path\\":"}` | |
| 10 | `tool-input-available` | Tool | `{"type":"tool-input-available","toolCallId":"tc1","toolName":"read_file","input":{"path":"/foo"}}` | ✓ |
| 11 | `tool-input-error` | Tool | `{"type":"tool-input-error","toolCallId":"tc1","toolName":"search","input":{},"errorText":"Timeout"}` | |
| 12 | `tool-output-available` | Tool | `{"type":"tool-output-available","toolCallId":"tc1","output":"result"}` | ✓ |
| 13 | `tool-output-error` | Tool | `{"type":"tool-output-error","toolCallId":"tc1","errorText":"Execution failed"}` | |
| 14 | `tool-output-denied` | Tool | `{"type":"tool-output-denied","toolCallId":"tc1"}` | |
| 15 | `tool-approval-request` | Tool | `{"type":"tool-approval-request","approvalId":"apr1","toolCallId":"tc1"}` | |
| 16 | `source-url` | Source | `{"type":"source-url","sourceId":"s1","url":"https://example.com","title":"Example"}` | |
| 17 | `source-document` | Source | `{"type":"source-document","sourceId":"s2","mediaType":"text/plain","title":"doc.txt"}` | |
| 18 | `file` | File | `{"type":"file","url":"data:image/png;base64,...","mediaType":"image/png"}` | |
| 19 | `start-step` | Step | `{"type":"start-step"}` | |
| 20 | `finish-step` | Step | `{"type":"finish-step"}` | |
| 21 | `finish` | Lifecycle | `{"type":"finish","finishReason":"stop"}` | ✓ |
| 22 | `abort` | Lifecycle | `{"type":"abort","reason":"cancelled"}` | |
| 23 | `error` | Lifecycle | `{"type":"error","errorText":"Something went wrong"}` | |
| 24 | `data-{name}` | Custom | `{"type":"data-custom","id":"d1","data":{"key":"value"}}` | |

#### pi-agent event → v6 frame mapping

| pi-agent event | v6 frame(s) emitted |
|---|---|
| `start()` (manual) | `start` |
| `MessageUpdateEvent` (text_delta) | `text-start` (first delta only) + `text-delta` |
| `ToolExecutionStartEvent` | `text-end` (if text open) + `tool-input-start` + `tool-input-available` |
| `ToolExecutionEndEvent` | `tool-output-available` |
| `AgentEndEvent` | `text-end` (if text open) + `finish` + `None` sentinel |
| stream terminator | `data: [DONE]\n` |

#### Frontend integration

```tsx
// simple-agent-frontend
import { useChat } from '@ai-sdk/react';

const { messages, sendMessage } = useChat({
  api: '/api/chat',
});
```

The frontend `useChat()` with v6 SDK parses these frames automatically and produces `message.parts` with text, tool, and reasoning parts.

#### Example stream

```
data: {"type":"start","messageId":"msg_abc123def45"}

data: {"type":"text-start","id":"txt_1"}

data: {"type":"text-delta","id":"txt_1","delta":"Let me"}

data: {"type":"text-delta","id":"txt_1","delta":" check the weather."}

data: {"type":"text-end","id":"txt_1"}

data: {"type":"tool-input-start","toolCallId":"tc1","toolName":"get_weather"}

data: {"type":"tool-input-available","toolCallId":"tc1","toolName":"get_weather","input":{"city":"San Francisco"}}

data: {"type":"tool-output-available","toolCallId":"tc1","output":{"temp":18,"unit":"celsius"}}

data: {"type":"text-start","id":"txt_2"}

data: {"type":"text-delta","id":"txt_2","delta":"It's 18°C in San Francisco."}

data: {"type":"text-end","id":"txt_2"}

data: {"type":"finish","finishReason":"stop"}

data: [DONE]
```

### Request format

```json
POST /api/chat
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "What's the weather in SF?"}
  ]
}
```

### v5 Protocol (legacy reference)

The older v5 protocol used a compact `{type_code}:{json}\n` format instead of SSE-like `data:` framing:

| Code | Meaning | Format |
|------|---------|--------|
| `0` | Text delta | `0:"text"\n` |
| `9` | Tool call | `9:{"toolCallId":"...","toolName":"...","args":{...}}\n` |
| `a` | Tool result | `a:{"toolCallId":"...","toolName":"...","result":{...}}\n` |
| `d` | Finish | `d:{"finishReason":"stop","usage":{...}}\n` |

See `/root/workspace/next-fastapi/api/index.py` for a reference implementation.
