import type {
  AgentEvent,
  AgentToolResult,
} from "@earendil-works/pi-agent-core";
import type { AssistantMessageEvent } from "@earendil-works/pi-ai";
import type { UIMessageChunk } from "ai";

/**
 * Maps pi-agent-core stopReason values to Vercel AI SDK finishReason values.
 */
const STOP_REASON_MAP: Record<string, "stop" | "tool-calls" | "length" | "error"> = {
  stop: "stop",
  toolUse: "tool-calls",
  length: "length",
  error: "error",
  aborted: "error",
};

function mapStopReasonToFinishReason(
  stopReason: string | undefined,
): "stop" | "tool-calls" | "length" | "error" {
  if (stopReason && stopReason in STOP_REASON_MAP) {
    return STOP_REASON_MAP[stopReason];
  }
  return "stop";
}

function toolResultContentToText(
  content: AgentToolResult<unknown>["content"],
): string {
  return content
    .map((part) => (part.type === "text" ? part.text : "[image]"))
    .join("\n");
}

/**
 * Buffered tool call state keyed by contentIndex.
 */
interface BufferedToolCall {
  toolCallId: string;
  toolName: string;
  deltas: string[];
}

/**
 * Converts pi-agent-core AgentEvents to Vercel AI SDK UIMessageChunks.
 *
 * Maintains internal state for tool call buffering — create a new
 * instance per request to avoid cross-request state leakage.
 */
export class StreamConverter {
  /** Counter for generating unique text part IDs per message */
  private textPartCounter = 0;
  /** Counter for generating unique reasoning part IDs per message */
  private reasoningPartCounter = 0;
  /** Buffered tool calls keyed by contentIndex. Flushed on toolcall_end. */
  private toolCallBuffers = new Map<number, BufferedToolCall>();

  /**
   * Reset per-message counters. Call at the start of each new assistant message.
   */
  private resetMessageState(): void {
    this.textPartCounter = 0;
    this.reasoningPartCounter = 0;
    this.toolCallBuffers.clear();
  }

  /**
   * Map a single pi AgentEvent to zero or more Vercel AI SDK UIMessageChunks.
   * Returns an empty array for events with no Vercel equivalent.
   */
  mapEvent(event: AgentEvent): UIMessageChunk[] {
    switch (event.type) {
      // ── Message boundaries ────────────────────────────────────
      case "message_start": {
        // Only assistant messages produce chunks
        if (event.message.role !== "assistant") return [];
        this.resetMessageState();
        return [{ type: "start" as const }];
      }

      case "message_end": {
        if (event.message.role !== "assistant") return [];
        const stopReason = (
          event.message as { stopReason?: string }
        ).stopReason;
        return [
          {
            type: "finish" as const,
            finishReason: mapStopReasonToFinishReason(stopReason),
          },
        ];
      }

      // ── Content parts (via assistantMessageEvent) ─────────────
      case "message_update": {
        if (event.message.role !== "assistant") return [];
        return this.mapAssistantMessageEvent(event.assistantMessageEvent);
      }

      // ── Tool execution ────────────────────────────────────────
      case "tool_execution_start":
      case "tool_execution_update": {
        // No Vercel equivalent — ignored
        return [];
      }

      case "tool_execution_end": {
        const result = event.result as AgentToolResult<unknown>;
        if (event.isError) {
          return [
            {
              type: "tool-output-error" as const,
              toolCallId: event.toolCallId,
              errorText: toolResultContentToText(result.content),
            },
          ];
        }
        return [
          {
            type: "tool-output-available" as const,
            toolCallId: event.toolCallId,
            output: toolResultContentToText(result.content),
          },
        ];
      }

      // ── Non-mapped lifecycle events ───────────────────────────
      case "agent_start":
      case "agent_end":
      case "turn_start":
      case "turn_end":
        return [];
    }
  }

  /**
   * Map AssistantMessageEvent sub-types to chunks.
   */
  private mapAssistantMessageEvent(
    event: AssistantMessageEvent,
  ): UIMessageChunk[] {
    switch (event.type) {
      // ── Text streaming ────────────────────────────────────────
      case "text_start":
        return [
          {
            type: "text-start" as const,
            id: `txt-${this.textPartCounter++}`,
          },
        ];

      case "text_delta":
        return [
          {
            type: "text-delta" as const,
            id: `txt-${this.textPartCounter - 1}`,
            delta: event.delta,
          },
        ];

      case "text_end":
        return [
          {
            type: "text-end" as const,
            id: `txt-${this.textPartCounter - 1}`,
          },
        ];

      // ── Reasoning / thinking streaming ────────────────────────
      case "thinking_start":
        return [
          {
            type: "reasoning-start" as const,
            id: `rsn-${this.reasoningPartCounter++}`,
          },
        ];

      case "thinking_delta":
        return [
          {
            type: "reasoning-delta" as const,
            id: `rsn-${this.reasoningPartCounter - 1}`,
            delta: event.delta,
          },
        ];

      case "thinking_end":
        return [
          {
            type: "reasoning-end" as const,
            id: `rsn-${this.reasoningPartCounter - 1}`,
          },
        ];

      // ── Tool call streaming (buffered) ─────────────────────────
      case "toolcall_start": {
        // Initialize buffer for this contentIndex
        this.toolCallBuffers.set(event.contentIndex, {
          toolCallId: "", // filled at toolcall_end
          toolName: "", // filled at toolcall_end
          deltas: [],
        });
        return [];
      }

      case "toolcall_delta": {
        const buffer = this.toolCallBuffers.get(event.contentIndex);
        if (buffer) {
          buffer.deltas.push(event.delta);
        }
        return [];
      }

      case "toolcall_end": {
        const buffer = this.toolCallBuffers.get(event.contentIndex);
        if (!buffer) return [];

        const chunks: UIMessageChunk[] = [];

        // tool-input-start
        chunks.push({
          type: "tool-input-start" as const,
          toolCallId: event.toolCall.id,
          toolName: event.toolCall.name,
        });

        // buffered tool-input-delta(s)
        for (const delta of buffer.deltas) {
          chunks.push({
            type: "tool-input-delta" as const,
            toolCallId: event.toolCall.id,
            inputTextDelta: delta,
          });
        }

        // tool-input-available
        chunks.push({
          type: "tool-input-available" as const,
          toolCallId: event.toolCall.id,
          toolName: event.toolCall.name,
          input: event.toolCall.arguments,
        });

        this.toolCallBuffers.delete(event.contentIndex);
        return chunks;
      }

      // ── Stream lifecycle ──────────────────────────────────────
      case "start":
        return []; // Nothing — we use message_start for the message boundary

      case "done":
        return []; // Nothing — finishReason comes from message_end

      case "error": {
        return [
          {
            type: "error" as const,
            errorText: event.error.errorMessage ?? "Unknown error",
          },
        ];
      }
    }
  }
}
