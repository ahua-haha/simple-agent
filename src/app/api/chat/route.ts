import {
  createUIMessageStream,
  createUIMessageStreamResponse,
} from "ai";
import { StreamConverter } from "@/lib/stream-converter";
import { resolveSession, runSession } from "@/lib/backend-client";
import type { AgentEvent } from "@earendil-works/pi-agent-core";

export async function POST(request: Request): Promise<Response> {
  // Parse and validate the request body
  let body: { id?: string; messages?: unknown };
  try {
    body = await request.json();
  } catch {
    return new Response("Invalid JSON in request body", { status: 400 });
  }

  if (!body.messages || !Array.isArray(body.messages)) {
    return new Response('Request body must contain { "messages": [...] }', {
      status: 400,
    });
  }

  const messages = body.messages as Array<{ role?: string; parts?: unknown[] }>;

  if (messages.length === 0) {
    return new Response("Messages array must not be empty", { status: 400 });
  }

  // Extract text from the last user message
  const lastMessage = messages[messages.length - 1];
  const userText = extractUserText(lastMessage);
  if (!userText) {
    return new Response(
      "Last message must be a user message with text content",
      { status: 400 },
    );
  }

  // Map Vercel chat ID to a backend session
  const chatId = body.id ?? "default";
  let sessionId: string;
  try {
    sessionId = await resolveSession(chatId);
  } catch (error) {
    return new Response(
      `Failed to create backend session: ${(error as Error).message}`,
      { status: 502 },
    );
  }

  const converter = new StreamConverter();

  const stream = createUIMessageStream({
    execute: async ({ writer }) => {
      // Forward user input to the backend
      const backendResponse = await runSession(sessionId, userText);

      if (!backendResponse.body) {
        throw new Error("Backend returned an empty response body");
      }

      // Parse the backend SSE stream
      const reader = backendResponse.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Process complete SSE messages (separated by \n\n)
          const parts = buffer.split("\n\n");
          // The last part may be incomplete — keep it in the buffer
          buffer = parts.pop() ?? "";

          for (const part of parts) {
            // Extract data: line from the SSE block (may also contain event: lines)
            const dataMatch = part.match(/^data:\s*(.+)$/m);
            if (!dataMatch) continue;

            const payload = dataMatch[1].trim();

            // Check for stream termination
            if (payload === "[DONE]") {
              return;
            }

            // Parse as snake_case JSON and convert to camelCase AgentEvent
            let event: AgentEvent;
            try {
              const raw = JSON.parse(payload);
              event = snakeToCamel(raw) as AgentEvent;
            } catch {
              // Skip non-JSON or malformed data lines
              continue;
            }

            // Convert and write chunks
            try {
              const chunks = converter.mapEvent(event);
              for (const chunk of chunks) {
                writer.write(chunk);
              }
            } catch {
              // Suppress individual conversion errors to keep the stream alive
            }
          }
        }

        // Process any remaining buffer content
        if (buffer.trim()) {
          const dataMatch = buffer.match(/^data:\s*(.+)$/m);
          if (dataMatch && dataMatch[1].trim() !== "[DONE]") {
            try {
              const raw = JSON.parse(dataMatch[1].trim());
              const event = snakeToCamel(raw) as AgentEvent;
              const chunks = converter.mapEvent(event);
              for (const chunk of chunks) {
                writer.write(chunk);
              }
            } catch {
              // Ignore
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    },
  });

  return createUIMessageStreamResponse({
    stream,
  });
}

/**
 * Recursively convert snake_case keys to camelCase.
 * Handles nested objects and arrays.
 */
function snakeToCamel(obj: unknown): unknown {
  if (Array.isArray(obj)) {
    return obj.map(snakeToCamel);
  }
  if (obj !== null && typeof obj === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(
      obj as Record<string, unknown>,
    )) {
      const camelKey = key.replace(/_([a-z])/g, (_, char: string) =>
        char.toUpperCase(),
      );
      result[camelKey] = snakeToCamel(value);
    }
    return result;
  }
  return obj;
}

/**
 * Extract text content from a user message in Vercel AI SDK UIMessage format.
 */
function extractUserText(message: {
  role?: string;
  parts?: unknown[];
}): string | null {
  if (message.role !== "user") return null;

  const parts = message.parts ?? [];
  if (!Array.isArray(parts)) return null;

  for (const part of parts) {
    if (
      typeof part === "object" &&
      part !== null &&
      "type" in part &&
      (part as Record<string, unknown>).type === "text" &&
      "text" in part
    ) {
      return String((part as Record<string, unknown>).text);
    }
  }

  return null;
}
