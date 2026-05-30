const BACKEND_URL =
  process.env.BACKEND_URL ?? "http://localhost:8080/api";

/**
 * Lightweight HTTP client for the pi-agent-core Python backend.
 *
 * Endpoints:
 * - POST /sessions          → { id: string, created_at, updated_at }
 * - POST /sessions/{id}/run → SSE stream of AgentEvent JSON
 */

interface Session {
  id: string;
}

/**
 * Create a new agent session on the backend.
 */
export async function createSession(): Promise<Session> {
  const response = await fetch(`${BACKEND_URL}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    throw new Error(
      `Failed to create session: ${response.status} ${await response.text()}`,
    );
  }

  return response.json();
}

/**
 * Run the agent on the backend with the given input.
 * Returns the raw Response object so the caller can stream the SSE body.
 */
export async function runSession(
  sessionId: string,
  input: string,
  signal?: AbortSignal,
): Promise<Response> {
  const response = await fetch(
    `${BACKEND_URL}/sessions/${encodeURIComponent(sessionId)}/run`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input }),
      signal,
    },
  );

  if (!response.ok) {
    const text = await response.text();
    throw new Error(
      `Failed to run session ${sessionId}: ${response.status} ${text}`,
    );
  }

  if (!response.body) {
    throw new Error("Backend returned an empty response body");
  }

  return response;
}

/**
 * In-memory map of Vercel chat ID → backend session ID.
 * Lost on server restart. Replace with a database for production.
 */
const sessionMap = new Map<string, string>();

/**
 * Resolve a backend session ID for a given Vercel chat ID.
 * Creates a new backend session if one doesn't exist.
 */
export async function resolveSession(chatId: string): Promise<string> {
  const existing = sessionMap.get(chatId);
  if (existing) return existing;

  const session = await createSession();
  sessionMap.set(chatId, session.id);
  return session.id;
}
