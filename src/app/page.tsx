"use client";

import { useChat } from "@ai-sdk/react";
import type { UIMessage } from "ai";
import { useState } from "react";

export default function ChatPage() {
  const { messages, sendMessage, status, stop, error } = useChat();
  const [input, setInput] = useState("");
  const isLoading = status === "submitted" || status === "streaming";

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim()) return;
    sendMessage({ text: input.trim() });
    setInput("");
  }

  return (
    <div style={styles.container}>
      <div style={styles.chatPanel}>
        <h1 style={styles.title}>Agent Chat</h1>
        <div style={styles.messages}>
          {messages.map((message) => (
            <MessageCard key={message.id} message={message} />
          ))}
        </div>
        {error && (
          <div style={styles.error}>
            Error: {error.message}
          </div>
        )}
        <form onSubmit={handleSubmit} style={styles.inputForm}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message..."
            disabled={isLoading}
            style={styles.input}
          />
          {isLoading ? (
            <button type="button" onClick={stop} style={styles.stopButton}>
              Stop
            </button>
          ) : (
            <button type="submit" disabled={!input.trim()} style={styles.sendButton}>
              Send
            </button>
          )}
        </form>
      </div>
    </div>
  );
}

function MessageCard({ message }: { message: UIMessage }) {
  return (
    <div
      style={{
        ...styles.message,
        ...(message.role === "user" ? styles.userMessage : styles.assistantMessage),
      }}
    >
      <div style={styles.roleLabel}>
        {message.role === "user" ? "You" : "Assistant"}
      </div>
      <div style={styles.parts}>
        {message.parts.map((part, i) => (
          <MessagePart key={i} part={part} />
        ))}
      </div>
    </div>
  );
}

function MessagePart({ part }: { part: UIMessage["parts"][number] }) {
  switch (part.type) {
    case "text": {
      const textPart = part as { type: "text"; text: string };
      return <div style={styles.text}>{textPart.text}</div>;
    }

    case "reasoning": {
      const reasoningPart = part as { type: "reasoning"; text: string };
      return (
        <details style={styles.reasoning}>
          <summary style={styles.reasoningSummary}>Reasoning</summary>
          <div style={styles.reasoningContent}>{reasoningPart.text}</div>
        </details>
      );
    }

    case "step-start":
      return <hr style={styles.stepDivider} />;

    default: {
      // Tool invocation parts: type starts with "tool-"
      const toolPart = part as {
        type: string;
        toolCallId?: string;
        toolName?: string;
        state?: string;
        input?: unknown;
        output?: unknown;
        errorText?: string;
      };

      if (part.type.startsWith("tool-")) {
        return (
          <details style={styles.toolCard}>
            <summary style={styles.toolSummary}>
              🔧 {toolPart.toolName ?? part.type.replace("tool-", "")}
              {" · "}
              <span style={styles.toolState}>{toolPart.state ?? "pending"}</span>
            </summary>
            <div style={styles.toolDetails}>
              {toolPart.input != null && (
                <div>
                  <strong>Input:</strong>
                  <pre style={styles.toolJson}>
                    {JSON.stringify(toolPart.input, null, 2)}
                  </pre>
                </div>
              )}
              {toolPart.output != null && (
                <div>
                  <strong>Output:</strong>
                  <pre style={styles.toolJson}>
                    {JSON.stringify(toolPart.output, null, 2)}
                  </pre>
                </div>
              )}
              {toolPart.errorText && (
                <div style={{ color: "#e53e3e" }}>
                  <strong>Error:</strong> {toolPart.errorText}
                </div>
              )}
            </div>
          </details>
        );
      }

      return null;
    }
  }
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    justifyContent: "center",
    height: "100vh",
    backgroundColor: "#f7fafc",
  },
  chatPanel: {
    display: "flex",
    flexDirection: "column",
    width: "100%",
    maxWidth: 720,
    height: "100vh",
    padding: "16px",
  },
  title: {
    fontSize: 20,
    fontWeight: 600,
    margin: "0 0 16px 0",
    padding: "8px 0",
    borderBottom: "1px solid #e2e8f0",
  },
  messages: {
    flex: 1,
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: 12,
    paddingBottom: 12,
  },
  message: {
    padding: "12px 16px",
    borderRadius: 8,
    maxWidth: "85%",
  },
  userMessage: {
    alignSelf: "flex-end",
    backgroundColor: "#ebf4ff",
  },
  assistantMessage: {
    alignSelf: "flex-start",
    backgroundColor: "#ffffff",
    border: "1px solid #e2e8f0",
  },
  roleLabel: {
    fontSize: 11,
    fontWeight: 600,
    textTransform: "uppercase",
    color: "#718096",
    marginBottom: 4,
  },
  parts: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
  },
  text: {
    fontSize: 14,
    lineHeight: 1.6,
    whiteSpace: "pre-wrap" as const,
  },
  reasoning: {
    fontSize: 13,
    backgroundColor: "#fefcbf",
    borderRadius: 4,
    padding: "4px 8px",
  },
  reasoningSummary: {
    cursor: "pointer",
    color: "#744210",
    fontWeight: 500,
  },
  reasoningContent: {
    fontSize: 12,
    color: "#744210",
    padding: "4px 0",
  },
  stepDivider: {
    border: "none",
    borderTop: "1px dashed #cbd5e0",
    margin: "4px 0",
  },
  toolCard: {
    fontSize: 13,
    backgroundColor: "#f0fff4",
    borderRadius: 4,
    padding: "4px 8px",
    border: "1px solid #c6f6d5",
  },
  toolSummary: {
    cursor: "pointer",
    color: "#22543d",
    fontWeight: 500,
  },
  toolState: {
    fontSize: 11,
    color: "#718096",
    fontStyle: "italic",
  },
  toolDetails: {
    padding: "4px 0",
    fontSize: 12,
  },
  toolJson: {
    backgroundColor: "#edf2f7",
    padding: 8,
    borderRadius: 4,
    overflow: "auto",
    fontSize: 11,
    margin: "4px 0",
  },
  inputForm: {
    display: "flex",
    gap: 8,
    padding: "12px 0 0 0",
    borderTop: "1px solid #e2e8f0",
  },
  input: {
    flex: 1,
    padding: "10px 14px",
    fontSize: 14,
    borderRadius: 8,
    border: "1px solid #cbd5e0",
    outline: "none",
  },
  sendButton: {
    padding: "10px 20px",
    fontSize: 14,
    fontWeight: 600,
    borderRadius: 8,
    border: "none",
    backgroundColor: "#3182ce",
    color: "white",
    cursor: "pointer",
  },
  stopButton: {
    padding: "10px 20px",
    fontSize: 14,
    fontWeight: 600,
    borderRadius: 8,
    border: "none",
    backgroundColor: "#e53e3e",
    color: "white",
    cursor: "pointer",
  },
  error: {
    padding: "8px 12px",
    backgroundColor: "#fff5f5",
    color: "#c53030",
    borderRadius: 6,
    fontSize: 13,
    marginBottom: 8,
  },
};
