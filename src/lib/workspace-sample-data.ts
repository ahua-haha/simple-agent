import type { WorkspaceFolder } from "./workspace-types";

export type {
  WorkspaceLanguage,
  WorkspaceViewMode,
  WorkspaceFile,
  WorkspaceFolder,
  WorkspaceNode,
} from "./workspace-types";

export const workspaceTree: WorkspaceFolder = {
  type: "folder",
  id: "root",
  name: "agent-frontend",
  path: "",
  children: [
    {
      type: "folder",
      id: "src",
      name: "src",
      path: "src",
      children: [
        {
          type: "folder",
          id: "src-app",
          name: "app",
          path: "src/app",
          children: [
            {
              type: "file",
              id: "app-page",
              name: "page.tsx",
              path: "src/app/page.tsx",
              language: "typescript",
              content: `"use client";

import { Thread } from "@/components/assistant-ui/thread";
import { AssistantRuntimeProvider } from "@assistant-ui/react";

export default function ChatPage() {
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <main className="h-dvh">
        <Thread />
      </main>
    </AssistantRuntimeProvider>
  );
}
`,
            },
          ],
        },
        {
          type: "folder",
          id: "src-lib",
          name: "lib",
          path: "src/lib",
          children: [
            {
              type: "file",
              id: "lib-utils",
              name: "utils.ts",
              path: "src/lib/utils.ts",
              language: "typescript",
              content: `import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
`,
            },
          ],
        },
      ],
    },
    {
      type: "folder",
      id: "docs",
      name: "docs",
      path: "docs",
      children: [
        {
          type: "file",
          id: "readme",
          name: "README.md",
          path: "README.md",
          language: "markdown",
          content: `# Agent Frontend

This mock workspace editor demonstrates file browsing and view modes.

## Supported Views

- Raw editing for every file
- Markdown preview for markdown files
- Format-specific views can be added later
`,
        },
        {
          type: "file",
          id: "notes",
          name: "notes.txt",
          path: "docs/notes.txt",
          language: "text",
          content: `Mock workspace notes

This file is plain text.
Only the raw editor is available.
`,
        },
      ],
    },
    {
      type: "file",
      id: "package-json",
      name: "package.json",
      path: "package.json",
      language: "json",
      content: `{
  "name": "agent-frontend",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build"
  }
}
`,
    },
  ],
};
// Helper functions are now in workspace-types.ts
export {
  flattenWorkspaceFiles,
  getFirstWorkspaceFile,
  getWorkspaceFileById,
  getWorkspaceViewModes,
} from "./workspace-types";
