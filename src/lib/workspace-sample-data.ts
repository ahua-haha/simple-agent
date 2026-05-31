export type WorkspaceLanguage = "typescript" | "markdown" | "json" | "text";

export type WorkspaceViewMode = "raw" | "preview";

export type WorkspaceFile = {
  type: "file";
  id: string;
  name: string;
  path: string;
  language: WorkspaceLanguage;
  content: string;
};

export type WorkspaceFolder = {
  type: "folder";
  id: string;
  name: string;
  path: string;
  children: WorkspaceNode[];
};

export type WorkspaceNode = WorkspaceFile | WorkspaceFolder;

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

export function flattenWorkspaceFiles(node: WorkspaceNode): WorkspaceFile[] {
  if (node.type === "file") return [node];
  return node.children.flatMap(flattenWorkspaceFiles);
}

export function getFirstWorkspaceFile(root: WorkspaceFolder): WorkspaceFile {
  const firstFile = flattenWorkspaceFiles(root)[0];
  if (!firstFile) {
    throw new Error("Workspace sample data must contain at least one file");
  }
  return firstFile;
}

export function getWorkspaceFileById(
  root: WorkspaceFolder,
  fileId: string,
): WorkspaceFile | undefined {
  return flattenWorkspaceFiles(root).find((file) => file.id === fileId);
}

export function getWorkspaceViewModes(
  file: WorkspaceFile,
): WorkspaceViewMode[] {
  if (file.language === "markdown") return ["raw", "preview"];
  return ["raw"];
}
