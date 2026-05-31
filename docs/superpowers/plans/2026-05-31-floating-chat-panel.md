# Floating Chat Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a workspace-local floating AI chat panel that overlays the `/workspace` editor area without resizing the editor.

**Architecture:** The workspace editor area becomes a relative positioning container. A new `WorkspaceChatPanel` component owns open/closed state, renders a left-edge handle, and renders an 85% width/height centered assistant-ui chat overlay that connects to `/api/chat`.

**Tech Stack:** Next.js App Router, React 19, TypeScript, Tailwind CSS 4, shadcn-ui `Button`/`Tooltip`, lucide-react, assistant-ui React AI SDK runtime.

---

## File Structure

- Create `src/components/workspace/workspace-chat-panel.tsx`: floating panel, handle, assistant-ui runtime.
- Modify `src/app/workspace/page.tsx`: wrap the editor pane in a relative container and render `WorkspaceChatPanel`.

## Task 1: Add Workspace Chat Panel Component

**Files:**
- Create: `src/components/workspace/workspace-chat-panel.tsx`

- [ ] **Step 1: Create the floating panel component**

Create `src/components/workspace/workspace-chat-panel.tsx` with this content:

```tsx
"use client";

import { Thread } from "@/components/assistant-ui/thread";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import {
  AssistantChatTransport,
  useChatRuntime,
} from "@assistant-ui/react-ai-sdk";
import { MessageSquareIcon, XIcon } from "lucide-react";
import { useMemo, useState } from "react";

export function WorkspaceChatPanel() {
  const [open, setOpen] = useState(false);
  const transport = useMemo(
    () => new AssistantChatTransport({ api: "/api/chat" }),
    [],
  );
  const runtime = useChatRuntime({ transport });

  return (
    <>
      {!open && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              variant="outline"
              size="icon"
              onClick={() => setOpen(true)}
              className="absolute left-3 top-1/2 z-10 size-9 -translate-y-1/2 rounded-full bg-background shadow-md"
              aria-label="Open AI chat"
            >
              <MessageSquareIcon className="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="right">Open AI chat</TooltipContent>
        </Tooltip>
      )}

      {open && (
        <div
          className="animate-in fade-in-0 slide-in-from-left-8 absolute left-1/2 top-1/2 z-20 flex h-[85%] w-[85%] -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-lg border bg-background shadow-2xl duration-200"
          role="dialog"
          aria-label="AI chat panel"
        >
          <header className="flex h-12 shrink-0 items-center justify-between border-b px-3">
            <div className="flex items-center gap-2">
              <MessageSquareIcon className="size-4 text-muted-foreground" />
              <h2 className="text-sm font-medium">AI Chat</h2>
            </div>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => setOpen(false)}
              aria-label="Close AI chat"
            >
              <XIcon className="size-4" />
            </Button>
          </header>
          <AssistantRuntimeProvider runtime={runtime}>
            <div className="min-h-0 flex-1">
              <Thread />
            </div>
          </AssistantRuntimeProvider>
        </div>
      )}
    </>
  );
}
```

- [ ] **Step 2: Verify component compiles**

Run:

```bash
npm run build
```

Expected:

- Build passes because the component is not mounted yet.

## Task 2: Mount Panel In Workspace Editor Area

**Files:**
- Modify: `src/app/workspace/page.tsx`

- [ ] **Step 1: Import `WorkspaceChatPanel`**

Update the import block in `src/app/workspace/page.tsx`:

```tsx
import { FileTree } from "@/components/workspace/file-tree";
import { WorkspaceChatPanel } from "@/components/workspace/workspace-chat-panel";
import { WorkspaceEditorShell } from "@/components/workspace/workspace-editor-shell";
```

- [ ] **Step 2: Wrap editor pane in a relative container**

Replace the direct `WorkspaceEditorShell` render at the end of `src/app/workspace/page.tsx` with:

```tsx
      <section className="relative flex min-w-0 flex-1">
        <WorkspaceEditorShell
          file={selectedFile}
          content={content}
          dirty={dirty}
          viewMode={viewMode}
          availableViewModes={availableViewModes}
          onViewModeChange={setViewMode}
          onContentChange={(nextContent) =>
            setEditedContents((current) => ({
              ...current,
              [selectedFile.id]: nextContent,
            }))
          }
        />
        <WorkspaceChatPanel />
      </section>
```

- [ ] **Step 3: Verify route compiles**

Run:

```bash
npm run build
```

Expected:

- Build passes.
- `/workspace` remains in the route table.
- `/api/chat` remains unchanged.

## Task 3: Browser Verification

**Files:**
- No file changes.

- [ ] **Step 1: Start dev server**

Run:

```bash
npm run dev
```

Expected:

- Next dev server starts at `http://localhost:3000` or reports another available port.

- [ ] **Step 2: Verify workspace floating panel**

Open:

```text
http://localhost:3000/workspace
```

Expected:

- File tree is visible.
- Editor is visible.
- AI chat handle appears on the left edge of the editor area.
- Clicking the handle opens the panel.
- Panel is centered inside the editor area.
- Panel uses about 85% of editor width and 85% of editor height.
- File tree remains visible while panel is open.
- Editor layout does not resize when panel opens.
- Panel contains the assistant-ui chat input.
- Clicking `X` closes the panel.

- [ ] **Step 3: Verify full-page chat still renders**

Open:

```text
http://localhost:3000
```

Expected:

- Full-page assistant-ui chat renders with composer.

## Self-Review

- Spec coverage: The plan covers the workspace-local overlay, 85% editor-area sizing, centered placement, left-edge handle, `X` close button, assistant-ui runtime, unchanged `/api/chat`, and browser verification.
- Completeness scan: Every task has concrete file paths, commands, and expected results.
- Type consistency: `WorkspaceChatPanel`, `AssistantChatTransport`, `useChatRuntime`, and the workspace route imports are named consistently.
