# Workspace File Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a mock `/workspace` page with a shadcn-ui file tree sidebar and editable raw/markdown-preview file viewer.

**Architecture:** The feature is a client-side mock workspace. Static sample file data lives in a small library module, the `/workspace` route owns selected-file/edit/view state, and focused components render the file tree, editor shell, raw editor, view switcher, and markdown preview.

**Tech Stack:** Next.js App Router, React 19, TypeScript, Tailwind CSS 4, shadcn-ui, lucide-react, react-markdown, remark-gfm.

---

## File Structure

- Create `src/lib/workspace-sample-data.ts`: sample file tree, types, tree flattening helper, first-file helper, view capability helper.
- Create `src/components/workspace/file-tree.tsx`: recursive sidebar tree.
- Create `src/components/workspace/file-view-switcher.tsx`: view mode tabs.
- Create `src/components/workspace/raw-file-editor.tsx`: editable raw content surface.
- Create `src/components/workspace/markdown-preview.tsx`: read-only markdown preview.
- Create `src/components/workspace/workspace-editor-shell.tsx`: right pane header and content shell.
- Create `src/app/workspace/page.tsx`: route-level client page and mock state owner.
- Add shadcn components: `scroll-area`, `separator`, `tabs`, `badge`.
- Add dependency: `react-markdown`.

## Task 1: Add UI And Markdown Dependencies

**Files:**
- Modify: `package.json`
- Modify: `package-lock.json`
- Create: `src/components/ui/scroll-area.tsx`
- Create: `src/components/ui/separator.tsx`
- Create: `src/components/ui/tabs.tsx`
- Create: `src/components/ui/badge.tsx`

- [ ] **Step 1: Install registry components and markdown renderer**

Run:

```bash
npx shadcn@latest add scroll-area separator tabs badge --yes
npm install react-markdown
```

Expected:

- shadcn creates the four UI component files.
- `react-markdown` is added to `package.json`.
- `remark-gfm` is already present and remains in dependencies.

- [ ] **Step 2: Verify dependency install**

Run:

```bash
npm run build
```

Expected:

- Build passes.
- Existing assistant chat route remains available.

## Task 2: Add Workspace Sample Data And Helpers

**Files:**
- Create: `src/lib/workspace-sample-data.ts`

- [ ] **Step 1: Create sample data module**

Create `src/lib/workspace-sample-data.ts` with this content:

```ts
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
```

- [ ] **Step 2: Verify TypeScript module**

Run:

```bash
npm run build
```

Expected:

- Build passes because the new module is self-contained.

## Task 3: Add File Tree Component

**Files:**
- Create: `src/components/workspace/file-tree.tsx`

- [ ] **Step 1: Create recursive file tree**

Create `src/components/workspace/file-tree.tsx` with this content:

```tsx
"use client";

import type { WorkspaceNode } from "@/lib/workspace-sample-data";
import { cn } from "@/lib/utils";
import {
  ChevronDownIcon,
  FileCodeIcon,
  FileIcon,
  FileJsonIcon,
  FileTextIcon,
  FolderIcon,
  FolderOpenIcon,
} from "lucide-react";

type FileTreeProps = {
  root: WorkspaceNode;
  selectedFileId: string;
  onSelectFile: (fileId: string) => void;
};

export function FileTree({
  root,
  selectedFileId,
  onSelectFile,
}: FileTreeProps) {
  return (
    <div className="space-y-0.5 text-sm">
      <FileTreeNode
        node={root}
        selectedFileId={selectedFileId}
        onSelectFile={onSelectFile}
        depth={0}
      />
    </div>
  );
}

function FileTreeNode({
  node,
  selectedFileId,
  onSelectFile,
  depth,
}: FileTreeProps & {
  node: WorkspaceNode;
  depth: number;
}) {
  if (node.type === "folder") {
    const isRoot = depth === 0;

    return (
      <div>
        <div
          className={cn(
            "flex h-7 items-center gap-1.5 rounded-md px-2 text-muted-foreground",
            isRoot && "font-medium text-foreground",
          )}
          style={{ paddingLeft: `${depth * 12 + 8}px` }}
        >
          {!isRoot && <ChevronDownIcon className="size-3.5" />}
          {isRoot ? (
            <FolderOpenIcon className="size-4" />
          ) : (
            <FolderIcon className="size-4" />
          )}
          <span className="truncate">{node.name}</span>
        </div>
        <div>
          {node.children.map((child) => (
            <FileTreeNode
              key={child.id}
              root={child}
              node={child}
              selectedFileId={selectedFileId}
              onSelectFile={onSelectFile}
              depth={depth + 1}
            />
          ))}
        </div>
      </div>
    );
  }

  const selected = node.id === selectedFileId;
  const Icon = getFileIcon(node.name);

  return (
    <button
      type="button"
      onClick={() => onSelectFile(node.id)}
      className={cn(
        "flex h-7 w-full items-center gap-1.5 rounded-md px-2 text-left text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
        selected && "bg-muted text-foreground",
      )}
      style={{ paddingLeft: `${depth * 12 + 8}px` }}
    >
      <Icon className="size-4 shrink-0" />
      <span className="truncate">{node.name}</span>
    </button>
  );
}

function getFileIcon(fileName: string) {
  if (fileName.endsWith(".tsx") || fileName.endsWith(".ts")) {
    return FileCodeIcon;
  }
  if (fileName.endsWith(".json")) return FileJsonIcon;
  if (fileName.endsWith(".md") || fileName.endsWith(".txt")) {
    return FileTextIcon;
  }
  return FileIcon;
}
```

- [ ] **Step 2: Run build to catch component errors**

Run:

```bash
npm run build
```

Expected:

- Build passes.

## Task 4: Add Editor View Components

**Files:**
- Create: `src/components/workspace/file-view-switcher.tsx`
- Create: `src/components/workspace/raw-file-editor.tsx`
- Create: `src/components/workspace/markdown-preview.tsx`

- [ ] **Step 1: Create file view switcher**

Create `src/components/workspace/file-view-switcher.tsx`:

```tsx
"use client";

import {
  Tabs,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import type { WorkspaceViewMode } from "@/lib/workspace-sample-data";

type FileViewSwitcherProps = {
  value: WorkspaceViewMode;
  modes: WorkspaceViewMode[];
  onValueChange: (value: WorkspaceViewMode) => void;
};

export function FileViewSwitcher({
  value,
  modes,
  onValueChange,
}: FileViewSwitcherProps) {
  return (
    <Tabs
      value={value}
      onValueChange={(nextValue) =>
        onValueChange(nextValue as WorkspaceViewMode)
      }
    >
      <TabsList>
        {modes.map((mode) => (
          <TabsTrigger key={mode} value={mode} className="capitalize">
            {mode === "raw" ? "Raw" : "Preview"}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}
```

- [ ] **Step 2: Create raw editor**

Create `src/components/workspace/raw-file-editor.tsx`:

```tsx
"use client";

type RawFileEditorProps = {
  value: string;
  onChange: (value: string) => void;
};

export function RawFileEditor({ value, onChange }: RawFileEditorProps) {
  return (
    <textarea
      value={value}
      onChange={(event) => onChange(event.target.value)}
      spellCheck={false}
      className="h-full min-h-0 w-full resize-none border-0 bg-slate-950 p-4 font-mono text-sm leading-6 text-slate-100 outline-none"
      aria-label="Raw file content"
    />
  );
}
```

- [ ] **Step 3: Create markdown preview**

Create `src/components/workspace/markdown-preview.tsx`:

```tsx
"use client";

import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

type MarkdownPreviewProps = {
  content: string;
};

export function MarkdownPreview({ content }: MarkdownPreviewProps) {
  return (
    <div className="h-full overflow-auto bg-background p-6 text-sm leading-6">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: (props) => (
            <h1 className="mb-4 text-2xl font-semibold" {...props} />
          ),
          h2: (props) => (
            <h2 className="mt-6 mb-3 text-lg font-semibold" {...props} />
          ),
          p: (props) => <p className="my-3" {...props} />,
          ul: (props) => (
            <ul className="my-3 list-disc space-y-1 pl-5" {...props} />
          ),
          ol: (props) => (
            <ol className="my-3 list-decimal space-y-1 pl-5" {...props} />
          ),
          code: (props) => (
            <code
              className="rounded bg-muted px-1 py-0.5 font-mono text-xs"
              {...props}
            />
          ),
          pre: (props) => (
            <pre
              className="my-4 overflow-auto rounded-lg bg-slate-950 p-4 text-slate-100"
              {...props}
            />
          ),
        }}
      >
        {content}
      </Markdown>
    </div>
  );
}
```

- [ ] **Step 4: Run build**

Run:

```bash
npm run build
```

Expected:

- Build passes after the view components are added.

## Task 5: Add Workspace Editor Shell

**Files:**
- Create: `src/components/workspace/workspace-editor-shell.tsx`

- [ ] **Step 1: Create editor shell**

Create `src/components/workspace/workspace-editor-shell.tsx`:

```tsx
"use client";

import { FileViewSwitcher } from "@/components/workspace/file-view-switcher";
import { MarkdownPreview } from "@/components/workspace/markdown-preview";
import { RawFileEditor } from "@/components/workspace/raw-file-editor";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import type {
  WorkspaceFile,
  WorkspaceViewMode,
} from "@/lib/workspace-sample-data";
import { FileCodeIcon, FileTextIcon } from "lucide-react";

type WorkspaceEditorShellProps = {
  file: WorkspaceFile | undefined;
  content: string;
  dirty: boolean;
  viewMode: WorkspaceViewMode;
  availableViewModes: WorkspaceViewMode[];
  onViewModeChange: (mode: WorkspaceViewMode) => void;
  onContentChange: (content: string) => void;
};

export function WorkspaceEditorShell({
  file,
  content,
  dirty,
  viewMode,
  availableViewModes,
  onViewModeChange,
  onContentChange,
}: WorkspaceEditorShellProps) {
  if (!file) {
    return (
      <section className="flex min-w-0 flex-1 items-center justify-center text-sm text-muted-foreground">
        Select a file to view its content.
      </section>
    );
  }

  const FileIcon = file.language === "markdown" ? FileTextIcon : FileCodeIcon;

  return (
    <section className="flex min-w-0 flex-1 flex-col bg-background">
      <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b px-4">
        <div className="flex min-w-0 items-center gap-3">
          <FileIcon className="size-4 shrink-0 text-muted-foreground" />
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="truncate text-sm font-medium">{file.name}</h1>
              {dirty && <Badge variant="secondary">Modified</Badge>}
            </div>
            <p className="truncate text-xs text-muted-foreground">
              {file.path}
            </p>
          </div>
        </div>
        <FileViewSwitcher
          value={viewMode}
          modes={availableViewModes}
          onValueChange={onViewModeChange}
        />
      </header>
      <Separator />
      <div className="min-h-0 flex-1">
        {viewMode === "preview" ? (
          <MarkdownPreview content={content} />
        ) : (
          <RawFileEditor value={content} onChange={onContentChange} />
        )}
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Run build**

Run:

```bash
npm run build
```

Expected:

- Build passes.

## Task 6: Add `/workspace` Route

**Files:**
- Create: `src/app/workspace/page.tsx`

- [ ] **Step 1: Create workspace route**

Create `src/app/workspace/page.tsx`:

```tsx
"use client";

import { FileTree } from "@/components/workspace/file-tree";
import { WorkspaceEditorShell } from "@/components/workspace/workspace-editor-shell";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  getFirstWorkspaceFile,
  getWorkspaceFileById,
  getWorkspaceViewModes,
  workspaceTree,
  type WorkspaceViewMode,
} from "@/lib/workspace-sample-data";
import { useEffect, useMemo, useState } from "react";

export default function WorkspacePage() {
  const firstFile = useMemo(() => getFirstWorkspaceFile(workspaceTree), []);
  const [selectedFileId, setSelectedFileId] = useState(firstFile.id);
  const [viewMode, setViewMode] = useState<WorkspaceViewMode>("raw");
  const [editedContents, setEditedContents] = useState<Record<string, string>>(
    {},
  );

  const selectedFile =
    getWorkspaceFileById(workspaceTree, selectedFileId) ?? firstFile;
  const availableViewModes = getWorkspaceViewModes(selectedFile);
  const content = editedContents[selectedFile.id] ?? selectedFile.content;
  const dirty = content !== selectedFile.content;

  useEffect(() => {
    if (!availableViewModes.includes(viewMode)) {
      setViewMode("raw");
    }
  }, [availableViewModes, viewMode]);

  return (
    <main className="flex h-dvh overflow-hidden bg-background text-foreground">
      <aside className="flex w-72 shrink-0 flex-col border-r bg-muted/30">
        <div className="flex h-14 shrink-0 items-center px-4">
          <div>
            <h1 className="text-sm font-medium">Workspace</h1>
            <p className="text-xs text-muted-foreground">Sample files</p>
          </div>
        </div>
        <Separator />
        <ScrollArea className="min-h-0 flex-1">
          <div className="p-2">
            <FileTree
              root={workspaceTree}
              selectedFileId={selectedFile.id}
              onSelectFile={(fileId) => {
                setSelectedFileId(fileId);
                const nextFile =
                  getWorkspaceFileById(workspaceTree, fileId) ?? firstFile;
                if (!getWorkspaceViewModes(nextFile).includes(viewMode)) {
                  setViewMode("raw");
                }
              }}
            />
          </div>
        </ScrollArea>
      </aside>
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
    </main>
  );
}
```

- [ ] **Step 2: Run build**

Run:

```bash
npm run build
```

Expected:

- Build passes.
- Route table includes `/workspace`.

## Task 7: Browser Verification

**Files:**
- No file changes.

- [ ] **Step 1: Start dev server**

Run:

```bash
npm run dev
```

Expected:

- Next dev server starts, usually at `http://localhost:3000`.

- [ ] **Step 2: Verify `/workspace` manually**

Open:

```text
http://localhost:3000/workspace
```

Expected:

- File tree appears in the left sidebar.
- `page.tsx` or the first sample file is selected.
- Raw editor appears in the right pane.
- Clicking `README.md` changes content and shows `Raw` and `Preview`.
- Clicking `Preview` renders markdown.
- Editing raw markdown content updates the preview after switching back to `Preview`.
- Dirty badge appears after editing raw content.
- Clicking `package.json` switches back to `Raw` and hides `Preview`.

- [ ] **Step 3: Verify chat page still works visually**

Open:

```text
http://localhost:3000
```

Expected:

- assistant-ui chat page still renders with the composer.

## Self-Review

- Spec coverage: The plan covers `/workspace`, classic split layout, sample data, raw editing, markdown preview, shadcn-ui components, dirty state, and browser verification.
- Placeholder scan: No placeholder tasks remain.
- Type consistency: `WorkspaceViewMode`, `WorkspaceFile`, `WorkspaceFolder`, and component prop names are consistent across tasks.
