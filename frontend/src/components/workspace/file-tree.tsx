"use client";

import type { WorkspaceNode } from "@/lib/workspace-types";
import { cn } from "@/lib/utils";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  ChevronRightIcon,
  FileCodeIcon,
  FileIcon,
  FileJsonIcon,
  FileTextIcon,
  FolderIcon,
  FolderOpenIcon,
} from "lucide-react";
import { useState } from "react";

type FileTreeProps = {
  root: WorkspaceNode | null;
  selectedFileId: string | null;
  onSelectFile: (fileId: string) => void;
  isLoading?: boolean;
  error?: string | null;
};

export function FileTree({
  root,
  selectedFileId,
  onSelectFile,
  isLoading = false,
  error = null,
}: FileTreeProps) {
  if (isLoading) {
    return (
      <div className="space-y-1 px-1 py-0.5">
        {Array.from({ length: 8 }).map((_, i) => (
          <div
            key={i}
            className="h-7 animate-pulse rounded-md bg-muted"
            style={{ width: `${60 + (i % 4) * 12}%`, marginLeft: `${(i % 3) * 12}px` }}
          />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="px-3 py-4 text-center text-sm text-destructive">
        <p>Failed to load files:</p>
        <p className="mt-1 text-xs text-muted-foreground">{error}</p>
      </div>
    );
  }

  if (!root) {
    return null;
  }

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

type FileTreeNodeProps = {
  node: WorkspaceNode;
  selectedFileId: string | null;
  onSelectFile: (fileId: string) => void;
  depth: number;
};

function FileTreeNode({
  node,
  selectedFileId,
  onSelectFile,
  depth,
}: FileTreeNodeProps) {
  if (node.type === "folder") {
    return (
      <FolderTreeNode
        node={node}
        selectedFileId={selectedFileId}
        onSelectFile={onSelectFile}
        depth={depth}
      />
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

function FolderTreeNode({
  node,
  selectedFileId,
  onSelectFile,
  depth,
}: {
  node: WorkspaceNode & { type: "folder" };
  selectedFileId: string | null;
  onSelectFile: (fileId: string) => void;
  depth: number;
}) {
  const isRoot = depth === 0;
  // Root folder is always open; sub-folders start collapsed unless they contain the selected file
  const startsOpen = isRoot || containsSelected(node, selectedFileId);
  const [open, setOpen] = useState(startsOpen);

  return (
    <Collapsible open={open} onOpenChange={isRoot ? undefined : setOpen}>
      <div>
        {isRoot ? (
          <div
            className={cn(
              "flex h-7 items-center gap-1.5 rounded-md px-2 font-medium text-foreground",
            )}
            style={{ paddingLeft: `${depth * 12 + 8}px` }}
          >
            <FolderOpenIcon className="size-4" />
            <span className="truncate">{node.name}</span>
          </div>
        ) : (
          <CollapsibleTrigger asChild>
            <button
              type="button"
              className={cn(
                "flex h-7 w-full items-center gap-1.5 rounded-md px-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
              )}
              style={{ paddingLeft: `${depth * 12 + 8}px` }}
            >
              <ChevronRightIcon
                className={cn(
                  "size-3.5 shrink-0 transition-transform",
                  open && "rotate-90",
                )}
              />
              {open ? (
                <FolderOpenIcon className="size-4" />
              ) : (
                <FolderIcon className="size-4" />
              )}
              <span className="truncate">{node.name}</span>
            </button>
          </CollapsibleTrigger>
        )}
      </div>
      <CollapsibleContent>
        {node.children.length === 0 ? (
          <p
            className="py-1 text-xs text-muted-foreground/60"
            style={{ paddingLeft: `${(depth + 1) * 12 + 8}px` }}
          >
            (empty)
          </p>
        ) : (
          node.children.map((child) => (
            <FileTreeNode
              key={child.id}
              node={child}
              selectedFileId={selectedFileId}
              onSelectFile={onSelectFile}
              depth={depth + 1}
            />
          ))
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}

/** Check if a folder (or its descendants) contains the currently selected file. */
function containsSelected(
  folder: WorkspaceNode & { type: "folder" },
  selectedFileId: string | null,
): boolean {
  if (!selectedFileId) return false;
  for (const child of folder.children) {
    if (child.type === "file" && child.id === selectedFileId) return true;
    if (child.type === "folder" && containsSelected(child, selectedFileId)) return true;
  }
  return false;
}

function getFileIcon(fileName: string) {
  if (
    fileName.endsWith(".tsx") ||
    fileName.endsWith(".ts") ||
    fileName.endsWith(".mts") ||
    fileName.endsWith(".cts")
  ) {
    return FileCodeIcon;
  }
  if (
    fileName.endsWith(".jsx") ||
    fileName.endsWith(".js") ||
    fileName.endsWith(".mjs") ||
    fileName.endsWith(".cjs")
  ) {
    return FileCodeIcon;
  }
  if (fileName.endsWith(".json")) return FileJsonIcon;
  if (fileName.endsWith(".md") || fileName.endsWith(".mdx") || fileName.endsWith(".txt")) {
    return FileTextIcon;
  }
  return FileIcon;
}
