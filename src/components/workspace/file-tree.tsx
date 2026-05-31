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

type FileTreeNodeProps = {
  node: WorkspaceNode;
  selectedFileId: string;
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
