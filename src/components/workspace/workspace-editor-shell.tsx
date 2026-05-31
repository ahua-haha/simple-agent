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
