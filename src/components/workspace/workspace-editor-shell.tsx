"use client";

import { FileViewSwitcher } from "@/components/workspace/file-view-switcher";
import { MarkdownPreview } from "@/components/workspace/markdown-preview";
import { MarkdownWysiwygEditor } from "@/components/workspace/markdown-wysiwyg-editor";
import { RawFileEditor } from "@/components/workspace/raw-file-editor";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import type {
  WorkspaceViewMode,
} from "@/lib/workspace-types";
import type { WorkspaceLanguage } from "@/lib/workspace-types";
import { FileCodeIcon, FileTextIcon, SaveIcon, Loader2Icon } from "lucide-react";
import { useEffect } from "react";

export type WorkspaceEditorFile = {
  name: string;
  path: string;
  language: WorkspaceLanguage;
  isBinary?: boolean;
};

type WorkspaceEditorShellProps = {
  file: WorkspaceEditorFile | undefined;
  content: string;
  dirty: boolean;
  viewMode: WorkspaceViewMode;
  availableViewModes: WorkspaceViewMode[];
  onViewModeChange: (mode: WorkspaceViewMode) => void;
  onContentChange: (content: string) => void;
  onSave?: () => void;
  isSaving?: boolean;
};

export function WorkspaceEditorShell({
  file,
  content,
  dirty,
  viewMode,
  availableViewModes,
  onViewModeChange,
  onContentChange,
  onSave,
  isSaving = false,
}: WorkspaceEditorShellProps) {
  // Ctrl+S / Cmd+S keyboard shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "s") {
        e.preventDefault();
        onSave?.();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onSave]);

  if (!file) {
    return (
      <section className="flex min-w-0 flex-1 items-center justify-center text-sm text-muted-foreground">
        Select a file to view its content.
      </section>
    );
  }

  if (file.isBinary) {
    return (
      <section className="flex min-w-0 flex-1 flex-col bg-background">
        <header className="flex h-14 shrink-0 items-center border-b px-4">
          <div className="flex min-w-0 items-center gap-3">
            <FileCodeIcon className="size-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <h1 className="truncate text-sm font-medium">{file.name}</h1>
              <p className="truncate text-xs text-muted-foreground">
                {file.path}
              </p>
            </div>
          </div>
        </header>
        <div className="flex flex-1 items-center justify-center">
          <p className="text-sm text-muted-foreground">
            This file appears to be binary and cannot be edited.
          </p>
        </div>
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
        <div className="flex items-center gap-2">
          {dirty && onSave && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onSave}
              disabled={isSaving}
            >
              {isSaving ? (
                <Loader2Icon className="size-3.5 animate-spin" />
              ) : (
                <SaveIcon className="size-3.5" />
              )}
              Save
            </Button>
          )}
          <FileViewSwitcher
            value={viewMode}
            modes={availableViewModes}
            onValueChange={onViewModeChange}
          />
        </div>
      </header>
      <Separator />
      <div className="min-h-0 flex-1">
        {viewMode === "preview" ? (
          <MarkdownPreview content={content} />
        ) : viewMode === "wysiwyg" ? (
          <MarkdownWysiwygEditor
            key={file.path}
            value={content}
            onChange={onContentChange}
          />
        ) : (
          <RawFileEditor value={content} onChange={onContentChange} />
        )}
      </div>
    </section>
  );
}
