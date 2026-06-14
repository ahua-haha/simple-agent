"use client";

import { FileTree } from "@/components/workspace/file-tree";
import { WorkspaceChatPanel } from "@/components/workspace/workspace-chat-panel";
import { WorkspaceEditorShell } from "@/components/workspace/workspace-editor-shell";
import { WorkspacePicker } from "@/components/workspace/workspace-picker";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { getWorkspaceViewModes } from "@/lib/workspace-types";
import { useWorkspaceStore } from "@/lib/workspace-store";
import type { WorkspaceViewMode } from "@/lib/workspace-types";
import { XIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

export default function WorkspacePage() {
  const {
    workspacePath,
    workspaceName,
    fileTree,
    isLoadingTree,
    treeError,
    selectedFileId,
    openedFiles,
    isLoadingFile,
    isSaving,
    openWorkspace,
    closeWorkspace,
    selectFile,
    updateFileContent,
    saveCurrentFile,
  } = useWorkspaceStore();

  const [viewMode, setViewMode] = useState<WorkspaceViewMode>("raw");

  // Derive current file state (always called, even when no workspace)
  const selectedFile = selectedFileId ? openedFiles[selectedFileId] : undefined;
  const content = selectedFile?.content ?? "";
  const dirty = selectedFile
    ? selectedFile.content !== selectedFile.originalContent
    : false;
  const availableViewModes = useMemo(() => {
    if (!selectedFile) return ["raw"] as WorkspaceViewMode[];
    return getWorkspaceViewModes({
      type: "file",
      id: selectedFileId!,
      name: selectedFile.name,
      path: selectedFile.path,
      language: selectedFile.language,
      content: selectedFile.content,
    });
  }, [selectedFile, selectedFileId]);

  // Reset view mode when switching files if current mode isn't available
  useEffect(() => {
    if (!availableViewModes.includes(viewMode)) {
      setViewMode("raw");
    }
  }, [availableViewModes, viewMode]);

  // If no workspace is open, show the picker
  if (!workspacePath) {
    return <WorkspacePicker />;
  }

  const editorFile = selectedFile
    ? {
        name: selectedFile.name,
        path: selectedFile.path,
        language: selectedFile.language,
        isBinary: selectedFile.isBinary,
      }
    : undefined;

  return (
    <main className="flex h-dvh overflow-hidden bg-background text-foreground">
      {/* Sidebar */}
      <aside className="flex w-72 shrink-0 flex-col border-r bg-muted/30">
        <div className="flex h-14 shrink-0 items-center justify-between px-4">
          <div className="min-w-0">
            <h1 className="truncate text-sm font-medium">{workspaceName}</h1>
            <p className="truncate text-xs text-muted-foreground">
              {workspacePath}
            </p>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={closeWorkspace}
            className="size-7 shrink-0"
            aria-label="Close workspace"
          >
            <XIcon className="size-4" />
          </Button>
        </div>
        <Separator />
        <ScrollArea className="min-h-0 flex-1">
          <div className="p-2">
            <FileTree
              root={fileTree}
              selectedFileId={selectedFileId}
              onSelectFile={selectFile}
              isLoading={isLoadingTree}
              error={treeError}
            />
          </div>
        </ScrollArea>
      </aside>

      {/* Editor area */}
      <section className="relative flex min-w-0 flex-1">
        {isLoadingFile ? (
          <div className="flex flex-1 items-center justify-center">
            <p className="text-sm text-muted-foreground">Loading file…</p>
          </div>
        ) : (
          <WorkspaceEditorShell
            file={editorFile}
            content={content}
            dirty={dirty}
            viewMode={viewMode}
            availableViewModes={availableViewModes}
            onViewModeChange={setViewMode}
            onContentChange={(nextContent) => {
              if (selectedFileId) {
                updateFileContent(selectedFileId, nextContent);
              }
            }}
            onSave={saveCurrentFile}
            isSaving={isSaving}
          />
        )}
        <WorkspaceChatPanel />
      </section>
    </main>
  );
}
