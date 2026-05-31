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
