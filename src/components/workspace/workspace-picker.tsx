"use client";

import { Button } from "@/components/ui/button";
import {
  getRecentWorkspaces,
  useWorkspaceStore,
} from "@/lib/workspace-store";
import { FolderOpenIcon, FolderIcon } from "lucide-react";
import { useEffect, useState } from "react";

export function WorkspacePicker() {
  const { openWorkspace, isLoadingTree, treeError } = useWorkspaceStore();
  const [dirPath, setDirPath] = useState("");
  const [recent, setRecent] = useState<ReturnType<typeof getRecentWorkspaces>>(
    [],
  );

  useEffect(() => {
    setRecent(getRecentWorkspaces());
  }, []);

  const handleOpen = () => {
    const trimmed = dirPath.trim();
    if (!trimmed) return;
    openWorkspace(trimmed);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleOpen();
    }
  };

  return (
    <main className="flex h-dvh items-center justify-center bg-background">
      <div className="w-full max-w-md rounded-xl border bg-card p-8 shadow-sm">
        <div className="mb-6 text-center">
          <FolderOpenIcon className="mx-auto mb-3 size-10 text-muted-foreground" />
          <h1 className="text-lg font-semibold">Open a Workspace</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Enter a directory path to browse and edit files.
          </p>
        </div>

        <div className="space-y-3">
          <div className="flex gap-2">
            <input
              type="text"
              value={dirPath}
              onChange={(e) => setDirPath(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="/Users/name/projects/my-app"
              className="flex-1 rounded-lg border bg-background px-3 py-2 text-sm outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-ring focus:ring-2 focus:ring-ring/30"
              autoFocus
            />
            <Button
              onClick={handleOpen}
              disabled={!dirPath.trim() || isLoadingTree}
            >
              {isLoadingTree ? "Opening…" : "Open"}
            </Button>
          </div>

          {treeError && (
            <p className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {treeError}
            </p>
          )}

          {recent.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-medium text-muted-foreground">
                Recent workspaces
              </p>
              <div className="space-y-1">
                {recent.map((item) => (
                  <button
                    key={item.path}
                    type="button"
                    onClick={() => {
                      setDirPath(item.path);
                      openWorkspace(item.path);
                    }}
                    className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors hover:bg-muted"
                  >
                    <FolderIcon className="size-4 shrink-0 text-muted-foreground" />
                    <div className="min-w-0">
                      <p className="truncate font-medium">{item.name}</p>
                      <p className="truncate text-xs text-muted-foreground">
                        {item.path}
                      </p>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
