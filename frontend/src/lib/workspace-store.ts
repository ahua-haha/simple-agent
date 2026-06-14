import { create } from "zustand";
import type { WorkspaceFolder, WorkspaceLanguage } from "./workspace-types";

// ---- Recent workspaces (localStorage) ----

const RECENT_KEY = "workspace:recent";
const MAX_RECENT = 5;

export type RecentWorkspace = {
  path: string;
  name: string;
  lastOpened: string; // ISO string
};

export function getRecentWorkspaces(): RecentWorkspace[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed as RecentWorkspace[];
  } catch {
    return [];
  }
}

function addRecentWorkspace(workspacePath: string): void {
  if (typeof window === "undefined") return;
  try {
    const recent = getRecentWorkspaces();
    const name = workspacePath.split("/").pop() || workspacePath;
    const existing = recent.findIndex((r) => r.path === workspacePath);
    if (existing !== -1) {
      recent.splice(existing, 1);
    }
    recent.unshift({ path: workspacePath, name, lastOpened: new Date().toISOString() });
    localStorage.setItem(RECENT_KEY, JSON.stringify(recent.slice(0, MAX_RECENT)));
  } catch {
    // localStorage not available
  }
}

// ---- Opened file cache ----

export type OpenedFile = {
  path: string;
  absolutePath: string;
  name: string;
  language: WorkspaceLanguage;
  content: string;
  originalContent: string;
  isBinary: boolean;
};

// ---- Store ----

export type WorkspaceState = {
  // Workspace
  workspacePath: string | null;
  workspaceName: string | null;
  fileTree: WorkspaceFolder | null;
  isLoadingTree: boolean;
  treeError: string | null;

  // Current file
  selectedFileId: string | null;
  openedFiles: Record<string, OpenedFile>;
  isLoadingFile: boolean;
  fileError: string | null;

  // Save
  isSaving: boolean;
  saveError: string | null;

  // Actions
  openWorkspace: (absPath: string) => Promise<void>;
  closeWorkspace: () => void;
  selectFile: (fileId: string) => Promise<void>;
  updateFileContent: (fileId: string, content: string) => void;
  saveCurrentFile: () => Promise<void>;
};

export const useWorkspaceStore = create<WorkspaceState>()((set, get) => ({
  workspacePath: null,
  workspaceName: null,
  fileTree: null,
  isLoadingTree: false,
  treeError: null,

  selectedFileId: null,
  openedFiles: {},
  isLoadingFile: false,
  fileError: null,

  isSaving: false,
  saveError: null,

  openWorkspace: async (absPath: string) => {
    set({ isLoadingTree: true, treeError: null, fileTree: null });

    try {
      const url = `/api/workspace/tree?path=${encodeURIComponent(absPath)}`;
      const res = await fetch(url);
      const json = await res.json();

      if (!res.ok) {
        set({
          isLoadingTree: false,
          treeError: (json.error as string) ?? "Failed to open workspace",
        });
        return;
      }

      const name = absPath.split("/").pop() || absPath;
      addRecentWorkspace(absPath);

      set({
        workspacePath: absPath,
        workspaceName: name,
        fileTree: json.tree as WorkspaceFolder,
        isLoadingTree: false,
        treeError: null,
        selectedFileId: null,
        openedFiles: {},
        fileError: null,
      });
    } catch (err) {
      set({
        isLoadingTree: false,
        treeError: err instanceof Error ? err.message : "Network error",
      });
    }
  },

  closeWorkspace: () => {
    set({
      workspacePath: null,
      workspaceName: null,
      fileTree: null,
      isLoadingTree: false,
      treeError: null,
      selectedFileId: null,
      openedFiles: {},
      isLoadingFile: false,
      fileError: null,
      isSaving: false,
      saveError: null,
    });
  },

  selectFile: async (fileId: string) => {
    const { openedFiles, workspacePath } = get();

    // Already loaded — just switch selection
    if (openedFiles[fileId]) {
      set({ selectedFileId: fileId, fileError: null });
      return;
    }

    if (!workspacePath) return;

    // Build absolute path: workspace root + relative file path
    const absolutePath = fileId.startsWith("/")
      ? fileId
      : `${workspacePath}/${fileId}`;

    set({ isLoadingFile: true, fileError: null });

    try {
      const url = `/api/workspace/read?path=${encodeURIComponent(absolutePath)}`;
      const res = await fetch(url);
      const json = await res.json();

      if (!res.ok) {
        set({
          isLoadingFile: false,
          fileError: (json.error as string) ?? "Failed to read file",
        });
        return;
      }

      const name = fileId.split("/").pop() || fileId;

      set({
        selectedFileId: fileId,
        isLoadingFile: false,
        fileError: null,
        openedFiles: {
          ...get().openedFiles,
          [fileId]: {
            path: fileId,
            absolutePath: json.path as string,
            name,
            language: json.language as WorkspaceLanguage,
            content: json.content as string,
            originalContent: json.content as string,
            isBinary: json.isBinary as boolean,
          },
        },
      });
    } catch (err) {
      set({
        isLoadingFile: false,
        fileError: err instanceof Error ? err.message : "Network error",
      });
    }
  },

  updateFileContent: (fileId: string, content: string) => {
    const { openedFiles } = get();
    const file = openedFiles[fileId];
    if (!file) return;

    set({
      openedFiles: {
        ...openedFiles,
        [fileId]: { ...file, content },
      },
      saveError: null,
    });
  },

  saveCurrentFile: async () => {
    const { selectedFileId, openedFiles } = get();
    if (!selectedFileId) return;

    const file = openedFiles[selectedFileId];
    if (!file || file.content === file.originalContent) return;

    set({ isSaving: true, saveError: null });

    try {
      const res = await fetch("/api/workspace/write", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          path: file.absolutePath,
          content: file.content,
        }),
      });
      const json = await res.json();

      if (!res.ok) {
        set({
          isSaving: false,
          saveError: (json.error as string) ?? "Failed to save file",
        });
        return;
      }

      set({
        isSaving: false,
        saveError: null,
        openedFiles: {
          ...get().openedFiles,
          [selectedFileId]: {
            ...file,
            originalContent: file.content,
          },
        },
      });
    } catch (err) {
      set({
        isSaving: false,
        saveError: err instanceof Error ? err.message : "Network error",
      });
    }
  },
}));
