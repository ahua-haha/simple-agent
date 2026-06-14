export type WorkspaceLanguage =
  | "typescript"
  | "javascript"
  | "markdown"
  | "json"
  | "css"
  | "python"
  | "yaml"
  | "shell"
  | "html"
  | "text";

export type WorkspaceViewMode = "raw" | "preview" | "wysiwyg";

export type WorkspaceFile = {
  type: "file";
  id: string;
  name: string;
  path: string;
  language: WorkspaceLanguage;
  content: string;
  /** Absolute path on disk — only set for real (non-mock) files */
  absolutePath?: string;
};

export type WorkspaceFolder = {
  type: "folder";
  id: string;
  name: string;
  path: string;
  children: WorkspaceNode[];
};

export type WorkspaceNode = WorkspaceFile | WorkspaceFolder;

export function flattenWorkspaceFiles(node: WorkspaceNode): WorkspaceFile[] {
  if (node.type === "file") return [node];
  return node.children.flatMap(flattenWorkspaceFiles);
}

export function getFirstWorkspaceFile(root: WorkspaceFolder): WorkspaceFile {
  const firstFile = flattenWorkspaceFiles(root)[0];
  if (!firstFile) {
    throw new Error(
      "Workspace data must contain at least one file",
    );
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
