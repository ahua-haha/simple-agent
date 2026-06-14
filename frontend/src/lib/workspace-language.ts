import type { WorkspaceLanguage } from "./workspace-types";

/**
 * Map a file extension to its workspace language for editor/syntax purposes.
 */
export function getFileLanguage(fileName: string): WorkspaceLanguage {
  const ext = fileName.split(".").pop()?.toLowerCase();

  switch (ext) {
    case "ts":
    case "tsx":
    case "mts":
    case "cts":
      return "typescript";

    case "js":
    case "jsx":
    case "mjs":
    case "cjs":
      return "javascript";

    case "md":
    case "mdx":
      return "markdown";

    case "json":
    case "jsonc":
    case "json5":
      return "json";

    case "css":
    case "scss":
    case "less":
      return "css";

    case "py":
      return "python";

    case "yaml":
    case "yml":
      return "yaml";

    case "sh":
    case "bash":
    case "zsh":
      return "shell";

    case "html":
    case "htm":
      return "html";

    default:
      return "text";
  }
}

/** Directories that should never be scanned. */
const SKIP_DIRECTORIES = new Set([
  "node_modules",
  ".git",
  ".next",
  "dist",
  "build",
  ".cache",
  "coverage",
  ".vercel",
  ".turbo",
  "__pycache__",
  ".venv",
  "venv",
  ".idea",
  ".vscode",
]);

/**
 * Returns true if a file or directory entry should be omitted from the workspace tree.
 */
export function shouldSkipEntry(name: string): boolean {
  // Hidden files / folders (leading dot, except the root placeholder)
  if (name.startsWith(".")) return true;

  // Well-known ignorable directories
  if (SKIP_DIRECTORIES.has(name)) return true;

  return false;
}
