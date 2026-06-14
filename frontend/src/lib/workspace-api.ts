import { readdir, readFile as fsReadFile, lstat, writeFile as fsWriteFile } from "node:fs/promises";
import path from "node:path";
import type { WorkspaceFolder, WorkspaceNode } from "./workspace-types";
import { getFileLanguage, shouldSkipEntry } from "./workspace-language";

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
const BINARY_SCAN_BYTES = 8192; // 8 KB

/**
 * Recursively walk a directory, returning a WorkspaceFolder tree.
 * Skips hidden files, well-known ignorable directories, and unreadable entries.
 */
export async function readDirectoryTree(
  absolutePath: string,
  rootPath: string = absolutePath,
): Promise<WorkspaceFolder> {
  const name = path.basename(absolutePath) || absolutePath;
  const relativePath = absolutePath === rootPath
    ? ""
    : path.relative(rootPath, absolutePath);

  const folder: WorkspaceFolder = {
    type: "folder",
    id: relativePath || name,
    name,
    path: relativePath,
    children: [],
  };

  let entries: string[];
  try {
    entries = await readdir(absolutePath);
  } catch {
    // Permission denied or doesn't exist — return empty folder
    return folder;
  }

  // Sort: folders first, then files; alphabetical within each group
  const withStats: { name: string; isDir: boolean }[] = [];
  for (const entry of entries) {
    if (shouldSkipEntry(entry)) continue;
    try {
      const stat = await lstat(path.join(absolutePath, entry));
      if (stat.isSymbolicLink()) continue; // skip symlinks to avoid cycles
      withStats.push({ name: entry, isDir: stat.isDirectory() });
    } catch {
      // Skip entries we can't stat
    }
  }

  withStats.sort((a, b) => {
    if (a.isDir !== b.isDir) return a.isDir ? -1 : 1;
    return a.name.localeCompare(b.name);
  });

  for (const { name: entryName, isDir } of withStats) {
    const entryPath = path.join(absolutePath, entryName);
    const entryRelativePath = path.relative(rootPath, entryPath);

    if (isDir) {
      const child = await readDirectoryTree(entryPath, rootPath);
      // Only include non-empty folders, or folders that aren't completely empty
      folder.children.push(child);
    } else {
      const language = getFileLanguage(entryName);
      const node: WorkspaceNode = {
        type: "file",
        id: entryRelativePath,
        name: entryName,
        path: entryRelativePath,
        language,
        content: "", // content loaded on demand
        absolutePath: entryPath,
      };
      folder.children.push(node);
    }
  }

  return folder;
}

export type ReadFileResult = {
  content: string;
  language: ReturnType<typeof getFileLanguage>;
  isBinary: boolean;
};

/**
 * Read a single file from disk.
 * Detects binary files by scanning for null bytes in the first 8 KB.
 * Returns empty content for binary files.
 */
export async function readFile(absolutePath: string): Promise<ReadFileResult> {
  const language = getFileLanguage(path.basename(absolutePath));

  let buffer: Buffer;
  try {
    const stat = await lstat(absolutePath);
    if (!stat.isFile()) {
      throw new Error("Path is not a file");
    }
    if (stat.size > MAX_FILE_SIZE) {
      throw new Error("File exceeds maximum size (10 MB)");
    }
    buffer = await fsReadFile(absolutePath);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") {
      throw new Error("File not found");
    }
    throw err;
  }

  // Detect binary by scanning for null bytes in the first portion
  const scanEnd = Math.min(buffer.length, BINARY_SCAN_BYTES);
  const isBinary = buffer.subarray(0, scanEnd).includes(0);

  if (isBinary) {
    return { content: "", language, isBinary: true };
  }

  // Try UTF-8 first, fall back to latin-1
  let content: string;
  try {
    const decoder = new TextDecoder("utf-8", { fatal: true });
    content = decoder.decode(buffer);
  } catch {
    // Treat as binary if UTF-8 decoding fails
    return { content: "", language, isBinary: true };
  }

  return { content, language, isBinary: false };
}

/**
 * Write content to a file on disk (UTF-8).
 * Only overwrites existing files — does not create new ones.
 */
export async function writeFile(
  absolutePath: string,
  content: string,
): Promise<void> {
  // Verify the file exists before writing
  try {
    const stat = await lstat(absolutePath);
    if (!stat.isFile()) {
      throw new Error("Path is not a file");
    }
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") {
      throw new Error("File not found");
    }
    throw err;
  }

  await fsWriteFile(absolutePath, content, "utf-8");
}
