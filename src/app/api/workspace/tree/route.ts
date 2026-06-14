import { NextResponse } from "next/server";
import { readDirectoryTree } from "@/lib/workspace-api";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const dirPath = searchParams.get("path");

  if (!dirPath) {
    return NextResponse.json(
      { error: "Missing 'path' query parameter" },
      { status: 400 },
    );
  }

  try {
    const tree = await readDirectoryTree(dirPath);
    return NextResponse.json({ tree });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";

    if (message.includes("ENOENT") || message.includes("not found")) {
      return NextResponse.json({ error: "Directory not found" }, { status: 404 });
    }
    if (message.includes("EACCES") || message.includes("permission")) {
      return NextResponse.json({ error: "Permission denied" }, { status: 403 });
    }

    return NextResponse.json({ error: message }, { status: 500 });
  }
}
