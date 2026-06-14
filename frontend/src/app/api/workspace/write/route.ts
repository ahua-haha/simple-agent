import { NextResponse } from "next/server";
import { writeFile } from "@/lib/workspace-api";

export async function PUT(request: Request) {
  let body: { path?: string; content?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid JSON body" },
      { status: 400 },
    );
  }

  const { path: filePath, content } = body;

  if (!filePath || typeof filePath !== "string") {
    return NextResponse.json(
      { error: "Missing 'path' field in request body" },
      { status: 400 },
    );
  }

  if (typeof content !== "string") {
    return NextResponse.json(
      { error: "Missing 'content' field in request body" },
      { status: 400 },
    );
  }

  try {
    await writeFile(filePath, content);
    return NextResponse.json({ success: true });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";

    if (message.includes("not found") || message.includes("ENOENT")) {
      return NextResponse.json({ error: "File not found" }, { status: 404 });
    }
    if (message.includes("not a file")) {
      return NextResponse.json({ error: "Path is not a file" }, { status: 400 });
    }
    if (message.includes("EACCES") || message.includes("permission")) {
      return NextResponse.json({ error: "Permission denied" }, { status: 403 });
    }

    return NextResponse.json({ error: message }, { status: 500 });
  }
}
