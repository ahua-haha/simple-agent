import { NextResponse } from "next/server";
import { readFile } from "@/lib/workspace-api";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const filePath = searchParams.get("path");

  if (!filePath) {
    return NextResponse.json(
      { error: "Missing 'path' query parameter" },
      { status: 400 },
    );
  }

  try {
    const result = await readFile(filePath);
    return NextResponse.json({ ...result, path: filePath });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";

    if (message.includes("not found") || message.includes("ENOENT")) {
      return NextResponse.json({ error: "File not found" }, { status: 404 });
    }
    if (message.includes("not a file")) {
      return NextResponse.json({ error: "Path is not a file" }, { status: 400 });
    }
    if (message.includes("exceeds maximum")) {
      return NextResponse.json({ error: message }, { status: 413 });
    }

    return NextResponse.json({ error: message }, { status: 500 });
  }
}
