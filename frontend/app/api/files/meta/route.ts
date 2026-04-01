import { promises as fs } from "fs";
import path from "path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

const mimeTypes: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp"
};

export async function GET(request: Request) {
  const url = new URL(request.url);
  const relPath = url.searchParams.get("path");

  if (!relPath) {
    return NextResponse.json(
      { error: "Missing required path parameter." },
      { status: 400 }
    );
  }

  const storageRoot = path.resolve(process.cwd(), "storage");
  const filePath = path.resolve(storageRoot, relPath);

  if (!filePath.startsWith(storageRoot)) {
    return NextResponse.json(
      {
        exists: false,
        size: null,
        contentType: null
      },
      { status: 400 }
    );
  }

  try {
    const stat = await fs.stat(filePath);
    const ext = path.extname(filePath).toLowerCase();
    const contentType = mimeTypes[ext] ?? "application/octet-stream";
    return NextResponse.json({
      exists: true,
      size: stat.size,
      contentType
    });
  } catch {
    return NextResponse.json({
      exists: false,
      size: null,
      contentType: null
    });
  }
}
