import { promises as fs } from "fs";
import path from "path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

const mimeTypes: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".tif": "image/tiff",
  ".tiff": "image/tiff"
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
    return NextResponse.json({ error: "Invalid path." }, { status: 400 });
  }

  try {
    const data = await fs.readFile(filePath);
    const ext = path.extname(filePath).toLowerCase();
    const contentType = mimeTypes[ext] ?? "application/octet-stream";

    return new NextResponse(data, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "private, max-age=300"
      }
    });
  } catch {
    return NextResponse.json({ error: "File not found." }, { status: 404 });
  }
}
