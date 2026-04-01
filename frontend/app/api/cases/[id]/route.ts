import { NextResponse } from "next/server";
import { getCaseById, readResults } from "@/lib/store";
import { sanitizeCaseForApi, sanitizeResultsForApi } from "@/lib/apiSanitizers";

export const runtime = "nodejs";

export async function GET(
  _request: Request,
  { params }: { params: { id: string } }
) {
  const caseRecord = await getCaseById(params.id);
  if (!caseRecord) {
    return NextResponse.json({ error: "Case not found" }, { status: 404 });
  }

  const results = await readResults(params.id);
  return NextResponse.json({
    case: sanitizeCaseForApi(caseRecord),
    results: sanitizeResultsForApi(results),
  });
}
