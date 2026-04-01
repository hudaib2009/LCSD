import { NextResponse } from "next/server";
import path from "path";
import { runInference } from "@/lib/inferenceClient";
import { getCaseById, saveResults, updateCase } from "@/lib/store";
import { sanitizeCaseForApi, sanitizeResultsForApi } from "@/lib/apiSanitizers";

export const runtime = "nodejs";

export async function POST(
  _request: Request,
  { params }: { params: { id: string } }
) {
  const caseRecord = await getCaseById(params.id);
  if (!caseRecord) {
    return NextResponse.json({ error: "Case not found" }, { status: 404 });
  }

  await updateCase(params.id, { status: "processing" });

  try {
    const imagePath = path.join(
      process.cwd(),
      "storage",
      params.id,
      caseRecord.fileName
    );
    const results = await runInference(
      caseRecord.modality,
      params.id,
      imagePath
    );
    const updatedAt = new Date().toISOString();
    results.updatedAt = updatedAt;
    console.log("[api/cases/run] explainability", results.explainability);
    await saveResults(params.id, results);
    await updateCase(params.id, {
      status: "complete",
      lastRunAt: results.runAt
    });

    return NextResponse.json({
      case: sanitizeCaseForApi(caseRecord),
      results: sanitizeResultsForApi(results),
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Inference failed.";
    await updateCase(params.id, { status: "pending" });
    return NextResponse.json({ error: message }, { status: 503 });
  }
}
