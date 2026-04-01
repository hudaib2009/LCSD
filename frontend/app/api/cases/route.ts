import { NextResponse } from "next/server";
import { randomUUID } from "crypto";
import type { CaseRecord, Modality } from "@/lib/types";
import { addCase, readCases, saveUpload } from "@/lib/store";
import { sanitizeCaseForApi } from "@/lib/apiSanitizers";

export const runtime = "nodejs";

export async function GET() {
  const cases = await readCases();
  return NextResponse.json(cases.map(sanitizeCaseForApi));
}

export async function POST(request: Request) {
  const formData = await request.formData();
  const modality = formData.get("modality") as Modality | null;
  const file = formData.get("file");

  if (!modality || !file || !(file instanceof File)) {
    return NextResponse.json(
      { error: "Missing modality or file." },
      { status: 400 }
    );
  }

  const id = `case-${randomUUID().slice(0, 8)}`;
  const fileName = file.name || "uploaded-study";
  const buffer = Buffer.from(await file.arrayBuffer());

  await saveUpload(id, fileName, buffer);

  const caseRecord: CaseRecord = {
    id,
    createdAt: new Date().toISOString(),
    modality,
    patientName: (formData.get("patientName") as string) || undefined,
    patientId: (formData.get("patientId") as string) || undefined,
    notes: (formData.get("notes") as string) || undefined,
    fileName,
    status: "pending",
    clinical: {
      age: formData.get("age") ? Number(formData.get("age")) : undefined,
      isSmoker: formData.get("isSmoker")
        ? formData.get("isSmoker") === "true"
        : undefined,
      packYears: formData.get("packYears")
        ? Number(formData.get("packYears"))
        : undefined,
      histology: (formData.get("histology") as string) || undefined,
      ecog: formData.get("ecog") ? Number(formData.get("ecog")) : undefined
    }
  };

  await addCase(caseRecord);

  const wantsJson = request.headers.get("accept")?.includes("application/json");
  if (wantsJson) {
    return NextResponse.json(sanitizeCaseForApi(caseRecord), { status: 201 });
  }

  return NextResponse.redirect(new URL(`/cases/${id}`, request.url));
}
