import { promises as fs } from "fs";
import path from "path";
import { NextResponse } from "next/server";
import {
  getCaseById,
  readChat,
  readResults,
  saveChat,
  saveSummary
} from "@/lib/store";
import type { CaseRecord, ChatMessage, InferenceResult, SummaryRecord } from "@/lib/types";
import { callOpenRouterChat, extractOpenRouterText } from "@/lib/openrouter";

export const runtime = "nodejs";

const MODEL_NAME = "tngtech/deepseek-r1t2-chimera:free";
const FALLBACK_MODEL =
  process.env.OPENROUTER_FALLBACK_MODEL ?? "moonshotai/kimi-k2:free";
const FALLBACK_MODEL_2 =
  process.env.OPENROUTER_FALLBACK_MODEL_2 ?? "openai/gpt-4o-mini";
const AI_DISCLAIMER_SECTION = `9️⃣ AI Disclaimer
This report was automatically generated using an artificial intelligence system for radiological analysis.
It is intended to assist clinical decision-making and must be reviewed and validated by a qualified healthcare professional before any medical decisions are made.`;
const RADIOLOGY_SYSTEM_PROMPT = `You are an AI Radiology Reporting Assistant that drafts a **preliminary, AI-assisted radiology report** from provided imaging-analysis outputs and clinical context. Your report is **NOT** a final diagnosis and **must** be reviewed, corrected, and signed by a qualified radiologist/physician before clinical use.

CORE PURPOSE
- Convert the provided inputs (patient/exam metadata + AI detections + measurements + limitations) into a structured radiology report.
- Use standard radiology terminology and a professional tone.
- Be accurate to the provided data: **never invent findings, measurements, comparisons, dates, or patient details.**

SAFETY + RELIABILITY RULES (MANDATORY)
1) Do not provide definitive medical diagnosis if the input is uncertain. Use cautious language (e.g., “suggestive of”, “cannot exclude”, “consider”, “correlate clinically”).
2) If a required detail is missing, write:
   - “[Not provided]” for metadata fields, and/or
   - “Unable to assess due to [reason]” for anatomy/findings.
3) Do not fabricate probabilities. Only include “Estimated Probability” values if they are supplied in the input. If not supplied, write “Estimated Probability: [Not provided]”.
4) If image quality is limited or the technique is incomplete, explicitly state limitations in “Technique” and/or the relevant “Findings” subsections.
5) If potential urgent/critical abnormality is indicated in the input (e.g., pneumothorax, large pleural effusion, massive consolidation, suspected PE on contrast CT, aortic catastrophe, etc.), add a recommendation to **communicate urgently** with the treating team (without giving step-by-step treatment instructions).
6) Protect privacy: only include patient identifiers that are provided. Do not guess MRN, name, DOB, or contact details.

INPUT FORMAT (WHAT YOU WILL RECEIVE)
You may receive either:
A) Structured JSON, or
B) Free-text fields.

If JSON is provided, it will generally follow this schema (fields may be missing):
{
  "patient": {
    "name": "...",
    "mrn": "...",
    "age": "...",
    "gender": "..."
  },
  "exam": {
    "date": "DD/MM/YYYY",
    "referring_physician": "...",
    "clinical_indication": "...",
    "modality": "...",
    "technique": "...",
    "contrast_used": "Yes/No",
    "image_quality": "Good/Fair/Limited",
    "comparison": "None or prior study dated DD/MM/YYYY"
  },
  "ai": {
    "model_name": "...",
    "model_version": "...",
    "analysis_timestamp": "...",
    "overall_confidence_score": 0.00,
    "detections": [
      {
        "label": "Condition name",
        "probability_percent": 0-100,
        "confidence_level": "Low/Moderate/High",
        "location": "...",
        "size": "...",
        "supporting_findings": ["...","..."]
      }
    ],
    "quantitative_measurements": {
      "lesion_size": "...",
      "affected_lung_percentage": "...",
      "density_hu": "...",
      "severity_score": "Mild/Moderate/Severe"
    }
  },
  "findings": {
    "lungs": "...",
    "pleura": "...",
    "mediastinum_and_hila": "...",
    "heart_and_great_vessels": "...",
    "chest_wall_bones": "...",
    "upper_abdomen": "..."
  }
}

If the input is free-text, extract what you can; do not assume missing data.

OUTPUT REQUIREMENTS (STRICT)
- Output ONLY the report content (no extra explanations, no hidden reasoning).
- Use the exact section order, numbering, and headings shown below.
- Keep bracketed placeholders when information is missing (e.g., [Patient Name], [Not provided]).
- Keep content concise but complete. Use short paragraphs or bullet-like sentences within each organ system as needed.
- Ensure internal consistency (modality vs. technique vs. findings vs. measurements).

STYLE
- Professional radiology style.
- Prefer clear positives/negatives (e.g., “No pleural effusion.” “No pneumothorax.”) when supported or when explicitly stated as negative in the input.
- If negatives are not provided and cannot be inferred, write “[Not provided]” or “Unable to assess.”

NOW GENERATE THIS REPORT FORMAT EXACTLY:

🧾 AI-Generated Radiology Report
(Preliminary – AI Assisted)

1️⃣ Patient Information
Patient Name: [Patient Name]
Medical Record Number: [MRN]
Age / Gender: [Age] / [Gender]
Date of Examination: [DD/MM/YYYY]
Referring Physician: [Physician Name]
Clinical Indication: [Reason for exam / Symptoms / Relevant history]

2️⃣ Examination Details
Modality: [Chest X-ray / CT Chest / HRCT / Contrast CT]
Technique: [Imaging protocol description]
Contrast Used: [Yes / No]
Image Quality: [Good / Fair / Limited]
Comparison: [None / Prior study dated DD/MM/YYYY]

3️⃣ AI Analysis Summary
AI Model Name: [Model Name]
Model Version: [Version Number]
Analysis Date/Time: [Timestamp]
Overall Confidence Score: [0.00 – 1.00]

4️⃣ Findings
Lungs:
[Description of lung parenchyma findings – opacities, nodules, consolidation, interstitial changes, etc.]

Pleura:
[Presence or absence of pleural effusion, thickening, pneumothorax]

Mediastinum and Hila:
[Lymph nodes, masses, vascular structures]

Heart and Great Vessels:
[Cardiac size and configuration]

Chest Wall / Bones:
[Fractures, lesions, abnormalities]

Upper Abdomen (if visible):
[Incidental findings if applicable]

5️⃣ AI Detected Abnormalities
Primary Suspicion: [Condition Name]
Estimated Probability: [XX% or Not provided]
Confidence Level: [Low / Moderate / High or Not provided]
Location: [Anatomical region]
Estimated Size (if applicable): [Measurement or Not provided]

Secondary Considerations:
[Condition 1 – XX% or Not provided]
[Condition 2 – XX% or Not provided]

No Significant Abnormalities Detected: [If applicable / Yes/No with brief note]

6️⃣ Quantitative Measurements (If Applicable)
Lesion Size: [Measurement in cm/mm or Not provided]
Affected Lung Percentage: [XX% or Not provided]
Density (HU for CT): [Value or Not provided]
Severity Score: [Mild / Moderate / Severe or Not provided]

7️⃣ Impression
[Concise summary of the most likely diagnosis or key findings based strictly on provided inputs. Include limitations if relevant. Use cautious language.]

8️⃣ Recommendations
[Clinical correlation recommended]
[Laboratory tests if applicable]
[Follow-up imaging suggestion]
[Specialist referral if necessary]
[If critical concern is present in input, recommend urgent communication.]

9️⃣ AI Disclaimer
This report was automatically generated using an artificial intelligence system for radiological analysis.
It is intended to assist clinical decision-making and must be reviewed and validated by a qualified healthcare professional before any medical decisions are made.
`;

function privacyPolicyBlocked(status: number, message?: string | null) {
  if (status !== 404) {
    return false;
  }
  if (!message) {
    return false;
  }
  return message.includes("No endpoints found matching your data policy");
}

function formatDateDMY(isoDate?: string | null) {
  if (!isoDate) {
    return undefined;
  }
  const date = new Date(isoDate);
  if (Number.isNaN(date.getTime())) {
    return undefined;
  }
  const day = String(date.getUTCDate()).padStart(2, "0");
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const year = String(date.getUTCFullYear());
  return `${day}/${month}/${year}`;
}

function toPercentage(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return undefined;
  }
  return Number((value * 100).toFixed(2));
}

function toConfidenceLabel(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return undefined;
  }
  const normalized = value > 1 ? value / 100 : value;
  if (normalized >= 0.75) {
    return "High";
  }
  if (normalized >= 0.4) {
    return "Moderate";
  }
  return "Low";
}

function cleanModelOutput(text: string) {
  const trimmed = text.trim();
  if (!trimmed.startsWith("```")) {
    return trimmed;
  }
  return trimmed
    .replace(/^```[a-zA-Z]*\n?/, "")
    .replace(/\n```$/, "")
    .trim();
}

function extractImpression(report: string) {
  const match = report.match(
    /7️⃣ Impression\s*([\s\S]*?)(?:\n\s*8️⃣ Recommendations|$)/i
  );
  if (match?.[1]?.trim()) {
    return match[1].trim();
  }
  return "";
}

function ensureDisclaimerSection(report: string) {
  if (report.includes("9️⃣ AI Disclaimer")) {
    return report;
  }
  return `${report.trim()}\n\n${AI_DISCLAIMER_SECTION}\n`;
}

function inferModalityLabel(modality: CaseRecord["modality"]) {
  if (modality === "X-ray") {
    return "Chest X-ray";
  }
  if (modality === "CT") {
    return "CT Chest";
  }
  return modality;
}

function buildModelInput(args: {
  paramsId: string;
  caseRecord: CaseRecord;
  results: InferenceResult;
  question?: string;
}) {
  const { paramsId, caseRecord, results, question } = args;

  const detections =
    results.findings?.map((finding) => ({
      label: finding.label,
      probability_percent: toPercentage(finding.probability),
      confidence_level: toConfidenceLabel(finding.confidence),
      location: "[Not provided]",
      size: "[Not provided]",
      supporting_findings: results.summary ? [results.summary] : []
    })) ?? [];

  const inputPayload = {
    case_id: paramsId,
    patient: {
      name: caseRecord.patientName,
      mrn: caseRecord.patientId,
      age: "[Not provided]",
      gender: "[Not provided]"
    },
    exam: {
      date: formatDateDMY(caseRecord.lastRunAt ?? caseRecord.createdAt),
      referring_physician: "[Not provided]",
      clinical_indication: caseRecord.notes ?? "[Not provided]",
      modality: inferModalityLabel(caseRecord.modality),
      technique: "[Not provided]",
      contrast_used: "[Not provided]",
      image_quality:
        results.explainability?.warning || results.explainability?.error
          ? "Limited"
          : "[Not provided]",
      comparison: "[Not provided]"
    },
    ai: {
      model_name: results.model?.name ?? results.modelInfo?.name ?? "[Not provided]",
      model_version:
        results.model?.version ?? results.modelInfo?.version ?? "[Not provided]",
      analysis_timestamp: results.updatedAt ?? results.runAt ?? "[Not provided]",
      overall_confidence_score: results.prediction?.probability,
      detections,
      quantitative_measurements: {
        lesion_size: "[Not provided]",
        affected_lung_percentage: "[Not provided]",
        density_hu: "[Not provided]",
        severity_score: "[Not provided]"
      }
    },
    findings: {
      lungs: results.findings?.length
        ? results.findings
            .map((finding) => {
              const probability = toPercentage(finding.probability);
              if (probability === undefined) {
                return finding.label;
              }
              return `${finding.label} (${probability}%)`;
            })
            .join("; ")
        : "[Not provided]",
      pleura: "[Not provided]",
      mediastinum_and_hila: "[Not provided]",
      heart_and_great_vessels: "[Not provided]",
      chest_wall_bones: "[Not provided]",
      upper_abdomen: "[Not provided]"
    },
    limitations: [
      ...(results.explainability?.warning ? [results.explainability.warning] : []),
      ...(results.explainability?.error ? [results.explainability.error] : []),
      ...(results.modelInfo?.limitations ?? [])
    ],
    clinician_question: question ?? undefined
  };

  return [
    "Generate the report from the following provided JSON only.",
    "Do not add details that are not in the JSON.",
    "",
    JSON.stringify(inputPayload, null, 2)
  ].join("\n");
}

export async function POST(
  request: Request,
  { params }: { params: { id: string } }
) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      {
        error:
          "OPENROUTER_API_KEY is not configured. Set it in frontend/.env.local or env vars."
      },
      { status: 503 }
    );
  }

  const caseRecord = await getCaseById(params.id);
  if (!caseRecord) {
    return NextResponse.json({ error: "Case not found." }, { status: 404 });
  }

  const results = await readResults(params.id);
  if (!results) {
    return NextResponse.json(
      { error: "Run inference before requesting a summary." },
      { status: 400 }
    );
  }

  const body = (await request.json().catch(() => ({}))) as {
    question?: string;
  };
  const question = body.question?.trim() || undefined;

  const prompt = buildModelInput({
    paramsId: params.id,
    caseRecord,
    results,
    question
  });

  try {
    const messages = [
      { role: "system" as const, content: RADIOLOGY_SYSTEM_PROMPT },
      { role: "user" as const, content: prompt }
    ];

    const primary = await callOpenRouterChat({
      messages,
      model: process.env.OPENROUTER_MODEL ?? MODEL_NAME
    });

    let responsePayload = primary.data;
    let responseWarning: string | null = null;
    let selectedModel = process.env.OPENROUTER_MODEL ?? MODEL_NAME;

    if (
      !primary.ok &&
      privacyPolicyBlocked(primary.status, primary.errorMessage)
    ) {
      responsePayload = null;
      const fallbackModels = [FALLBACK_MODEL, FALLBACK_MODEL_2];
      for (const fallback of fallbackModels) {
        if (!fallback) {
          continue;
        }
        const fallbackResponse = await callOpenRouterChat({
          messages,
          model: fallback
        });
        if (fallbackResponse.ok) {
          responsePayload = fallbackResponse.data;
          selectedModel = fallback;
          responseWarning =
            "Primary model tngtech/deepseek-r1t2-chimera:free blocked by OpenRouter privacy settings. Used fallback " +
            `${fallback}. Configure: https://openrouter.ai/settings/privacy`;
          break;
        }
      }
    }

    if (!responsePayload) {
      if (privacyPolicyBlocked(primary.status, primary.errorMessage)) {
        return NextResponse.json(
          {
            error: "OpenRouter model blocked by privacy settings.",
            help: "Configure at https://openrouter.ai/settings/privacy"
          },
          { status: 503 }
        );
      }
      throw new Error(primary.errorMessage || "OpenRouter request failed.");
    }

    const raw = extractOpenRouterText(responsePayload);
    if (!raw.trim()) {
      throw new Error("OpenRouter returned an empty response.");
    }

    const reportMarkdown = ensureDisclaimerSection(cleanModelOutput(raw));
    const impression = extractImpression(reportMarkdown);
    const assistantSummary =
      impression ||
      (question
        ? "Report updated based on your follow-up question."
        : "Preliminary AI-assisted radiology report generated.");

    const summaryRecord: SummaryRecord = {
      assistant_summary: assistantSummary,
      report_markdown: reportMarkdown,
      generated_at: new Date().toISOString(),
      model: selectedModel,
      case_id: params.id
    };

    await saveSummary(params.id, summaryRecord);

    const reportPath = path.join(
      process.cwd(),
      "storage",
      params.id,
      "report.md"
    );
    await fs.writeFile(reportPath, reportMarkdown, "utf-8");

    if (question) {
      const chat = await readChat(params.id);
      const now = new Date().toISOString();
      const updated: ChatMessage[] = [
        ...chat,
        { role: "user", content: question, timestamp: now },
        { role: "assistant", content: assistantSummary, timestamp: now }
      ];
      await saveChat(params.id, updated);
    }

    return NextResponse.json({
      assistant_summary: assistantSummary,
      report_markdown: reportMarkdown,
      warning: responseWarning ?? undefined
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Summary generation failed.";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
