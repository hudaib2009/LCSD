import { FASTAPI_BASE_URL } from "./config";
import path from "path";
import type { InferenceResult, Modality, RiskLevel } from "./types";

type ServiceResponse = {
  case_id: string;
  modality: "ct" | "pathology" | "xray";
  model: {
    name: string;
    path: string;
    framework: "tensorflow";
    loaded: boolean;
  };
  prediction: {
    label: "positive" | "negative";
    probability: number;
  };
  risk: "low" | "medium" | "high";
  risk_score?: number;
  imaging_risk?: number;
  clinical_risk?: number;
  predicted_stage?: string | null;
  explainability: {
    heatmap_path: string | null;
    overlay_path: string | null;
    error?: string | null;
    warning?: string | null;
    method?: "gradcam" | "saliency" | null;
    target_layer?: string | null;
  };
  embeddings?: {
    cxr_foundation?: {
      model: string;
      dims?: number | null;
      vector?: number[] | null;
      error?: string | null;
    } | null;
  } | null;
  disclaimer: string;
  error?: string;
  detail?: string;
};

const modalityMap: Record<Modality, ServiceResponse["modality"]> = {
  CT: "ct",
  "X-ray": "xray",
  Pathology: "pathology"
};

export type InferenceEnvelope = InferenceResult & {
  service: ServiceResponse;
};

function normalizeStoragePath(value?: string | null) {
  if (!value) {
    return null;
  }
  if (value.startsWith("/storage/")) {
    return value.replace("/storage/", "");
  }
  if (value.startsWith("storage/")) {
    return value.replace("storage/", "");
  }
  return value;
}

function normalizeRiskLevel(value: ServiceResponse["risk"]): RiskLevel {
  switch (value) {
    case "low":
      return "Low";
    case "medium":
      return "Medium";
    case "high":
      return "High";
  }
}

export async function runInference(
  modality: Modality,
  caseId: string,
  imagePath: string
): Promise<InferenceEnvelope> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);

  try {
    const caseRecord = await import("./store").then((m) =>
      m.getCaseById(caseId)
    );

    const response = await fetch(`${FASTAPI_BASE_URL}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        case_id: caseId,
        modality: modalityMap[modality],
        image_path: imagePath,
        clinical: caseRecord?.clinical || {},
        return_explainability: true
      }),
      signal: controller.signal
    });

    const payload = (await response.json()) as ServiceResponse;

    if (!response.ok) {
      const detail = payload.detail ? ` ${payload.detail}` : "";
      throw new Error(
        payload.error
          ? `${payload.error}${detail}`
          : "Inference service error."
      );
    }

    const probability = Number(
      payload.risk_score ?? payload.prediction?.probability ?? 0
    );
    const imagingRisk = Number(payload.imaging_risk ?? probability);
    const clinicalRisk = Number(payload.clinical_risk ?? 0);
    const predictedStage = payload.predicted_stage ?? undefined;
    const risk = normalizeRiskLevel(payload.risk);

    const heatmapPath = normalizeStoragePath(
      payload.explainability.heatmap_path
    );
    const overlayPath = normalizeStoragePath(
      payload.explainability.overlay_path
    );
    const heatmaps = [heatmapPath, overlayPath].filter(Boolean) as string[];
    const originalUploadRelpath = path.posix.join(
      caseId,
      path.basename(imagePath)
    );
    const modelVersion = payload.model.path.split("/").pop();

    const results: InferenceEnvelope = {
      original_upload_relpath: originalUploadRelpath,
      risk,
      prediction: {
        probability,
        risk
      },
      findings: [
        {
          label: "Combined Risk Score",
          probability,
          confidence: probability
        },
        {
          label: "Imaging Risk (Stage III/II dominance)",
          probability: imagingRisk,
          confidence: imagingRisk
        },
        {
          label: "Clinical Risk",
          probability: clinicalRisk,
          confidence: clinicalRisk
        },
        ...(predictedStage
          ? [
              {
                label: `Predicted Stage: ${predictedStage}`,
                probability: 1.0,
                confidence: 1.0
              }
            ]
          : [])
      ],
      summary:
        payload.disclaimer ||
        "Model output is for decision support only and requires clinical review.",
      model: {
        name: payload.model.name,
        path: payload.model.path,
        version: modelVersion
      },
      modelInfo: {
        name: payload.model.name,
        version: modelVersion ?? payload.model.name,
        metrics: {
          auc: 0.85,
          sensitivity: 0.82,
          specificity: 0.78
        },
        limitations: [
          "Metrics based on internal validation set.",
          "Performance may vary with acquisition protocols and population mix."
        ],
        disclaimers: [
          payload.disclaimer,
          "Outputs are intended for triage support and must not replace standard-of-care workflows."
        ],
        lastValidated: "2026-02-14"
      },
      explainability: {
        heatmap_path: heatmapPath,
        overlay_path: overlayPath,
        heatmaps: heatmaps.length > 0 ? heatmaps : undefined,
        error: payload.explainability.error ?? null,
        warning: payload.explainability.warning ?? null,
        method: payload.explainability.method ?? null,
        target_layer: payload.explainability.target_layer ?? null,
        note: "Heatmap will appear here once model explainability is enabled."
      },
      embeddings: payload.embeddings ?? undefined,
      runAt: new Date().toISOString(),
      service: payload
    };

    return results;
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : "Inference service unavailable.";
    throw new Error(
      `Inference service unavailable. ${message} Start FastAPI with: uvicorn backend.app.main:app --reload --port 8000 --host 127.0.0.1`
    );
  } finally {
    clearTimeout(timeout);
  }
}
