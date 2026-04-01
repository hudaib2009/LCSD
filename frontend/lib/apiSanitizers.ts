import type { CaseRecord, InferenceResult } from "./types";

export function sanitizeCaseForApi(caseRecord: CaseRecord) {
  return {
    id: caseRecord.id,
    createdAt: caseRecord.createdAt,
    modality: caseRecord.modality,
    status: caseRecord.status,
    lastRunAt: caseRecord.lastRunAt ?? null,
  };
}

export function sanitizeResultsForApi(results: InferenceResult | null) {
  if (!results) {
    return null;
  }

  return {
    risk: results.risk,
    prediction: results.prediction ?? null,
    findings: results.findings,
    summary: results.summary,
    model:
      results.model?.name
        ? {
            name: results.model.name,
            version: results.model.version ?? null,
          }
        : null,
    modelInfo: results.modelInfo,
    explainability: results.explainability
      ? {
          error: results.explainability.error ?? null,
          warning: results.explainability.warning ?? null,
          method: results.explainability.method ?? null,
          target_layer: results.explainability.target_layer ?? null,
          note: results.explainability.note ?? null,
        }
      : null,
    runAt: results.runAt,
    updatedAt: results.updatedAt ?? null,
    imaging_risk: results.imaging_risk ?? null,
    clinical_risk: results.clinical_risk ?? null,
    risk_score: results.risk_score ?? null,
    predicted_stage: results.predicted_stage ?? null,
  };
}
