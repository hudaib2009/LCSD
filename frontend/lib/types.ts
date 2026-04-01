export type Modality = "CT" | "X-ray" | "Pathology";

export type RiskLevel = "Low" | "Medium" | "High";

export type Finding = {
  label: string;
  probability: number;
  confidence: number;
};

export type ModelInfo = {
  name: string;
  version: string;
  metrics: {
    auc: number;
    sensitivity: number;
    specificity: number;
  };
  limitations: string[];
  disclaimers: string[];
  lastValidated: string;
};

export type Explainability = {
  heatmaps?: string[];
  heatmap_path?: string | null;
  overlay_path?: string | null;
  error?: string | null;
  warning?: string | null;
  method?: "gradcam" | "saliency" | null;
  target_layer?: string | null;
  note?: string;
};

export type ModelDescriptor = {
  name: string;
  path: string;
  version?: string;
};

export type EmbeddingVector = {
  model: string;
  dims?: number | null;
  vector?: number[] | null;
  error?: string | null;
};

export type InferenceResult = {
  original_upload_relpath?: string;
  risk: RiskLevel;
  prediction?: {
    probability: number;
    risk?: RiskLevel;
  };
  findings: Finding[];
  summary: string;
  model?: ModelDescriptor;
  modelInfo: ModelInfo;
  explainability: Explainability;
  embeddings?: {
    cxr_foundation?: EmbeddingVector | null;
  };
  runAt: string;
  updatedAt?: string;
  service?: unknown;
  // New fields for NSCLC risk scoring
  imaging_risk?: number;
  clinical_risk?: number;
  risk_score?: number;
  predicted_stage?: string;
};

export type SummaryRecord = {
  assistant_summary: string;
  report_markdown: string;
  generated_at: string;
  model: string;
  case_id: string;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
};

export type CaseRecord = {
  id: string;
  createdAt: string;
  modality: Modality;
  patientName?: string;
  patientId?: string;
  notes?: string;
  fileName: string;
  status: "pending" | "processing" | "complete";
  lastRunAt?: string;
  clinical?: {
    age?: number;
    isSmoker?: boolean;
    packYears?: number;
    histology?: string;
    ecog?: number;
  };
};
