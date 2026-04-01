"use client";

import { useState } from "react";

type XrayResult = {
  abnormal_score: number;
  pred_label: "Normal" | "Abnormal";
  heatmap?: string | null;
  heatmap_method?: string | null;
  heatmap_warning?: string | null;
  heatmap_error?: string | null;
};

type CtResult = {
  ct_prob_mean: number;
  ct_pred: 0 | 1;
  positive_slices: number;
  total_slices: number;
  ratio_positive: number;
  extent_score: number;
  stage_proxy: "I" | "II" | "III" | "IV";
  threshold: number;
};

type PathologyResult = {
  subtype: string;
  confidence: number;
  heatmap?: string | null;
  heatmap_method?: string | null;
  heatmap_warning?: string | null;
  heatmap_error?: string | null;
};

type PlanResult = {
  risk_tier: "Low" | "Medium" | "High";
  final_risk_score: number;
  clinical_score?: number;
  imaging_risk?: number;
  stage_proxy?: string;
  path_subtype?: string;
  summary: string;
  next_steps: string[];
  warnings: string[];
};

const FASTAPI_BASE_URL =
  process.env.NEXT_PUBLIC_FASTAPI_BASE_URL ?? "http://127.0.0.1:8000";

function formatScore(value?: number | null) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(3);
}

export default function DemoPage() {
  const [xrayFile, setXrayFile] = useState<File | null>(null);
  const [xrayResult, setXrayResult] = useState<XrayResult | null>(null);
  const [xrayLoading, setXrayLoading] = useState(false);
  const [xrayError, setXrayError] = useState<string | null>(null);

  const [ctFiles, setCtFiles] = useState<File[]>([]);
  const [ctResult, setCtResult] = useState<CtResult | null>(null);
  const [ctLoading, setCtLoading] = useState(false);
  const [ctError, setCtError] = useState<string | null>(null);

  const [pathFile, setPathFile] = useState<File | null>(null);
  const [pathResult, setPathResult] = useState<PathologyResult | null>(null);
  const [pathLoading, setPathLoading] = useState(false);
  const [pathError, setPathError] = useState<string | null>(null);

  const [stageProxy, setStageProxy] = useState<"I" | "II" | "III" | "IV">("I");
  const [subtype, setSubtype] = useState("Unknown");

  const [clinical, setClinical] = useState({
    age: "",
    sex: "F",
    smoker: false,
    packYears: "",
    ecog: "0",
    weightLoss: false
  });

  const [planResult, setPlanResult] = useState<PlanResult | null>(null);
  const [planLoading, setPlanLoading] = useState(false);
  const [planError, setPlanError] = useState<string | null>(null);

  async function runXray() {
    if (!xrayFile) {
      setXrayError("Select an X-ray image first.");
      return;
    }
    setXrayLoading(true);
    setXrayError(null);
    try {
      const form = new FormData();
      form.append("file", xrayFile);
      const response = await fetch(`${FASTAPI_BASE_URL}/predict/xray`, {
        method: "POST",
        body: form
      });
      const payload = (await response.json()) as XrayResult & {
        error?: string;
        detail?: string;
      };
      if (!response.ok) {
        throw new Error(payload.error ?? "X-ray inference failed.");
      }
      setXrayResult(payload);
    } catch (err) {
      setXrayError(err instanceof Error ? err.message : "X-ray inference failed.");
    } finally {
      setXrayLoading(false);
    }
  }

  async function runCt() {
    if (ctFiles.length === 0) {
      setCtError("Select CT slices first.");
      return;
    }
    setCtLoading(true);
    setCtError(null);
    try {
      const form = new FormData();
      ctFiles.forEach((file) => form.append("files", file));
      const response = await fetch(`${FASTAPI_BASE_URL}/predict/ct`, {
        method: "POST",
        body: form
      });
      const payload = (await response.json()) as CtResult & {
        error?: string;
        detail?: string;
      };
      if (!response.ok) {
        throw new Error(payload.error ?? "CT inference failed.");
      }
      setCtResult(payload);
      setStageProxy(payload.stage_proxy);
    } catch (err) {
      setCtError(err instanceof Error ? err.message : "CT inference failed.");
    } finally {
      setCtLoading(false);
    }
  }

  async function runPathology() {
    if (!pathFile) {
      setPathError("Select a pathology image first.");
      return;
    }
    setPathLoading(true);
    setPathError(null);
    try {
      const form = new FormData();
      form.append("file", pathFile);
      const response = await fetch(`${FASTAPI_BASE_URL}/predict/pathology`, {
        method: "POST",
        body: form
      });
      const payload = (await response.json()) as PathologyResult & {
        error?: string;
        detail?: string;
      };
      if (!response.ok) {
        throw new Error(payload.error ?? "Pathology inference failed.");
      }
      setPathResult(payload);
      setSubtype(payload.subtype ?? "Unknown");
    } catch (err) {
      setPathError(
        err instanceof Error ? err.message : "Pathology inference failed."
      );
    } finally {
      setPathLoading(false);
    }
  }

  async function runPlan() {
    setPlanLoading(true);
    setPlanError(null);
    try {
      if (!clinical.age) {
        setPlanError("Enter patient age before generating a plan.");
        setPlanLoading(false);
        return;
      }
      if (clinical.smoker && !clinical.packYears) {
        setPlanError("Enter pack-years for smokers.");
        setPlanLoading(false);
        return;
      }
      const payload = {
        clinical: {
          age: Number(clinical.age),
          sex: clinical.sex,
          smoker: clinical.smoker,
          pack_years: clinical.smoker ? Number(clinical.packYears || 0) : 0,
          ecog: Number(clinical.ecog),
          weight_loss: clinical.weightLoss
        },
        imaging: {
          ct_prob_mean: ctResult?.ct_prob_mean ?? null,
          ratio_positive: ctResult?.ratio_positive ?? null,
          stage_proxy: stageProxy,
          path_subtype: subtype,
          path_confidence: pathResult?.confidence ?? null,
          xray_abnormal_score: xrayResult?.abnormal_score ?? null
        }
      };

      const response = await fetch(`${FASTAPI_BASE_URL}/plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = (await response.json()) as PlanResult & {
        error?: string;
        detail?: string;
      };
      if (!response.ok) {
        throw new Error(result.error ?? "Plan generation failed.");
      }
      setPlanResult(result);
    } catch (err) {
      setPlanError(
        err instanceof Error ? err.message : "Plan generation failed."
      );
    } finally {
      setPlanLoading(false);
    }
  }

  return (
    <div className="px-6 py-10 lg:px-12">
      <header className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-slate">
            ISEF demo
          </p>
          <h1 className="mt-3 font-[var(--font-fraunces)] text-4xl text-ink">
            Lung screening decision support
          </h1>
          <p className="mt-3 max-w-2xl text-sm text-slate">
            Upload imaging inputs, add clinical context, and generate a rule-based
            treatment plan. Outputs are educational decision support only.
          </p>
        </div>
        <p className="max-w-sm text-xs text-slate">
          The public repo does not bundle sample medical studies. Upload local test
          files to exercise the demo flow.
        </p>
      </header>

      <section className="mt-8 grid gap-6">
        <div className="rounded-2xl border border-slate/10 bg-white/80 p-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h2 className="font-[var(--font-fraunces)] text-2xl text-ink">
                1) Chest X-ray screening
              </h2>
              <p className="mt-2 text-sm text-slate">
                EfficientNetB0 abnormal/normal screening with Grad-CAM overlay.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <input
                type="file"
                accept="image/*"
                onChange={(event) =>
                  setXrayFile(event.target.files?.[0] ?? null)
                }
                className="text-sm"
              />
              <button
                type="button"
                onClick={runXray}
                className="rounded-full bg-ink px-4 py-2 text-sm font-semibold text-white"
              >
                {xrayLoading ? "Running..." : "Run X-ray"}
              </button>
            </div>
          </div>
          {xrayError ? (
            <p className="mt-3 text-sm text-rose-600">{xrayError}</p>
          ) : null}
          {xrayResult ? (
            <div className="mt-5 grid gap-4 lg:grid-cols-[1.2fr_1fr]">
              <div className="rounded-xl border border-slate/10 bg-white p-4">
                <p className="text-xs uppercase text-slate">Score</p>
                <p className="mt-2 text-2xl font-semibold text-ink">
                  {formatScore(xrayResult.abnormal_score)} ({xrayResult.pred_label})
                </p>
                <p className="mt-2 text-xs text-slate">
                  Grad-CAM method: {xrayResult.heatmap_method ?? "n/a"}
                </p>
                {xrayResult.heatmap_warning ? (
                  <p className="mt-2 text-xs text-amber-600">
                    {xrayResult.heatmap_warning}
                  </p>
                ) : null}
                {xrayResult.heatmap_error ? (
                  <p className="mt-2 text-xs text-rose-600">
                    {xrayResult.heatmap_error}
                  </p>
                ) : null}
              </div>
              <div className="rounded-xl border border-slate/10 bg-white p-4">
                {xrayResult.heatmap ? (
                  <img
                    src={xrayResult.heatmap}
                    alt="X-ray heatmap"
                    className="h-56 w-full rounded-lg object-contain"
                  />
                ) : (
                  <div className="flex h-56 items-center justify-center rounded-lg bg-mist text-xs text-slate">
                    Heatmap will appear after inference.
                  </div>
                )}
              </div>
            </div>
          ) : null}
        </div>

        <div className="rounded-2xl border border-slate/10 bg-white/80 p-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h2 className="font-[var(--font-fraunces)] text-2xl text-ink">
                2) CT detection + extent staging proxy
              </h2>
              <p className="mt-2 text-sm text-slate">
                Upload multiple axial slices to compute slice probabilities and
                extent-based stage proxy.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <input
                type="file"
                accept="image/*"
                multiple
                onChange={(event) =>
                  setCtFiles(Array.from(event.target.files ?? []))
                }
                className="text-sm"
              />
              <button
                type="button"
                onClick={runCt}
                className="rounded-full bg-ink px-4 py-2 text-sm font-semibold text-white"
              >
                {ctLoading ? "Running..." : "Run CT"}
              </button>
            </div>
          </div>
          {ctError ? (
            <p className="mt-3 text-sm text-rose-600">{ctError}</p>
          ) : null}
          {ctResult ? (
            <div className="mt-5 grid gap-4 md:grid-cols-2">
              <div className="rounded-xl border border-slate/10 bg-white p-4">
                <p className="text-xs uppercase text-slate">Extent score</p>
                <p className="mt-2 text-2xl font-semibold text-ink">
                  {formatScore(ctResult.extent_score)}
                </p>
                <p className="mt-2 text-xs text-slate">
                  Positive slices: {ctResult.positive_slices} /{" "}
                  {ctResult.total_slices} (ratio {formatScore(ctResult.ratio_positive)})
                </p>
              </div>
              <div className="rounded-xl border border-slate/10 bg-white p-4">
                <p className="text-xs uppercase text-slate">Stage proxy</p>
                <p className="mt-2 text-2xl font-semibold text-ink">
                  {ctResult.stage_proxy}
                </p>
                <p className="mt-2 text-xs text-slate">
                  Threshold: {formatScore(ctResult.threshold)}
                </p>
              </div>
            </div>
          ) : null}
        </div>

        <div className="rounded-2xl border border-slate/10 bg-white/80 p-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h2 className="font-[var(--font-fraunces)] text-2xl text-ink">
                3) Pathology subtype
              </h2>
              <p className="mt-2 text-sm text-slate">
                Classify ACA vs SCC vs Normal with optional heatmap overlay.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <input
                type="file"
                accept="image/*"
                onChange={(event) =>
                  setPathFile(event.target.files?.[0] ?? null)
                }
                className="text-sm"
              />
              <button
                type="button"
                onClick={runPathology}
                className="rounded-full bg-ink px-4 py-2 text-sm font-semibold text-white"
              >
                {pathLoading ? "Running..." : "Run pathology"}
              </button>
            </div>
          </div>
          {pathError ? (
            <p className="mt-3 text-sm text-rose-600">{pathError}</p>
          ) : null}
          {pathResult ? (
            <div className="mt-5 grid gap-4 lg:grid-cols-[1.2fr_1fr]">
              <div className="rounded-xl border border-slate/10 bg-white p-4">
                <p className="text-xs uppercase text-slate">Subtype</p>
                <p className="mt-2 text-2xl font-semibold text-ink">
                  {pathResult.subtype}
                </p>
                <p className="mt-2 text-xs text-slate">
                  Confidence: {formatScore(pathResult.confidence)}
                </p>
                {pathResult.heatmap_warning ? (
                  <p className="mt-2 text-xs text-amber-600">
                    {pathResult.heatmap_warning}
                  </p>
                ) : null}
                {pathResult.heatmap_error ? (
                  <p className="mt-2 text-xs text-rose-600">
                    {pathResult.heatmap_error}
                  </p>
                ) : null}
              </div>
              <div className="rounded-xl border border-slate/10 bg-white p-4">
                {pathResult.heatmap ? (
                  <img
                    src={pathResult.heatmap}
                    alt="Pathology heatmap"
                    className="h-56 w-full rounded-lg object-contain"
                  />
                ) : (
                  <div className="flex h-56 items-center justify-center rounded-lg bg-mist text-xs text-slate">
                    Heatmap will appear after inference.
                  </div>
                )}
              </div>
            </div>
          ) : null}
        </div>

        <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
          <div className="rounded-2xl border border-slate/10 bg-white/80 p-6">
            <h2 className="font-[var(--font-fraunces)] text-2xl text-ink">
              4) Clinical inputs
            </h2>
            <p className="mt-2 text-sm text-slate">
              Lightweight clinical features used for rule-based risk fusion.
            </p>
            <div className="mt-5 grid gap-4 md:grid-cols-2">
              <label className="text-sm text-slate">
                Age
                <input
                  type="number"
                  value={clinical.age}
                  onChange={(event) =>
                    setClinical({ ...clinical, age: event.target.value })
                  }
                  className="mt-2 w-full rounded-lg border border-slate/20 bg-white px-3 py-2 text-ink"
                />
              </label>
              <label className="text-sm text-slate">
                Sex
                <select
                  value={clinical.sex}
                  onChange={(event) =>
                    setClinical({ ...clinical, sex: event.target.value })
                  }
                  className="mt-2 w-full rounded-lg border border-slate/20 bg-white px-3 py-2 text-ink"
                >
                  <option value="F">Female</option>
                  <option value="M">Male</option>
                </select>
              </label>
              <label className="text-sm text-slate">
                Pack-years
                <input
                  type="number"
                  value={clinical.packYears}
                  onChange={(event) =>
                    setClinical({ ...clinical, packYears: event.target.value })
                  }
                  className="mt-2 w-full rounded-lg border border-slate/20 bg-white px-3 py-2 text-ink"
                />
              </label>
              <label className="text-sm text-slate">
                ECOG (0-4)
                <select
                  value={clinical.ecog}
                  onChange={(event) =>
                    setClinical({ ...clinical, ecog: event.target.value })
                  }
                  className="mt-2 w-full rounded-lg border border-slate/20 bg-white px-3 py-2 text-ink"
                >
                  <option value="0">0</option>
                  <option value="1">1</option>
                  <option value="2">2</option>
                  <option value="3">3</option>
                  <option value="4">4</option>
                </select>
              </label>
              <label className="flex items-center gap-2 text-sm text-slate">
                <input
                  type="checkbox"
                  checked={clinical.smoker}
                  onChange={(event) =>
                    setClinical({ ...clinical, smoker: event.target.checked })
                  }
                />
                Current/former smoker
              </label>
              <label className="flex items-center gap-2 text-sm text-slate">
                <input
                  type="checkbox"
                  checked={clinical.weightLoss}
                  onChange={(event) =>
                    setClinical({
                      ...clinical,
                      weightLoss: event.target.checked
                    })
                  }
                />
                Unintentional weight loss
              </label>
            </div>
          </div>

          <div className="rounded-2xl border border-slate/10 bg-white/80 p-6">
            <h2 className="font-[var(--font-fraunces)] text-2xl text-ink">
              5) Plan generation
            </h2>
            <p className="mt-2 text-sm text-slate">
              Fuse imaging + clinical signals into a risk tier and treatment plan.
            </p>
            <div className="mt-5 grid gap-4">
              <label className="text-sm text-slate">
                Stage proxy
                <select
                  value={stageProxy}
                  onChange={(event) =>
                    setStageProxy(event.target.value as "I" | "II" | "III" | "IV")
                  }
                  className="mt-2 w-full rounded-lg border border-slate/20 bg-white px-3 py-2 text-ink"
                >
                  <option value="I">I</option>
                  <option value="II">II</option>
                  <option value="III">III</option>
                  <option value="IV">IV</option>
                </select>
              </label>
              <label className="text-sm text-slate">
                Pathology subtype
                <select
                  value={subtype}
                  onChange={(event) => setSubtype(event.target.value)}
                  className="mt-2 w-full rounded-lg border border-slate/20 bg-white px-3 py-2 text-ink"
                >
                  <option value="ACA">ACA</option>
                  <option value="SCC">SCC</option>
                  <option value="Normal">Normal</option>
                  <option value="Unknown">Unknown</option>
                </select>
              </label>
              <button
                type="button"
                onClick={runPlan}
                className="rounded-full bg-ink px-4 py-2 text-sm font-semibold text-white"
              >
                {planLoading ? "Generating..." : "Generate plan"}
              </button>
            </div>
            {planError ? (
              <p className="mt-3 text-sm text-rose-600">{planError}</p>
            ) : null}
            {planResult ? (
              <div className="mt-5 rounded-xl border border-slate/10 bg-white p-4">
                <p className="text-xs uppercase text-slate">Final risk tier</p>
                <p className="mt-2 text-2xl font-semibold text-ink">
                  {planResult.risk_tier} ({formatScore(planResult.final_risk_score)})
                </p>
                <p className="mt-2 text-sm text-slate">{planResult.summary}</p>
                <div className="mt-3 grid gap-2 text-xs text-slate">
                  <span>Imaging score: {formatScore(planResult.imaging_risk)}</span>
                  <span>Clinical score: {formatScore(planResult.clinical_score)}</span>
                </div>
                <div className="mt-4">
                  <p className="text-xs uppercase text-slate">Next steps</p>
                  <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate">
                    {planResult.next_steps.map((step) => (
                      <li key={step}>{step}</li>
                    ))}
                  </ul>
                </div>
                <div className="mt-4">
                  <p className="text-xs uppercase text-slate">Warnings</p>
                  <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-amber-700">
                    {planResult.warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </section>

      <section className="mt-10 grid gap-4 rounded-2xl border border-slate/10 bg-white/70 p-6 text-sm text-slate">
        <h3 className="font-[var(--font-fraunces)] text-xl text-ink">
          Current demo inputs
        </h3>
        <div className="grid gap-2">
          <span>
            X-ray: {xrayFile ? xrayFile.name : "None"} | Score{" "}
            {formatScore(xrayResult?.abnormal_score ?? null)}
          </span>
          <span>
            CT: {ctFiles.length > 0 ? `${ctFiles.length} slices` : "None"} | Stage proxy{" "}
            {ctResult?.stage_proxy ?? stageProxy}
          </span>
          <span>
            Pathology: {pathFile ? pathFile.name : "None"} | Subtype{" "}
            {pathResult?.subtype ?? subtype}
          </span>
        </div>
      </section>

      <div className="mt-8 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-800">
        Educational decision support only. Not for clinical diagnosis or medical
        decision making.
      </div>
    </div>
  );
}
