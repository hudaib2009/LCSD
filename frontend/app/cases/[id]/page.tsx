import Link from "next/link";
import { notFound } from "next/navigation";
import { FindingsTable } from "@/components/FindingsTable";
import { RiskBadge } from "@/components/RiskBadge";
import { RunInferenceButton } from "@/components/RunInferenceButton";
import { ExplainabilityPanel } from "@/components/ExplainabilityPanel";
import { SummaryAssistant } from "@/components/SummaryAssistant";
import { Tabs } from "@/components/Tabs";
import { getCaseById, readChat, readResults, readSummary } from "@/lib/store";
import path from "path";

export const dynamic = "force-dynamic";

export default async function CaseDetailPage({
  params
}: {
  params: { id: string };
}) {
  const caseRecord = await getCaseById(params.id);
  if (!caseRecord) {
    notFound();
  }

  const results = await readResults(params.id);
  const summary = await readSummary(params.id);
  const chat = await readChat(params.id);

  const uploadRelpath =
    results?.original_upload_relpath ??
    path.posix.join(caseRecord.id, caseRecord.fileName);
  const uploadSrc = `/api/files?path=${encodeURIComponent(uploadRelpath)}`;
  const isImage = /\.(png|jpe?g|gif|webp|tif|tiff)$/i.test(
    caseRecord.fileName
  );

  const uploadContent = (
    <div className="rounded-2xl border border-slate/10 bg-white/80 p-5">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate">
        Original upload
      </p>
      {isImage ? (
        <div className="mt-4 grid gap-4 md:grid-cols-[180px_1fr] md:items-start">
          <div className="rounded-xl border border-slate/10 bg-white p-3">
            <img
              src={uploadSrc}
              alt="Uploaded study thumbnail"
              className="h-32 w-full rounded-lg object-cover"
            />
            <p className="mt-2 text-xs text-slate">Thumbnail</p>
          </div>
          <div className="rounded-xl border border-slate/10 bg-white p-3">
            <img
              src={uploadSrc}
              alt="Uploaded study full view"
              className="max-h-[420px] w-full rounded-lg object-contain"
            />
            <p className="mt-2 text-xs text-slate">Full view</p>
          </div>
        </div>
      ) : (
        <p className="mt-3 text-sm text-slate">
          The uploaded file is not an image and cannot be previewed here.
        </p>
      )}
    </div>
  );

  const findingsContent = (
    <div className="grid gap-6">
      {uploadContent}
      {results ? (
        <>
          <div className="rounded-2xl border border-slate/10 bg-white/80 p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate">
              Risk overview
            </p>
            <div className="mt-3 flex items-center gap-4">
              <RiskBadge level={results.risk} />
              <p className="text-sm text-slate">{results.summary}</p>
            </div>
          </div>
          <FindingsTable findings={results.findings} />
        </>
      ) : (
        <div className="rounded-2xl border border-dashed border-slate/20 bg-white/70 p-6 text-sm text-slate">
          Run inference to generate findings and risk assessment.
        </div>
      )}
    </div>
  );

  const explainability = results?.explainability;
  const rawHeatmap =
    explainability?.heatmap_path ??
    (results?.service as { explainability?: { heatmap_path?: string | null } })
      ?.explainability?.heatmap_path;
  const rawOverlay =
    explainability?.overlay_path ??
    (results?.service as { explainability?: { overlay_path?: string | null } })
      ?.explainability?.overlay_path;

  const explainabilityContent = (
    <ExplainabilityPanel
      heatmapPath={rawHeatmap}
      overlayPath={rawOverlay}
      error={results?.explainability?.error ?? null}
      version={results?.updatedAt ?? results?.runAt ?? String(Date.now())}
    />
  );

  const summaryContent = results ? (
    <SummaryAssistant
      caseId={caseRecord.id}
      modality={caseRecord.modality}
      probability={results?.prediction?.probability}
      risk={results?.risk}
      initialSummary={summary?.assistant_summary ?? results?.summary}
      initialReport={summary?.report_markdown ?? null}
      initialChat={chat}
    />
  ) : (
    <div className="rounded-2xl border border-dashed border-slate/20 bg-white/70 p-6 text-sm text-slate">
      Run inference to enable the medical chatbot and report drafting.
    </div>
  );

  const modelInfo = results?.modelInfo;
  const modelContent = modelInfo ? (
    <div className="grid gap-4">
      <div className="rounded-2xl border border-slate/10 bg-white/80 p-6">
        <h3 className="font-[var(--font-fraunces)] text-xl text-ink">
          {modelInfo.name}
        </h3>
        <p className="mt-1 text-xs text-slate">Version {modelInfo.version}</p>
        <div className="mt-4 grid gap-3 text-sm text-slate sm:grid-cols-3">
          <div className="rounded-xl bg-mist px-4 py-3">
            <p className="text-xs uppercase text-slate">AUC</p>
            <p className="text-lg font-semibold text-ink">
              {modelInfo.metrics.auc.toFixed(2)}
            </p>
          </div>
          <div className="rounded-xl bg-mist px-4 py-3">
            <p className="text-xs uppercase text-slate">Sensitivity</p>
            <p className="text-lg font-semibold text-ink">
              {modelInfo.metrics.sensitivity.toFixed(2)}
            </p>
          </div>
          <div className="rounded-xl bg-mist px-4 py-3">
            <p className="text-xs uppercase text-slate">Specificity</p>
            <p className="text-lg font-semibold text-ink">
              {modelInfo.metrics.specificity.toFixed(2)}
            </p>
          </div>
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-slate/10 bg-white/80 p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate">
            Limitations
          </p>
          <ul className="mt-3 space-y-2 text-sm text-slate">
            {modelInfo.limitations.map((item) => (
              <li key={item}>• {item}</li>
            ))}
          </ul>
        </div>
        <div className="rounded-2xl border border-slate/10 bg-white/80 p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate">
            Disclaimers
          </p>
          <ul className="mt-3 space-y-2 text-sm text-slate">
            {modelInfo.disclaimers.map((item) => (
              <li key={item}>• {item}</li>
            ))}
          </ul>
          <p className="mt-3 text-xs text-slate">
            Last validated: {modelInfo.lastValidated}
          </p>
        </div>
      </div>
    </div>
  ) : (
    <div className="rounded-2xl border border-dashed border-slate/20 bg-white/70 p-6 text-sm text-slate">
      Model info will populate after inference.
    </div>
  );

  return (
    <div className="px-6 py-10 lg:px-12">
      <header className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <Link
            href="/dashboard"
            className="text-xs font-semibold uppercase tracking-[0.2em] text-slate"
          >
            Back to dashboard
          </Link>
          <h1 className="mt-3 font-[var(--font-fraunces)] text-3xl text-ink">
            {caseRecord.patientName ?? "Unknown patient"}
          </h1>
          <p className="mt-2 text-sm text-slate">
            {caseRecord.modality} • {caseRecord.fileName} •{" "}
            {caseRecord.patientId ?? "No patient ID"}
          </p>
          {caseRecord.notes ? (
            <p className="mt-3 max-w-2xl text-sm text-slate">
              Notes: {caseRecord.notes}
            </p>
          ) : null}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {results ? <RiskBadge level={results.risk} /> : null}
          <RunInferenceButton caseId={caseRecord.id} />
        </div>
      </header>

      <section className="mt-8">
        {results ? (
          <div className="mb-8 grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-2xl border border-slate/10 bg-white p-5 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-wider text-slate">
                Combined Risk
              </p>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-3xl font-bold text-ink">
                  {(results.risk_score ?? 0).toFixed(2)}
                </span>
                <span className="text-sm font-medium text-slate">/ 1.0</span>
              </div>
              <RiskBadge level={results.risk} />
            </div>

            <div className="rounded-2xl border border-slate/10 bg-white p-5 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-wider text-slate">
                Projected Stage
              </p>
              <div className="mt-2 text-3xl font-bold text-ink">
                {results.predicted_stage ?? "N/A"}
              </div>
              <p className="text-xs text-slate">Based on imaging patterns</p>
            </div>

            <div className="rounded-2xl border border-slate/10 bg-white p-5 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-wider text-slate">
                Imaging Risk
              </p>
              <div className="mt-2 text-2xl font-semibold text-ink">
                {(results.imaging_risk ?? 0).toFixed(2)}
              </div>
              <p className="text-xs text-slate">70% weight</p>
            </div>

            <div className="rounded-2xl border border-slate/10 bg-white p-5 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-wider text-slate">
                Clinical Risk
              </p>
              <div className="mt-2 text-2xl font-semibold text-ink">
                {(results.clinical_risk ?? 0).toFixed(2)}
              </div>
              <p className="text-xs text-slate">30% weight</p>
            </div>
          </div>
        ) : null}

        <Tabs
          items={[
            { id: "findings", label: "Findings", content: findingsContent },
            {
              id: "explainability",
              label: "Explainability",
              content: explainabilityContent
            },
            { id: "summary", label: "Medical Chatbot", content: summaryContent },
            { id: "model", label: "Model info", content: modelContent }
          ]}
        />
      </section>

      <div className="mt-8 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-800">
        Not a diagnosis. Use clinical judgment and confirm with radiology or
        pathology review.
      </div>
    </div>
  );
}
