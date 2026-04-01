import Link from "next/link";

const modelInfo = {
  name: "CSD Imaging Triage",
  version: "0.8.2-mock",
  metrics: {
    auc: 0.92,
    sensitivity: 0.88,
    specificity: 0.85
  },
  limitations: [
    "Validated on retrospective datasets only.",
    "Performance may vary with scanner protocols and site-specific prevalence.",
    "Not evaluated for pediatric populations.",
    "May underperform on rare pathologies or low-quality scans."
  ],
  disclaimers: [
    "Not a diagnosis. Use clinical judgment and confirm with radiology/pathology review.",
    "Outputs are intended for triage support and must not replace standard-of-care workflows.",
    "Use in accordance with local governance and informed consent policies."
  ],
  lastValidated: "2024-06-15"
};

export default function ModelInfoPage() {
  return (
    <div className="px-6 py-10 lg:px-12">
      <header className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-slate">
            Model info
          </p>
          <h1 className="mt-2 font-[var(--font-fraunces)] text-3xl text-ink">
            {modelInfo.name}
          </h1>
          <p className="mt-1 text-xs text-slate">Version {modelInfo.version}</p>
        </div>
        <Link
          href="/dashboard"
          className="rounded-full border border-slate/20 px-4 py-2 text-xs font-semibold text-slate"
        >
          Back to dashboard
        </Link>
      </header>

      <section className="mt-8 grid gap-6">
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="rounded-2xl border border-slate/10 bg-white/80 p-5">
            <p className="text-xs uppercase text-slate">AUC</p>
            <p className="mt-2 text-3xl font-semibold text-ink">
              {modelInfo.metrics.auc.toFixed(2)}
            </p>
          </div>
          <div className="rounded-2xl border border-slate/10 bg-white/80 p-5">
            <p className="text-xs uppercase text-slate">Sensitivity</p>
            <p className="mt-2 text-3xl font-semibold text-ink">
              {modelInfo.metrics.sensitivity.toFixed(2)}
            </p>
          </div>
          <div className="rounded-2xl border border-slate/10 bg-white/80 p-5">
            <p className="text-xs uppercase text-slate">Specificity</p>
            <p className="mt-2 text-3xl font-semibold text-ink">
              {modelInfo.metrics.specificity.toFixed(2)}
            </p>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <div className="rounded-2xl border border-slate/10 bg-white/80 p-6">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate">
              Limitations
            </p>
            <ul className="mt-4 space-y-2 text-sm text-slate">
              {modelInfo.limitations.map((item) => (
                <li key={item}>• {item}</li>
              ))}
            </ul>
          </div>
          <div className="rounded-2xl border border-slate/10 bg-white/80 p-6">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate">
              Disclaimers
            </p>
            <ul className="mt-4 space-y-2 text-sm text-slate">
              {modelInfo.disclaimers.map((item) => (
                <li key={item}>• {item}</li>
              ))}
            </ul>
            <p className="mt-3 text-xs text-slate">
              Last validated: {modelInfo.lastValidated}
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
