import Link from "next/link";
import { CaseCard } from "@/components/CaseCard";
import { readCases, readResults } from "@/lib/store";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  const cases = await readCases();
  const caseWithRisk = await Promise.all(
    cases.map(async (item) => {
      const results = await readResults(item.id);
      return { item, risk: results?.risk ?? null };
    })
  );

  return (
    <div className="px-6 py-10 lg:px-12">
      <header className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-slate">
            Clinical decision support
          </p>
          <h1 className="mt-3 font-[var(--font-fraunces)] text-4xl text-ink">
            Imaging triage dashboard
          </h1>
          <p className="mt-3 max-w-2xl text-sm text-slate">
            Review recent cases, run inference, and share model context with the
            care team. Outputs are decision support only.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <Link
            href="/cases/new"
            className="rounded-full bg-ink px-5 py-2 text-sm font-semibold text-white shadow-soft"
          >
            New case
          </Link>
          <Link
            href="/demo"
            className="rounded-full border border-slate/20 px-5 py-2 text-sm font-semibold text-slate"
          >
            Demo workflow
          </Link>
          <Link
            href="/model-info"
            className="rounded-full border border-slate/20 px-5 py-2 text-sm font-semibold text-slate"
          >
            Model info
          </Link>
        </div>
      </header>

      <section className="mt-10 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {caseWithRisk.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-slate/30 bg-white/60 p-6 text-sm text-slate">
            No cases yet. Upload a new study to get started.
          </div>
        ) : (
          caseWithRisk.map(({ item, risk }) => (
            <CaseCard key={item.id} item={item} risk={risk} />
          ))
        )}
      </section>

      <div className="mt-8 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-800">
        Not a diagnosis. Use clinical judgment and confirm with radiology or
        pathology review.
      </div>
    </div>
  );
}
