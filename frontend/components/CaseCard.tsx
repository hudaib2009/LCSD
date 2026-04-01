import Link from "next/link";
import type { CaseRecord, RiskLevel } from "@/lib/types";
import { RiskBadge } from "./RiskBadge";

export function CaseCard({
  item,
  risk
}: {
  item: CaseRecord;
  risk?: RiskLevel | null;
}) {
  return (
    <Link
      href={`/cases/${item.id}`}
      className="group rounded-2xl border border-slate/10 bg-white/70 p-5 shadow-soft transition hover:-translate-y-1 hover:border-sea/30"
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate">
            {item.modality} case
          </p>
          <h3 className="mt-2 font-[var(--font-fraunces)] text-xl text-ink">
            {item.patientName ?? "Unknown patient"}
          </h3>
          <p className="mt-1 text-sm text-slate">
            {item.patientId ?? "No patient ID"} • {item.fileName}
          </p>
        </div>
        {risk ? <RiskBadge level={risk} /> : null}
      </div>
      <div className="mt-4 flex items-center justify-between text-xs text-slate">
        <span>Status: {item.status}</span>
        <span>{new Date(item.createdAt).toLocaleString()}</span>
      </div>
    </Link>
  );
}
