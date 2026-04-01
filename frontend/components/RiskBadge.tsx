import type { RiskLevel } from "@/lib/types";

const styles: Record<RiskLevel, string> = {
  Low: "bg-emerald-100 text-emerald-700 border-emerald-200",
  Medium: "bg-amber-100 text-amber-700 border-amber-200",
  High: "bg-rose-100 text-rose-700 border-rose-200"
};

export function RiskBadge({ level }: { level: RiskLevel }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide ${styles[level]}`}
    >
      {level} risk
    </span>
  );
}
