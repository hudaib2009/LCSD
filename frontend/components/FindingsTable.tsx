import type { Finding } from "@/lib/types";

export function FindingsTable({ findings }: { findings: Finding[] }) {
  return (
    <div className="overflow-hidden rounded-2xl border border-slate/10 bg-white">
      <table className="w-full text-left text-sm">
        <thead className="bg-mist text-xs uppercase tracking-wide text-slate">
          <tr>
            <th className="px-4 py-3">Finding</th>
            <th className="px-4 py-3">Probability</th>
            <th className="px-4 py-3">Confidence</th>
          </tr>
        </thead>
        <tbody>
          {findings.map((finding) => (
            <tr key={finding.label} className="border-t border-slate/10">
              <td className="px-4 py-3 font-medium text-ink">
                {finding.label}
              </td>
              <td className="px-4 py-3">
                {(finding.probability * 100).toFixed(0)}%
              </td>
              <td className="px-4 py-3">
                {(finding.confidence * 100).toFixed(0)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
