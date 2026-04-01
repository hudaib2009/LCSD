"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export function RunInferenceButton({ caseId }: { caseId: string }) {
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const runInference = async () => {
    setLoading(true);
    await fetch(`/api/cases/${caseId}/run`, { method: "POST" });
    router.refresh();
    setLoading(false);
  };

  return (
    <button
      type="button"
      onClick={runInference}
      className="rounded-full bg-sea px-5 py-2 text-sm font-semibold text-white disabled:opacity-60"
      disabled={loading}
    >
      {loading ? "Running..." : "Run inference"}
    </button>
  );
}
