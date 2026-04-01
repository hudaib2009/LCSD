"use client";

import { useEffect, useMemo, useState } from "react";

type ExplainabilityPanelProps = {
  heatmapPath?: string | null;
  overlayPath?: string | null;
  error?: string | null;
  version?: string;
};

type MetaResponse = {
  exists: boolean;
  size: number | null;
  contentType: string | null;
};

type LoadFailure = {
  url: string;
  message: string;
};

async function fetchMeta(path: string) {
  const response = await fetch(
    `/api/files/meta?path=${encodeURIComponent(path)}`
  );
  if (!response.ok) {
    return { status: response.status, data: null as MetaResponse | null };
  }
  const data = (await response.json()) as MetaResponse;
  return { status: response.status, data };
}

async function probeStatus(url: string) {
  try {
    const response = await fetch(url);
    return `Failed to load (HTTP ${response.status})`;
  } catch {
    return "Failed to load (onError triggered)";
  }
}

function fileUrl(path: string, version?: string) {
  const base = `/api/files?path=${encodeURIComponent(path)}`;
  if (!version) {
    return base;
  }
  const separator = base.includes("?") ? "&" : "?";
  return `${base}${separator}v=${encodeURIComponent(version)}`;
}

export function ExplainabilityPanel({
  heatmapPath,
  overlayPath,
  error,
  version
}: ExplainabilityPanelProps) {
  const [heatmapMeta, setHeatmapMeta] = useState<MetaResponse | null>(null);
  const [overlayMeta, setOverlayMeta] = useState<MetaResponse | null>(null);
  const [heatmapMetaStatus, setHeatmapMetaStatus] = useState<number | null>(
    null
  );
  const [overlayMetaStatus, setOverlayMetaStatus] = useState<number | null>(
    null
  );
  const [heatmapFailure, setHeatmapFailure] = useState<LoadFailure | null>(
    null
  );
  const [overlayFailure, setOverlayFailure] = useState<LoadFailure | null>(null);
  const [heatmapFailed, setHeatmapFailed] = useState(false);
  const [overlayFailed, setOverlayFailed] = useState(false);

  const heatmapUrl = useMemo(() => {
    return heatmapPath ? fileUrl(heatmapPath, version) : null;
  }, [heatmapPath, version]);
  const overlayUrl = useMemo(() => {
    return overlayPath ? fileUrl(overlayPath, version) : null;
  }, [overlayPath, version]);

  useEffect(() => {
    let mounted = true;
    async function loadMeta() {
      if (heatmapPath) {
        const result = await fetchMeta(heatmapPath);
        if (mounted) {
          setHeatmapMeta(result.data);
          setHeatmapMetaStatus(result.status);
        }
      } else {
        setHeatmapMeta(null);
        setHeatmapMetaStatus(null);
      }

      if (overlayPath) {
        const result = await fetchMeta(overlayPath);
        if (mounted) {
          setOverlayMeta(result.data);
          setOverlayMetaStatus(result.status);
        }
      } else {
        setOverlayMeta(null);
        setOverlayMetaStatus(null);
      }
    }
    loadMeta();
    return () => {
      mounted = false;
    };
  }, [heatmapPath, overlayPath]);

  useEffect(() => {
    setHeatmapFailure(null);
    setOverlayFailure(null);
    setHeatmapFailed(false);
    setOverlayFailed(false);
  }, [heatmapPath, overlayPath, version]);

  const placeholder = "/placeholders/heatmap-ct.png";
  const overlayHref = overlayPath ? fileUrl(overlayPath, version) : null;
  const heatmapHref = heatmapPath ? fileUrl(heatmapPath, version) : null;

  function explainReason(kind: "heatmap" | "overlay") {
    const path = kind === "heatmap" ? heatmapPath : overlayPath;
    const meta = kind === "heatmap" ? heatmapMeta : overlayMeta;
    const metaStatus = kind === "heatmap" ? heatmapMetaStatus : overlayMetaStatus;
    const failure = kind === "heatmap" ? heatmapFailure : overlayFailure;

    if (failure) {
      return "Failed to stream file from /api/files.";
    }
    if (!path) {
      return "No explainability generated yet. Run inference.";
    }
    if (metaStatus === 400) {
      return "Explainability path points outside storage.";
    }
    if (meta && !meta.exists) {
      return "Explainability file missing on disk.";
    }
    if (error) {
      return error;
    }
    return "Explainability file status unknown.";
  }

  return (
    <div className="grid gap-6">
      <div className="w-full max-w-5xl mx-auto">
        <div className="grid grid-cols-1 items-start gap-4 md:grid-cols-2">
          <div className="rounded-lg border border-slate/10 bg-white p-2">
            <div className="text-sm font-medium mb-2">Overlay</div>
            <div className="relative w-full overflow-hidden rounded-md border border-slate/10 bg-white">
              {overlayUrl && !overlayFailed ? (
                <img
                  src={overlayUrl}
                  alt="Explainability overlay"
                  className="block w-full h-auto object-contain rounded-md"
                  onError={async () => {
                    const message = await probeStatus(overlayUrl);
                    setOverlayFailed(true);
                    setOverlayFailure({ url: overlayUrl, message });
                  }}
                />
              ) : (
                <img
                  src={placeholder}
                  alt="Explainability overlay placeholder"
                  className="block w-full h-auto object-contain rounded-md"
                />
              )}
            </div>
            {overlayHref ? (
              <a
                href={overlayHref}
                target="_blank"
                rel="noreferrer"
                className="text-xs underline mt-1 inline-block text-slate"
              >
                Open full size
              </a>
            ) : null}
            <p className="mt-2 text-xs text-rose-600">
              {explainReason("overlay")}
            </p>
            {overlayFailure ? (
              <p className="mt-1 text-xs text-slate">
                {overlayFailure.message} — {overlayFailure.url}
              </p>
            ) : null}
          </div>

          <div className="rounded-lg border border-slate/10 bg-white p-2">
            <div className="text-sm font-medium mb-2">Heatmap</div>
            <div className="relative w-full overflow-hidden rounded-md border border-slate/10 bg-white">
              {heatmapUrl && !heatmapFailed ? (
                <img
                  src={heatmapUrl}
                  alt="Explainability heatmap"
                  className="block w-full h-auto object-contain rounded-md"
                  onError={async () => {
                    const message = await probeStatus(heatmapUrl);
                    setHeatmapFailed(true);
                    setHeatmapFailure({ url: heatmapUrl, message });
                  }}
                />
              ) : (
                <img
                  src={placeholder}
                  alt="Explainability heatmap placeholder"
                  className="block w-full h-auto object-contain rounded-md"
                />
              )}
            </div>
            {heatmapHref ? (
              <a
                href={heatmapHref}
                target="_blank"
                rel="noreferrer"
                className="text-xs underline mt-1 inline-block text-slate"
              >
                Open full size
              </a>
            ) : null}
            <p className="mt-2 text-xs text-rose-600">
              {explainReason("heatmap")}
            </p>
            {heatmapFailure ? (
              <p className="mt-1 text-xs text-slate">
                {heatmapFailure.message} — {heatmapFailure.url}
              </p>
            ) : null}
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-slate/10 bg-white/80 p-4 text-xs text-slate">
        <p className="font-semibold uppercase tracking-[0.2em] text-slate">
          Debug details
        </p>
        <div className="mt-3 space-y-2">
          <p>overlay_path: {overlayPath ?? "null"}</p>
          <p>heatmap_path: {heatmapPath ?? "null"}</p>
          <p>overlay_path missing: {overlayPath ? "no" : "yes"}</p>
          <p>heatmap_path missing: {heatmapPath ? "no" : "yes"}</p>
          {overlayMeta ? (
            <p>
              overlay meta: exists={String(overlayMeta.exists)} size=
              {overlayMeta.size ?? "null"} contentType={overlayMeta.contentType ?? "null"}
            </p>
          ) : null}
          {heatmapMeta ? (
            <p>
              heatmap meta: exists={String(heatmapMeta.exists)} size=
              {heatmapMeta.size ?? "null"} contentType={heatmapMeta.contentType ?? "null"}
            </p>
          ) : null}
          {error ? <p>explainability.error: {error}</p> : null}
        </div>
      </div>
    </div>
  );
}
