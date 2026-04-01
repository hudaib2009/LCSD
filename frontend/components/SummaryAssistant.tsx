"use client";

import { useState } from "react";
import type { ChatMessage } from "@/lib/types";

const safetyLines = [
  "This report was automatically generated using an artificial intelligence system for radiological analysis.",
  "It is intended to assist clinical decision-making and must be reviewed and validated by a qualified healthcare professional before any medical decisions are made."
];
const CHATBOT_MODEL = "tngtech/deepseek-r1t2-chimera:free";

type SummaryAssistantProps = {
  caseId: string;
  modality: string;
  probability?: number;
  risk?: string;
  initialSummary?: string | null;
  initialReport?: string | null;
  initialChat: ChatMessage[];
};

export function SummaryAssistant({
  caseId,
  modality,
  probability,
  risk,
  initialSummary,
  initialReport,
  initialChat
}: SummaryAssistantProps) {
  const [assistantSummary, setAssistantSummary] = useState(
    initialSummary ?? ""
  );
  const [reportMarkdown, setReportMarkdown] = useState(initialReport ?? "");
  const [messages, setMessages] = useState<ChatMessage[]>(initialChat);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [warning, setWarning] = useState("");
  const missingKeyMessage =
    error.includes("OPENROUTER_API_KEY") && error.includes("not configured");

  async function submit(questionText?: string) {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`/api/cases/${caseId}/summary`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(questionText ? { question: questionText } : {})
      });

      const payload = (await response.json()) as {
        assistant_summary?: string;
        report_markdown?: string;
        warning?: string;
        help?: string;
        error?: string;
      };

      if (!response.ok) {
        const help = payload.help ? ` ${payload.help}` : "";
        throw new Error(`${payload.error || "Unable to generate report."}${help}`);
      }

      const summary = payload.assistant_summary ?? "";
      const report = payload.report_markdown ?? "";

      setAssistantSummary(summary);
      setReportMarkdown(report);
      setWarning(payload.warning ?? "");

      if (questionText) {
        const now = new Date().toISOString();
        setMessages((prev) => [
          ...prev,
          { role: "user", content: questionText, timestamp: now },
          { role: "assistant", content: summary, timestamp: now }
        ]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
      setWarning("");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-6">
      <div className="rounded-2xl border border-slate/10 bg-white/80 p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate">
              Medical Chatbot
            </p>
            <h3 className="mt-2 font-[var(--font-fraunces)] text-xl text-ink">
              AI-assisted radiology report
            </h3>
          </div>
          <button
            type="button"
            onClick={() => submit()}
            disabled={loading}
            className="rounded-full border border-ink/20 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-ink transition hover:border-ink disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? "Generating..." : "Generate report"}
          </button>
        </div>

        <div className="mt-4 grid gap-4 text-sm text-slate md:grid-cols-2">
          <div className="rounded-xl bg-mist px-4 py-3">
            <p className="text-xs uppercase text-slate">Modality</p>
            <p className="text-base font-semibold text-ink">{modality}</p>
          </div>
          <div className="rounded-xl bg-mist px-4 py-3">
            <p className="text-xs uppercase text-slate">Chatbot Model</p>
            <p className="text-base font-semibold text-ink">{CHATBOT_MODEL}</p>
          </div>
          <div className="rounded-xl bg-mist px-4 py-3">
            <p className="text-xs uppercase text-slate">Probability</p>
            <p className="text-base font-semibold text-ink">
              {probability !== undefined ? probability.toFixed(3) : "Pending"}
            </p>
          </div>
          <div className="rounded-xl bg-mist px-4 py-3">
            <p className="text-xs uppercase text-slate">Risk</p>
            <p className="text-base font-semibold text-ink">
              {risk ?? "Pending"}
            </p>
          </div>
        </div>

        <div className="mt-6 rounded-xl border border-slate/10 bg-white px-4 py-3 text-sm text-slate">
          {assistantSummary
            ? assistantSummary
            : "Generate a report or ask a follow-up question to update it."}
        </div>

        {warning ? (
          <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
            {warning}
          </div>
        ) : null}
        {error ? (
          <div className="mt-3 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
            <p>{error}</p>
            {missingKeyMessage ? (
              <p className="mt-2 text-rose-700">
                Add `OPENROUTER_API_KEY=YOUR_KEY_HERE` to `frontend/.env.local`, restart
                `npm run dev`, then retry.
              </p>
            ) : null}
            {error.includes("privacy") ? (
              <p className="mt-2">
                Update OpenRouter privacy settings:
                {" "}
                <a
                  className="underline decoration-rose-300 underline-offset-4"
                  href="https://openrouter.ai/settings/privacy"
                  target="_blank"
                  rel="noreferrer"
                >
                  openrouter.ai/settings/privacy
                </a>
              </p>
            ) : null}
          </div>
        ) : null}
      </div>

      <div className="rounded-2xl border border-slate/10 bg-white/80 p-6">
        <div className="flex items-center justify-between gap-3">
          <h4 className="font-[var(--font-fraunces)] text-lg text-ink">
            Draft radiology report
          </h4>
        </div>
        <pre className="mt-4 whitespace-pre-wrap rounded-xl bg-mist px-4 py-4 text-sm text-ink">
          {reportMarkdown || "Run the medical chatbot to draft a report."}
        </pre>
        <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-800">
          {safetyLines.map((line) => (
            <p key={line}>{line}</p>
          ))}
        </div>
      </div>

      <div className="rounded-2xl border border-slate/10 bg-white/80 p-6">
        <h4 className="font-[var(--font-fraunces)] text-lg text-ink">
          Chat
        </h4>
        <div className="mt-4 space-y-3">
          {messages.length === 0 ? (
            <p className="text-sm text-slate">
              Ask a follow-up question and the chatbot will respond here.
            </p>
          ) : (
            messages.map((message, index) => (
              <div
                key={`${message.role}-${index}`}
                className={`rounded-xl px-4 py-3 text-sm ${
                  message.role === "user"
                    ? "bg-ink/5 text-ink"
                    : "bg-mist text-slate"
                }`}
              >
                <p className="text-xs font-semibold uppercase tracking-[0.2em]">
                  {message.role}
                </p>
                <p className="mt-2 whitespace-pre-wrap">{message.content}</p>
              </div>
            ))
          )}
        </div>
        <div className="mt-5 flex flex-col gap-3 md:flex-row">
          <input
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask a clinical follow-up question..."
            className="flex-1 rounded-full border border-slate/20 bg-white px-4 py-3 text-sm text-ink outline-none focus:border-ink/40"
          />
          <button
            type="button"
            onClick={() => {
              const trimmed = question.trim();
              if (!trimmed) {
                return;
              }
              setQuestion("");
              submit(trimmed);
            }}
            disabled={loading}
            className="rounded-full bg-ink px-5 py-3 text-xs font-semibold uppercase tracking-[0.2em] text-white transition hover:bg-ink/90 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? "Sending..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
