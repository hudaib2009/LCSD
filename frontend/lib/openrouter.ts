type OpenRouterMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

type OpenRouterResponse = {
  ok: boolean;
  status: number;
  data: unknown | null;
  errorMessage: string | null;
};

const DEFAULT_MODEL = "tngtech/deepseek-r1t2-chimera:free";

export async function callOpenRouterChat(args: {
  messages: OpenRouterMessage[];
  model?: string;
}) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    return {
      ok: false,
      status: 503,
      data: null,
      errorMessage:
        "OPENROUTER_API_KEY is not configured. Set it in frontend/.env.local or env vars."
    } satisfies OpenRouterResponse;
  }

  const model = args.model || process.env.OPENROUTER_MODEL || DEFAULT_MODEL;

  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "http://localhost:3000",
      "X-Title": "CSD Clinical Support"
    },
    body: JSON.stringify({
      model,
      messages: args.messages,
      temperature: 0.2
    })
  });

  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  const errorMessage =
    typeof payload === "object" && payload
      ? (payload as { error?: { message?: string } }).error?.message ?? null
      : null;

  return {
    ok: response.ok,
    status: response.status,
    data: payload,
    errorMessage
  } satisfies OpenRouterResponse;
}

export function extractOpenRouterText(payload: unknown) {
  if (!payload || typeof payload !== "object") {
    return "";
  }
  const data = payload as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  return data.choices?.[0]?.message?.content ?? "";
}
