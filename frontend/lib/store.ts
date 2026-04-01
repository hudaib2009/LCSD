import { promises as fs } from "fs";
import path from "path";
import type {
  CaseRecord,
  ChatMessage,
  InferenceResult,
  SummaryRecord
} from "./types";

const dataDir = path.join(process.cwd(), "data");
const storageDir = path.join(process.cwd(), "storage");
const casesPath = path.join(dataDir, "cases.json");

async function ensureDir(dir: string) {
  await fs.mkdir(dir, { recursive: true });
}

export async function readCases(): Promise<CaseRecord[]> {
  try {
    const payload = await fs.readFile(casesPath, "utf-8");
    return JSON.parse(payload) as CaseRecord[];
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      await ensureDir(dataDir);
      await fs.writeFile(casesPath, "[]", "utf-8");
      return [];
    }
    throw error;
  }
}

export async function writeCases(cases: CaseRecord[]) {
  await ensureDir(dataDir);
  await fs.writeFile(casesPath, JSON.stringify(cases, null, 2), "utf-8");
}

export async function getCaseById(id: string) {
  const cases = await readCases();
  return cases.find((item) => item.id === id) ?? null;
}

export async function addCase(caseRecord: CaseRecord) {
  const cases = await readCases();
  cases.unshift(caseRecord);
  await writeCases(cases);
}

export async function updateCase(id: string, update: Partial<CaseRecord>) {
  const cases = await readCases();
  const next = cases.map((item) =>
    item.id === id ? { ...item, ...update } : item
  );
  await writeCases(next);
}

export async function saveUpload(id: string, fileName: string, buffer: Buffer) {
  const caseDir = path.join(storageDir, id);
  await ensureDir(caseDir);
  const filePath = path.join(caseDir, fileName);
  await fs.writeFile(filePath, buffer);
  return filePath;
}

export async function saveResults(id: string, results: InferenceResult) {
  const caseDir = path.join(storageDir, id);
  await ensureDir(caseDir);
  const resultsPath = path.join(caseDir, "results.json");
  await fs.writeFile(resultsPath, JSON.stringify(results, null, 2), "utf-8");
  return resultsPath;
}

export async function readResults(id: string): Promise<InferenceResult | null> {
  try {
    const resultsPath = path.join(storageDir, id, "results.json");
    const payload = await fs.readFile(resultsPath, "utf-8");
    return JSON.parse(payload) as InferenceResult;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return null;
    }
    throw error;
  }
}

export async function readSummary(id: string): Promise<SummaryRecord | null> {
  try {
    const summaryPath = path.join(storageDir, id, "summary.json");
    const payload = await fs.readFile(summaryPath, "utf-8");
    return JSON.parse(payload) as SummaryRecord;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return null;
    }
    throw error;
  }
}

export async function saveSummary(id: string, summary: SummaryRecord) {
  const caseDir = path.join(storageDir, id);
  await ensureDir(caseDir);
  const summaryPath = path.join(caseDir, "summary.json");
  await fs.writeFile(summaryPath, JSON.stringify(summary, null, 2), "utf-8");
  return summaryPath;
}

export async function readChat(id: string): Promise<ChatMessage[]> {
  try {
    const chatPath = path.join(storageDir, id, "chat.json");
    const payload = await fs.readFile(chatPath, "utf-8");
    return JSON.parse(payload) as ChatMessage[];
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return [];
    }
    throw error;
  }
}

export async function saveChat(id: string, messages: ChatMessage[]) {
  const caseDir = path.join(storageDir, id);
  await ensureDir(caseDir);
  const chatPath = path.join(caseDir, "chat.json");
  await fs.writeFile(chatPath, JSON.stringify(messages, null, 2), "utf-8");
  return chatPath;
}
