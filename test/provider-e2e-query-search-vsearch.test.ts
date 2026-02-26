import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import Database from "better-sqlite3";

type Provider = "local" | "voyage" | "openai";

type CommandResult = {
  exitCode: number;
  stdout: string;
  stderr: string;
  stdoutLog: string;
  stderrLog: string;
};

const PROVIDERS: Provider[] = ["local", "voyage", "openai"];
const COLLECTION_NAME = process.env.QMD_E2E_COLLECTION_NAME || "provider-e2e-docs";
const QUERY_TEXT = process.env.QMD_E2E_QUERY_TEXT || "新概念";
const TOP_N = process.env.QMD_E2E_TOP_N || "5";
const RUN_REMOTE = process.env.QMD_E2E_RUN_REMOTE === "1";

const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const qmdScript = join(projectRoot, "src", "qmd.ts");

let runtimeRoot = "";
let logRoot = "";
let fixtureRoot = "";
let collectionPath = "";

function skipReason(provider: Provider): string | null {
  if ((provider === "voyage" || provider === "openai") && !RUN_REMOTE) {
    return "remote providers disabled (set QMD_E2E_RUN_REMOTE=1)";
  }
  if (provider === "voyage" && !process.env.QMD_VOYAGE_API_KEY) {
    return "missing QMD_VOYAGE_API_KEY";
  }
  if (provider === "openai" && !process.env.QMD_OPENAI_API_KEY && !process.env.QMD_OPENAI_API_BASE) {
    return "missing QMD_OPENAI_API_KEY or QMD_OPENAI_API_BASE";
  }
  return null;
}

function providerEnv(provider: Provider): Record<string, string> {
  if (provider === "local") return { QMD_PROVIDER: "local" };
  if (provider === "voyage") return { QMD_PROVIDER: "voyage" };
  return { QMD_PROVIDER: "openai" };
}

async function createProviderEnv(provider: Provider): Promise<{ dbPath: string; configDir: string }> {
  const providerRoot = join(runtimeRoot, provider);
  const configDir = join(providerRoot, "config");
  const dbPath = join(providerRoot, "index.sqlite");
  await mkdir(configDir, { recursive: true });
  await writeFile(join(configDir, "index.yml"), "collections: {}\n");
  return { dbPath, configDir };
}

async function runQmdStep(
  provider: Provider,
  step: string,
  args: string[],
  extraEnv: Record<string, string>,
  dbPath: string,
  configDir: string,
): Promise<CommandResult> {
  const providerLogDir = join(logRoot, provider);
  await mkdir(providerLogDir, { recursive: true });
  const stdoutLog = join(providerLogDir, `${step}.stdout.log`);
  const stderrLog = join(providerLogDir, `${step}.stderr.log`);

  const proc = spawn("node", ["--import", "tsx", qmdScript, ...args], {
    cwd: projectRoot,
    env: {
      ...process.env,
      ...extraEnv,
      INDEX_PATH: dbPath,
      QMD_CONFIG_DIR: configDir,
      NO_COLOR: "1",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  const stdoutPromise = new Promise<string>((resolve, reject) => {
    let data = "";
    proc.stdout?.on("data", (chunk: Buffer) => { data += chunk.toString(); });
    proc.once("error", reject);
    proc.stdout?.once("end", () => resolve(data));
  });
  const stderrPromise = new Promise<string>((resolve, reject) => {
    let data = "";
    proc.stderr?.on("data", (chunk: Buffer) => { data += chunk.toString(); });
    proc.once("error", reject);
    proc.stderr?.once("end", () => resolve(data));
  });
  const codePromise = new Promise<number>((resolve, reject) => {
    proc.once("error", reject);
    proc.once("close", (code) => resolve(code ?? 1));
  });

  const [stdout, stderr, exitCode] = await Promise.all([stdoutPromise, stderrPromise, codePromise]);
  await writeFile(stdoutLog, stdout);
  await writeFile(stderrLog, stderr);

  return { exitCode, stdout, stderr, stdoutLog, stderrLog };
}

async function createFixtureDocs(dir: string): Promise<void> {
  await mkdir(dir, { recursive: true });
  await writeFile(
    join(dir, "notes.md"),
    `# 新概念英语学习笔记

这是用于 provider E2E 测试的样例文档。
主题：新概念、英语教材、课文、语法。
`
  );
  await writeFile(
    join(dir, "crawler.md"),
    `# Crawler Docs

Collection add/remove/embed/query integration test fixture.
Includes multilingual terms: 新概念, semantic search, rerank.
`
  );
  await writeFile(
    join(dir, "meeting.md"),
    `# Meeting Memo

Action items:
1. run qmd embed -f
2. run qmd query "新概念"
`
  );
}

function getVectorCount(dbPath: string): number {
  const db = new Database(dbPath, { readonly: true });
  try {
    const row = db.prepare("SELECT COUNT(*) as n FROM content_vectors").get() as { n: number };
    return row.n;
  } catch {
    return 0;
  } finally {
    db.close();
  }
}

describe("provider e2e flow", () => {
  beforeAll(async () => {
    runtimeRoot = await mkdtemp(join(tmpdir(), "qmd-provider-e2e-"));
    logRoot = join(projectRoot, "test", ".tmp", `provider-e2e-${Date.now()}`);
    fixtureRoot = join(projectRoot, "test", ".tmp", `provider-e2e-fixtures-${Date.now()}`);
    collectionPath = fixtureRoot;
    await mkdir(logRoot, { recursive: true });
    await createFixtureDocs(fixtureRoot);
  });

  afterAll(async () => {
    if (runtimeRoot) {
      await rm(runtimeRoot, { recursive: true, force: true });
    }
    if (fixtureRoot) {
      await rm(fixtureRoot, { recursive: true, force: true });
    }
  });

  for (const provider of PROVIDERS) {
    const reason = skipReason(provider);
    const title = `[${provider}] rm/add/embed/search/vsearch/query flow`;

    if (reason) {
      test.skip(`${title} (skip: ${reason})`, async () => {
        // Marked as skipped at definition time.
      });
      continue;
    }

    test(title, async () => {
      const { dbPath, configDir } = await createProviderEnv(provider);
      const env = providerEnv(provider);

      await runQmdStep(provider, "collection_rm", ["collection", "rm", COLLECTION_NAME], env, dbPath, configDir);

      const addResult = await runQmdStep(
        provider,
        "collection_add",
        ["collection", "add", collectionPath, "--name", COLLECTION_NAME],
        env,
        dbPath,
        configDir,
      );
      expect(addResult.exitCode, `[${provider}] collection add failed. logs: ${addResult.stdoutLog} | ${addResult.stderrLog}`).toBe(0);

      const embedResult = await runQmdStep(provider, "embed_force", ["embed", "-f"], env, dbPath, configDir);
      expect(embedResult.exitCode, `[${provider}] embed -f failed. logs: ${embedResult.stdoutLog} | ${embedResult.stderrLog}`).toBe(0);
      const vectorCount = getVectorCount(dbPath);
      expect(vectorCount, `[${provider}] embed finished but no vectors were written (db: ${dbPath})`).toBeGreaterThan(0);

      const searchResult = await runQmdStep(provider, "search", ["search", "--json", "-n", TOP_N, QUERY_TEXT], env, dbPath, configDir);
      expect(searchResult.exitCode, `[${provider}] search failed. logs: ${searchResult.stdoutLog} | ${searchResult.stderrLog}`).toBe(0);

      const vsearchResult = await runQmdStep(provider, "vsearch", ["vsearch", "--json", "-n", TOP_N, QUERY_TEXT], env, dbPath, configDir);
      expect(vsearchResult.exitCode, `[${provider}] vsearch failed. logs: ${vsearchResult.stdoutLog} | ${vsearchResult.stderrLog}`).toBe(0);

      const queryResult = await runQmdStep(provider, "query", ["query", "--json", "-n", TOP_N, QUERY_TEXT], env, dbPath, configDir);
      expect(queryResult.exitCode, `[${provider}] query failed. logs: ${queryResult.stdoutLog} | ${queryResult.stderrLog}`).toBe(0);
    }, 30 * 60 * 1000);
  }
});
