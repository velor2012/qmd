import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const ENV_KEYS = [
  "QMD_RERANK_CONTEXT_SIZE",
  "RERANK_CONTEXT_SIZE",
];

const envBackup = new Map<string, string | undefined>();

describe("LLM env prefix support", () => {
  beforeEach(() => {
    for (const key of ENV_KEYS) {
      envBackup.set(key, process.env[key]);
      delete process.env[key];
    }
    vi.resetModules();
  });

  afterEach(() => {
    for (const key of ENV_KEYS) {
      const value = envBackup.get(key);
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    envBackup.clear();
    vi.resetModules();
  });

  test("LlamaCpp reads QMD_RERANK_CONTEXT_SIZE at module init", async () => {
    process.env.QMD_RERANK_CONTEXT_SIZE = "3072";
    const llm = await import("../src/llm.js");
    expect((llm.LlamaCpp as any).RERANK_CONTEXT_SIZE).toBe(3072);
  });

  test("LlamaCpp falls back to legacy RERANK_CONTEXT_SIZE when QMD_ prefix is missing", async () => {
    process.env.RERANK_CONTEXT_SIZE = "4096";
    const llm = await import("../src/llm.js");
    expect((llm.LlamaCpp as any).RERANK_CONTEXT_SIZE).toBe(4096);
  });
});
