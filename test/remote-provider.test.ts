import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { RemoteLLM } from "../src/remote.js";
import { disposeDefaultLLM, getDefaultLLM, getProviderInfo } from "../src/llm.js";

const ENV_KEYS = [
  "QMD_VOYAGE_API_KEY",
  "QMD_VOYAGE_API_BASE",
  "QMD_VOYAGE_EMBED_MODEL",
  "QMD_VOYAGE_RERANK_MODEL",
  "QMD_OPENAI_API_KEY",
  "QMD_OPENAI_API_BASE",
  "QMD_OPENAI_EMBED_MODEL",
  "QMD_OPENAI_RERANK_MODEL",
  "QMD_PROVIDER",
  "VOYAGE_API_KEY",
  "VOYAGE_API_BASE",
  "VOYAGE_EMBED_MODEL",
  "VOYAGE_RERANK_MODEL",
  "OPENAI_API_KEY",
  "OPENAI_API_BASE",
  "OPENAI_EMBED_MODEL",
  "OPENAI_RERANK_MODEL",
];

const envBackup = new Map<string, string | undefined>();

function setFetchMock(payload: unknown, status: number = 200) {
  const mock = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
    text: async () => JSON.stringify(payload),
  });
  vi.stubGlobal("fetch", mock as unknown as typeof fetch);
  return mock;
}

describe("Remote providers", () => {
  beforeEach(() => {
    for (const key of ENV_KEYS) {
      envBackup.set(key, process.env[key]);
      delete process.env[key];
    }
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  afterEach(async () => {
    await disposeDefaultLLM();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    for (const key of ENV_KEYS) {
      const value = envBackup.get(key);
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    envBackup.clear();
  });

  test("RemoteLLM(voyage) sends query input_type and default model", async () => {
    process.env.QMD_VOYAGE_API_KEY = "test-key";
    const fetchMock = setFetchMock({
      object: "list",
      data: [{ object: "embedding", embedding: [0.1, 0.2], index: 0 }],
      model: "voyage-4-lite",
      usage: { total_tokens: 12 },
    });

    const llm = new RemoteLLM({ provider: "voyage" });
    const result = await llm.embed("hello world", { isQuery: true });

    expect(result).not.toBeNull();
    expect(result!.model).toBe("voyage-4-lite");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("https://api.voyageai.com/v1/embeddings");
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.model).toBe("voyage-4-lite");
    expect(body.input_type).toBe("query");
    expect(body.input).toEqual(["hello world"]);
  });

  test("RemoteLLM embedBatch maps response by returned index", async () => {
    process.env.QMD_VOYAGE_API_KEY = "test-key";
    setFetchMock({
      object: "list",
      data: [
        { object: "embedding", embedding: [0.3], index: 1 },
        { object: "embedding", embedding: [0.1], index: 0 },
      ],
      model: "voyage-4-lite",
      usage: { total_tokens: 9 },
    });

    const llm = new RemoteLLM({ provider: "voyage" });
    const result = await llm.embedBatch(["a", "b"], { isQuery: false });

    expect(result).toHaveLength(2);
    expect(result[0]!.embedding).toEqual([0.1]);
    expect(result[1]!.embedding).toEqual([0.3]);
  });

  test("RemoteLLM(openai) rerank falls back to input order", async () => {
    process.env.QMD_OPENAI_API_BASE = "http://localhost:11434/v1";
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const llm = new RemoteLLM({ provider: "openai" });

    const result = await llm.rerank("query", [
      { file: "a.md", text: "A" },
      { file: "b.md", text: "B" },
    ]);

    expect(result.model).toBe("none");
    expect(result.results.map((r) => r.file)).toEqual(["a.md", "b.md"]);
    expect(result.results[0]!.score).toBeGreaterThan(result.results[1]!.score);
    expect(warnSpy).toHaveBeenCalled();
  });

  test("RemoteLLM(openai) uses OPENAI_RERANK_MODEL for /rerank requests", async () => {
    process.env.QMD_OPENAI_API_BASE = "http://localhost:11434/v1";
    process.env.OPENAI_RERANK_MODEL = "BAAI/bge-reranker-v2-m3";
    const fetchMock = setFetchMock({
      object: "list",
      data: [
        { index: 1, relevance_score: 0.91 },
        { index: 0, relevance_score: 0.22 },
      ],
      model: "BAAI/bge-reranker-v2-m3",
      usage: { total_tokens: 18 },
    });
    const llm = new RemoteLLM({ provider: "openai" });

    const result = await llm.rerank("query", [
      { file: "a.md", text: "A" },
      { file: "b.md", text: "B" },
    ]);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://localhost:11434/v1/rerank");
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.model).toBe("BAAI/bge-reranker-v2-m3");
    expect(body.query).toBe("query");
    expect(body.documents).toEqual(["A", "B"]);
    expect(result.model).toBe("BAAI/bge-reranker-v2-m3");
    expect(result.results.map((r) => r.file)).toEqual(["b.md", "a.md"]);
  });

  test("RemoteLLM(voyage) throws when API key is missing", () => {
    expect(() => new RemoteLLM({ provider: "voyage" })).toThrow("Voyage API key required");
  });

  test("getDefaultLLM selects remote provider and keeps singleton", async () => {
    process.env.QMD_PROVIDER = "openai";
    process.env.QMD_OPENAI_API_BASE = "http://localhost:11434/v1";

    const llm1 = await getDefaultLLM();
    const llm2 = await getDefaultLLM();

    expect(llm1).toBe(llm2);
    expect((llm1 as RemoteLLM).getProvider()).toBe("openai");
  });

  test("disposeDefaultLLM resets provider singleton", async () => {
    process.env.QMD_PROVIDER = "openai";
    process.env.QMD_OPENAI_API_BASE = "http://localhost:11434/v1";
    const llm1 = await getDefaultLLM();

    await disposeDefaultLLM();
    const llm2 = await getDefaultLLM();

    expect(llm2).not.toBe(llm1);
    expect((llm2 as RemoteLLM).getProvider()).toBe("openai");
  });

  test("getProviderInfo reports provider-specific model defaults", () => {
    expect(getProviderInfo()).toEqual({
      provider: "local",
      embedModel: "embeddinggemma",
      rerankModel: "ExpedientFalcon/qwen3-reranker:0.6b-q8_0",
    });

    process.env.QMD_PROVIDER = "voyage";
    expect(getProviderInfo()).toEqual({
      provider: "voyage",
      embedModel: "voyage-4-lite",
      rerankModel: "rerank-2",
    });

    process.env.QMD_PROVIDER = "openai";
    process.env.QMD_OPENAI_EMBED_MODEL = "text-embedding-3-large";
    process.env.OPENAI_RERANK_MODEL = "BAAI/bge-reranker-v2-m3";
    expect(getProviderInfo()).toEqual({
      provider: "openai",
      embedModel: "text-embedding-3-large",
      rerankModel: "BAAI/bge-reranker-v2-m3",
    });
  });
});
