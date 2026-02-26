# Provider E2E Test Plan (query/search/vsearch)

## Goal
For each provider (`local`, `voyage`, `openai`), run one complete CLI flow:

1. `npm run qmd -- collection rm provider-e2e-docs`
2. `npm run qmd -- collection add <fixture-dir> --name provider-e2e-docs`
3. `npm run qmd -- embed -f`
4. `npm run qmd -- search --json -n 5 "新概念"`
5. `npm run qmd -- vsearch --json -n 5 "新概念"`
6. `npm run qmd -- query --json -n 5 "新概念"`

`<fixture-dir>` is created by the test under `test/.tmp/provider-e2e-fixtures-<timestamp>/` and deleted in cleanup.

## Provider Environment Matrix

### local
- `QMD_PROVIDER=local`

### voyage
- `QMD_PROVIDER=voyage`
- Required: `QMD_VOYAGE_API_KEY`
- Optional:
  - `QMD_VOYAGE_API_BASE` (default: `https://api.voyageai.com/v1`)
  - `QMD_VOYAGE_EMBED_MODEL`
  - `QMD_VOYAGE_RERANK_MODEL`

### openai
- `QMD_PROVIDER=openai`
- Required (either one):
  - `QMD_OPENAI_API_KEY`
  - `QMD_OPENAI_API_BASE` (for OpenAI-compatible local/hosted endpoint)
- Optional:
  - `QMD_OPENAI_EMBED_MODEL`
  - `QMD_OPENAI_RERANK_MODEL`

## Preconditions
1. `npm install` complete.
2. Provider environment variables are set before each run.
3. For remote providers, network access to the configured endpoint is available.

## Pass / Fail / Skip Rules

### PASS
- Flow commands complete with exit code `0`.
- `embed -f` writes vectors to `content_vectors` (count > 0).

### FAIL
- Required command exits non-zero (except tolerated `collection rm` not-found).
- `embed -f` succeeded but vectors were not written.

### SKIP
- `voyage` or `openai` credentials missing.
- Remote run disabled (default). Set `QMD_E2E_RUN_REMOTE=1` to execute `voyage/openai`.

## Automation Notes
- Vitest test file: `test/provider-e2e-query-search-vsearch.test.ts`
- Logs: `test/.tmp/provider-e2e-<timestamp>/<provider>/<step>.{stdout,stderr}.log`
- Tests run against isolated `INDEX_PATH` and `QMD_CONFIG_DIR` to avoid mutating user default index.
- Test fixture markdown files are generated under `test/.tmp/...` and cleaned up in `afterAll`.
