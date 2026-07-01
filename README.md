# Easy Chinese Model Router

A one-file router for cheap, capable Chinese LLMs — DeepSeek, Qwen, Kimi,
GLM, MiniMax, MiMo (Xiaomi), inclusionAI (Ant Group), Hunyuan (Tencent) and
Seed (ByteDance) — via [OpenRouter](https://openrouter.ai). It classifies
each prompt, picks the best-fit model for the task and budget — using
benchmark-grounded quality and strength data, not guesses — and falls back
through the remaining candidates on errors.

Use it three ways:

- **CLI** — `python router.py --prompt "..."`
- **Python library** — `EasyChineseModelRouter().chat(...)`
- **OpenAI-compatible server** — `python server.py`, then point OpenCode,
  Cline, Zed, or any OpenAI SDK at it ([guide below](#use-with-opencode))

Routing is **auditable** and, by default, **deterministic and offline**:
classification is keyword/heuristic based (English + Chinese patterns), and
`--dry-run` explains every decision without a single network call. For better
accuracy, `--classifier llm` labels the task with a cheap model instead
(~50-token call, falls back to keywords on any error).

## Quick start

```bash
pip install -r requirements.txt

# .env in the repo root:
#   OPENROUTER_API_KEY=sk-or-...

python router.py --prompt "fix this Python bug" --dry-run   # see the decision
python router.py --prompt "write a FastAPI endpoint"        # actually chat
```

## How routing works

1. **Classify** the prompt into a task kind — `simple`, `coding`, `reasoning`,
   `long_context`, `translation`, or `creative` — using bilingual (EN + CJK)
   keyword patterns and a CJK-aware token estimate. With `--classifier llm`,
   a cheap model does the labeling instead (long-context is always decided
   locally from the token estimate; keyword fallback on any error).
2. **Route**: filter the model table by context window, allowed families, and
   tool support, then sort by a mode-dependent blend of cost and quality
   scores, with a bonus for models whose strengths match the task. Strengths
   are grounded in Artificial Analysis per-domain benchmarks — e.g. DeepSeek's
   1M-token windows don't earn it a long-context preference (worst cohort
   AA-LCR), and its 4% non-hallucination rate disqualifies it from factual
   Q&A, which routes to MiMo/MiniMax instead.
3. **Chat** with the top model; on retryable errors (429/5xx/timeouts) retry
   with backoff, on hard errors fall through to the next model in the route.
   Reasoning is enabled for coding/reasoning/long-context tasks and explicitly
   disabled for light ones — the catalog's models all think by default, and
   hidden thinking tokens can dominate the cost of a simple request.

Modes: `auto` (default; cheap for simple tasks, stronger for coding/reasoning),
`cheap`, `balanced`, `quality`.

## CLI usage

```bash
python router.py --prompt "..." [options]

--mode auto|cheap|balanced|quality   # routing strategy (default: auto)
--classifier keyword|llm             # task detection (default: keyword/offline)
--classifier-model <slug>            # model for --classifier llm
--live-pricing                       # cost scores from live OpenRouter prices
--log-file requests.jsonl            # per-request model/latency/cost log
--family deepseek --family qwen      # restrict families (repeatable)
--stream                             # stream tokens as they arrive
--dry-run                            # print the routing decision, no network
--session-id my-chat                 # pin follow-up turns to the same model
--history msgs.json                  # prior turns as a JSON message list
--list-models                        # print the built-in model table
--validate-models                    # check slugs against the live catalog
--max-tokens 2048  --temperature 0.2
```

## Models

15 current-generation models across 9 families, spanning flagship routes down
to a $0.01/M ultra-budget tier. Catalog-validated and benchmark-calibrated
2026-07-02 (AA =
[Artificial Analysis Intelligence Index](https://artificialanalysis.ai/leaderboards/models) v4.1):

| Model | AA index | Routed for | Notes |
|---|---|---|---|
| `z-ai/glm-5.2` | 51 (#1 open-weights) | reasoning, coding, long context | #1 cohort coding + agentic |
| `qwen/qwen3.7-max` | 46 | coding, reasoning, creative | #2 cohort coding |
| `deepseek/deepseek-v4-pro` | 44 | reasoning, coding | #2 cohort agentic |
| `minimax/minimax-m3` | 44 | simple, reasoning, long context, creative | #1 cohort AA-LCR + non-hallucination |
| `moonshotai/kimi-k2.7-code` | 42 | coding | coding specialist (Coding Index 60.8, #3 cohort) |
| `xiaomi/mimo-v2.5-pro` | 42 | coding, long context | weak reasoning benchmarks |
| `deepseek/deepseek-v4-flash` | 40 | coding, translation | heavy hallucinator on facts |
| `xiaomi/mimo-v2.5` | 40 (est.) | simple, coding, reasoning, long context | budget default |
| `qwen/qwen3.7-plus` | 39 | simple, translation, creative, long context | budget tier |
| `bytedance-seed/seed-2.0-lite` | unverified | simple | stronger Seed route |
| `bytedance-seed/seed-2.0-mini` | unverified | simple | cheap Seed tier |
| `tencent/hy3-preview` | 34 (AA est.) | simple | preview build, very cheap |
| `inclusionai/ring-2.6-1t` | 31 | coding | Coding Index 42.8; budget coding fallback |
| `inclusionai/ling-2.6-1t` | 26 (AA est.) | — (fallback only) | non-reasoning twin of ring |
| `inclusionai/ling-2.6-flash` | unverified | simple, translation | $0.01/M in — cheapest route |

The table lives at the top of `router.py` (`MODELS`) and is meant to be
edited: slugs drift (`--validate-models` checks them against the live
catalog). All three routing axes are grounded in data: `quality_score` is
calibrated to the AA index (`7.6 + 0.1 * (index - 40)`), `strengths` to AA's
per-domain charts (Coding/Agentic Index, AA-LCR, GPQA/HLE/CritPt,
AA-Omniscience, IFBench), and with `--live-pricing` the cost scores are
derived from OpenRouter's real price list (`1 + log2(price/cheapest)`, cached
to disk daily, stale-tolerant offline). The three "unverified" models (both
Seeds and ling-2.6-flash) are not yet covered by Artificial Analysis; they
carry conservative estimated scores and minimal strengths until benchmarks
land. Candidates not yet added are tracked in `MODELS_TODO.md`.

With `--log-file`, every completed request appends a JSONL line with the
routed model, task, latency, token counts (including hidden reasoning tokens)
and the actual cost reported by OpenRouter — so the router's spending is
auditable after the fact, not just its decisions.

## OpenAI-compatible server

```bash
python server.py                     # http://127.0.0.1:8787/v1
python server.py --port 9000 --family deepseek --family qwen
python server.py --classifier llm    # LLM-based task detection (see above)
python server.py --live-pricing --log-file requests.jsonl
```

Endpoints:

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Streaming (SSE) and non-streaming; `tools` / `tool_calls` passed through |
| `GET /v1/models` | The four routing modes, presented as models |
| `GET /health` | Liveness check |

The client requests a *mode* as its "model" (`auto`, `cheap`, `balanced`,
`quality`; provider-prefixed ids like `myrouter/auto` also work), and the
router picks the actual model per request. Your OpenRouter key stays on the
server — clients never see it. Each conversation is pinned to its
first-routed model (best-effort, keyed by a hash of the first user message) so
provider prompt caches stay warm across agent turns.

Quick check with curl:

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "hello"}]}'
```

### Use with OpenCode

1. Start the server: `python server.py`
2. Add the provider to your `opencode.json` (project-level, or global at
   `~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "myrouter": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Chinese Model Router",
      "options": { "baseURL": "http://127.0.0.1:8787/v1" },
      "models": {
        "auto":     { "name": "Router (auto)" },
        "cheap":    { "name": "Router (cheap)" },
        "balanced": { "name": "Router (balanced)" },
        "quality":  { "name": "Router (quality)" }
      }
    }
  },
  "model": "myrouter/auto"
}
```

3. Launch `opencode`. Use `/models` inside OpenCode to switch between the
   router modes.

Notes:

- The `/v1` suffix in `baseURL` is required — OpenCode expects an
  OpenAI-compatible endpoint.
- Setting the top-level `"model"` matters; leaving it blank causes confusing
  errors in some OpenCode versions.
- No API key is needed on the OpenCode side (the server holds the OpenRouter
  key). If your OpenCode version insists on one, set any placeholder value.

The same provider block works for Cline, Zed, and anything else that speaks
the OpenAI chat-completions protocol — only the config file differs.

## Evals

An offline harness proves the classifier works and guards against regressions:

```bash
python evals/run_eval.py                    # accuracy, confusion, mode comparison
python evals/run_eval.py --json             # machine-readable report
python evals/run_eval.py --update-baseline  # record the new baseline
```

The labeled set (`evals/prompts.jsonl`, EN + ZH + mixed across all six task
kinds) currently scores 98.1% classification accuracy. The harness exits
non-zero if accuracy drops below `--min-accuracy` (default 90%) or below the
recorded baseline.

## Tests

```bash
python -m unittest discover -s tests -v
```

No test touches the network; the server tests run against a stub router on
localhost.

## Configuration

| Env var | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Required. Your OpenRouter key (put it in `.env`) |
| `OPENROUTER_REFERER` | Optional. Sent as `HTTP-Referer` for attribution |
| `OPENROUTER_APP_NAME` | Optional. Sent as `X-Title` for attribution |

## Repo layout

```
router.py            # model table, classifier, router, OpenRouter client, CLI
server.py            # OpenAI-compatible HTTP server (stdlib only)
evals/               # labeled prompt set + offline eval harness
tests/               # unit + HTTP integration tests (no network)
FEATURES_TODO.md     # feature backlog and completed-feature notes
MODELS_TODO.md       # model families still to add
```
