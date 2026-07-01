# SOTA feature backlog

Candidate upgrades to make the router state-of-the-art. Captured 2026-06-30 for
later; nothing here is implemented yet.

**Standing constraints (from earlier decisions):**
- No LLM-in-the-loop router (don't pay an LLM call to decide routing per request).
- No `openrouter/auto` delegation.
- Keep routing **deterministic, offline, and auditable** (`--dry-run` must still
  explain every decision).

---

## 1. Smarter classification (semantic, still local)

Replace/augment the keyword regex in `classify()` with **local semantic
embeddings** (e.g. a small sentence-transformer) for intent detection. Stays
deterministic and offline — no per-call LLM — but far more robust to phrasing,
typos, and mixed CN/EN than keyword matching.

```
prompt -> embed locally -> nearest task centroid (cosine sim)
       -> confidence score -> route
       -> fall back to regex if the embedder is unavailable
```

- Adds a confidence score, enabling "low-confidence -> safer/stronger model".
- Trade-off: a model dependency + startup load time; keep regex as fallback.

## 2. Calibrate scores to reality

Replace the subjective `cost_score` / `quality_score` heuristics with real data:

- `cost_score` <- **live $/Mtok** from the OpenRouter pricing API.
- `quality_score` <- **published benchmarks** (coding/reasoning leaderboards,
  LMArena, etc.), ideally **per-task** rather than one global number.

Makes routing reflect actual cost and capability instead of hand-tuned guesses.

## 3. Eval & benchmark harness — DONE (2026-07-01)

Implemented as `evals/run_eval.py` + `evals/prompts.jsonl` (52 labeled EN/ZH/mixed
prompts across all six task kinds). Fully offline and deterministic — no network.

```
python evals/run_eval.py                    # accuracy, confusion, mode comparison
python evals/run_eval.py --json             # machine-readable report
python evals/run_eval.py --update-baseline  # record accuracy in evals/baseline.json
```

- Reports overall / per-language / per-task classification accuracy plus a
  misclassification list and a per-mode routing comparison (mean cost/quality
  score of the selected model).
- Regression guard: exits non-zero if accuracy drops below `--min-accuracy`
  (default 90%) or below the recorded baseline (currently 98.1%).
- Wired into the unit suite via `tests/test_eval.py` (accuracy floors, dataset
  coverage, mode-ordering invariants).
- Not covered (needs live calls, deferred): $/req and p50/p95 latency measurement.

## 4. Production hardening

Speed and reliability features:

- **Latency-aware routing** — factor measured latency into the score.
- **Parallel race** of top-N models, first good response wins (opt-in; costs more).
- **Response caching** keyed by prompt hash to skip repeat calls.
- **Structured metrics/logging** — model, latency, tokens, cost per request.

---

## Open questions to resolve before starting

1. What does "SOTA" prioritize: best routing *decisions*, best *models*, or best
   *engineering/production-readiness*?
2. Do the standing constraints (no LLM-in-loop, no `openrouter/auto`) still hold?
3. One focused upgrade, or level up across the board?

See also: [MODELS_TODO.md](MODELS_TODO.md) for missing model families.
