# Candidate models to add later

Chinese-origin LLM families available on OpenRouter (catalog snapshot 2026-07-01,
338 total models) that the router does **not** currently cover. Verify slugs with
`py router.py --validate-models` after adding any of these — slugs drift.

> Note: `cost_score` / `quality_score` in `router.py` are subjective heuristics,
> not benchmarked. Assign them deliberately when adding a family.

## New families (not in the table at all)

| Family | Slug(s) | Status / notes |
|---|---|---|
| ~~inclusionAI~~ | `inclusionai/ling-2.6-flash`, `inclusionai/ring-2.6-1t` | ADDED 2026-07-01. Ring scored from AA per-eval charts (Coding 42.8, Agentic 18.9); `ling-2.6-1t` skipped — AA-LCR 35% and no niche vs ring. |
| ~~ByteDance Seed~~ | `bytedance-seed/seed-2.0-lite`, `-mini` | ADDED 2026-07-01 (unverified scores). |
| ~~Tencent Hunyuan~~ | `tencent/hy3-preview` | ADDED 2026-07-01 (unverified scores; still a preview build). |
| **StepFun** | `stepfun/step-3.7-flash`, `stepfun/step-3.5-flash` | Cheap/fast general models. |
| **Baidu ERNIE** | `baidu/ernie-4.5-vl-424b-a47b` | Still only a vision (VL) variant exposed; wait for a standard chat build. |

## Recommendation when revisiting

The 2026-07-01 additions carry conservative unverified quality scores and
minimal strengths ("simple") — recalibrate them against Artificial Analysis
as soon as they appear in the index, like the rest of the table.

1. Recalibrate **inclusionAI / Seed / Hunyuan** when AA benchmarks land.
2. Consider **StepFun** (`step-3.7-flash`) next.
3. Hold **ERNIE** until a non-VL chat model ships.

## Also noted (within existing families, not yet added)

- Qwen coding-specialized: `qwen/qwen3-coder-plus` / `qwen/qwen3-coder-next` — could strengthen the `coding` route.
- ~~Qwen mid-tier: `qwen/qwen3.6-plus` / `qwen/qwen3.7-plus`~~ — DONE: `qwen3.7-plus` replaced `qwen3.6-flash` as the Qwen budget tier (2026-07-01).
- GLM cheap tier: `z-ai/glm-5-turbo` ($1.20/M in, 262k ctx) is the current-gen option; `glm-4.7-flash` is cheaper ($0.06/M) but previous-gen, which conflicts with the current-gen-only policy. GLM vision: `z-ai/glm-5v-turbo`.
- MiniMax budget tier: `minimax/minimax-m2.7` ($0.18/M in) if a cheap MiniMax slot is ever wanted; previous-gen though.
