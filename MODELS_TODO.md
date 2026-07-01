# Candidate models to add later

Chinese-origin LLM families available on OpenRouter (catalog snapshot 2026-07-01,
338 total models) that the router does **not** currently cover. Verify slugs with
`py router.py --validate-models` after adding any of these — slugs drift.

> Note: `cost_score` / `quality_score` in `router.py` are subjective heuristics,
> not benchmarked. Assign them deliberately when adding a family.

## New families (not in the table at all)

| Family | Slug(s) | Status / notes |
|---|---|---|
| **inclusionAI (Ant Group)** | `inclusionai/ling-2.6-flash`, `inclusionai/ring-2.6-1t`, `inclusionai/ling-2.6-1t` | Ling/Ring line. `ling-2.6-flash` is $0.01/M in — 9x cheaper than deepseek-v4-flash, cheapest Chinese model on the catalog. Not yet in AA benchmarks; add once quality is measurable. |
| **ByteDance Seed** | `bytedance-seed/seed-2.0-lite`, `bytedance-seed/seed-2.0-mini` | Real general chat models, cheap ($0.10–0.25/M in), 262k ctx. Not yet in AA benchmarks. |
| **StepFun** | `stepfun/step-3.7-flash`, `stepfun/step-3.5-flash` | Cheap/fast general models. |
| **Tencent Hunyuan** | `tencent/hy3-preview`, `tencent/hunyuan-a13b-instruct` | `hy3-preview` is newest but still a preview build. |
| **Baidu ERNIE** | `baidu/ernie-4.5-vl-424b-a47b` | Still only a vision (VL) variant exposed; wait for a standard chat build. |

## Recommendation when revisiting

Policy: add a model only once Artificial Analysis publishes index/per-domain
scores for it, so quality_score and strengths stay benchmark-grounded
(2026-07-01 decision).

1. Watch **inclusionAI** (`ling-2.6-flash`) — at $0.01/M in, any reasonable AA
   score makes it the new budget tier.
2. Then **ByteDance Seed** (`seed-2.0-lite`/`seed-2.0-mini`) and **StepFun**
   (`step-3.7-flash`).
3. Consider **Hunyuan** (`hy3-preview`) once it leaves preview.
4. Hold **ERNIE** until a non-VL chat model ships.

## Also noted (within existing families, not yet added)

- Qwen coding-specialized: `qwen/qwen3-coder-plus` / `qwen/qwen3-coder-next` — could strengthen the `coding` route.
- ~~Qwen mid-tier: `qwen/qwen3.6-plus` / `qwen/qwen3.7-plus`~~ — DONE: `qwen3.7-plus` replaced `qwen3.6-flash` as the Qwen budget tier (2026-07-01).
- GLM cheap tier: `z-ai/glm-5-turbo` ($1.20/M in, 262k ctx) is the current-gen option; `glm-4.7-flash` is cheaper ($0.06/M) but previous-gen, which conflicts with the current-gen-only policy. GLM vision: `z-ai/glm-5v-turbo`.
- MiniMax budget tier: `minimax/minimax-m2.7` ($0.18/M in) if a cheap MiniMax slot is ever wanted; previous-gen though.
