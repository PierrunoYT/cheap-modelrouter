"""Offline eval harness for the router (FEATURES_TODO item 3).

Runs the labeled prompt set through ``classify()`` and ``route()`` with no
network calls, then reports:

  - classification accuracy (overall, per language, per task)
  - a confusion matrix of expected vs predicted task kinds
  - routing comparison across modes (selected model + mean cost/quality scores)

Acts as a regression guard: exits non-zero if accuracy falls below
``--min-accuracy`` or below the recorded baseline (evals/baseline.json).

Usage:
  python evals/run_eval.py                    # human-readable report
  python evals/run_eval.py --json             # machine-readable report
  python evals/run_eval.py --update-baseline  # record current accuracy as baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router import EasyChineseModelRouter, Mode  # noqa: E402

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(EVALS_DIR, "prompts.jsonl")
DEFAULT_BASELINE = os.path.join(EVALS_DIR, "baseline.json")

MODES: list[Mode] = ["auto", "cheap", "balanced", "quality"]


def load_cases(path: str = DEFAULT_DATASET) -> list[dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if "prompt" not in case or "expected" not in case:
                raise ValueError(f"{path}:{line_no}: needs 'prompt' and 'expected'")
            # "repeat" lets the dataset stay small while still exercising the
            # long-context threshold with synthetic large prompts.
            case["prompt"] = case["prompt"] * int(case.get("repeat", 1))
            cases.append(case)
    return cases


def evaluate(cases: list[dict]) -> dict:
    router = EasyChineseModelRouter()

    per_lang: dict[str, Counter] = defaultdict(Counter)
    per_task: dict[str, Counter] = defaultdict(Counter)
    confusion: dict[str, Counter] = defaultdict(Counter)
    failures: list[dict] = []

    for case in cases:
        expected = case["expected"]
        predicted = router.classify(case["prompt"])
        lang = case.get("lang", "unknown")
        correct = predicted == expected

        per_lang[lang]["total"] += 1
        per_task[expected]["total"] += 1
        confusion[expected][predicted] += 1
        if correct:
            per_lang[lang]["correct"] += 1
            per_task[expected]["correct"] += 1
        else:
            failures.append(
                {
                    "prompt": case["prompt"][:80],
                    "expected": expected,
                    "predicted": predicted,
                    "lang": lang,
                }
            )

    total = len(cases)
    correct = total - len(failures)

    # Mode comparison: same prompts, how does each mode route, and what does
    # that imply for relative cost/quality? Uses the heuristic scores from the
    # model table -- deterministic and offline, like everything else here.
    mode_stats: dict[str, dict] = {}
    for mode in MODES:
        mode_router = EasyChineseModelRouter(mode=mode)
        selections: Counter = Counter()
        cost_sum = 0.0
        quality_sum = 0.0
        for case in cases:
            top = mode_router.route(case["prompt"])[0]
            selections[top.name] += 1
            cost_sum += top.cost_score
            quality_sum += top.quality_score
        mode_stats[mode] = {
            "mean_cost_score": round(cost_sum / total, 3),
            "mean_quality_score": round(quality_sum / total, 3),
            "selections": dict(selections.most_common()),
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "per_lang": {
            lang: {
                "total": c["total"],
                "correct": c["correct"],
                "accuracy": round(c["correct"] / c["total"], 4),
            }
            for lang, c in sorted(per_lang.items())
        },
        "per_task": {
            task: {
                "total": c["total"],
                "correct": c["correct"],
                "accuracy": round(c["correct"] / c["total"], 4),
            }
            for task, c in sorted(per_task.items())
        },
        "confusion": {task: dict(preds) for task, preds in sorted(confusion.items())},
        "failures": failures,
        "modes": mode_stats,
    }


def print_report(report: dict) -> None:
    print(f"Classification accuracy: {report['correct']}/{report['total']} "
          f"({report['accuracy']:.1%})")

    print("\nBy language:")
    for lang, stats in report["per_lang"].items():
        print(f"  {lang:8} {stats['correct']:>3}/{stats['total']:<3} ({stats['accuracy']:.1%})")

    print("\nBy task:")
    for task, stats in report["per_task"].items():
        print(f"  {task:14} {stats['correct']:>3}/{stats['total']:<3} ({stats['accuracy']:.1%})")

    if report["failures"]:
        print("\nMisclassifications:")
        for f in report["failures"]:
            print(f"  [{f['lang']}] expected={f['expected']} predicted={f['predicted']}"
                  f"  {f['prompt']!r}")

    print("\nMode comparison (mean scores of the selected model, lower cost / higher quality):")
    for mode, stats in report["modes"].items():
        top = next(iter(stats["selections"]))
        print(f"  {mode:9} cost={stats['mean_cost_score']:<5} "
              f"quality={stats['mean_quality_score']:<5} most_selected={top}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline router eval harness")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--json", action="store_true", help="Print full report as JSON")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.9,
        help="Fail (exit 1) if overall accuracy drops below this (default: 0.9)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Record the current accuracy as the regression baseline",
    )
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    report = evaluate(cases)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report)

    if args.update_baseline:
        with open(args.baseline, "w", encoding="utf-8") as fh:
            json.dump({"accuracy": report["accuracy"], "total": report["total"]}, fh, indent=2)
        print(f"\nBaseline updated: accuracy={report['accuracy']:.1%}")
        return 0

    failed = False
    if report["accuracy"] < args.min_accuracy:
        print(f"\nFAIL: accuracy {report['accuracy']:.1%} < required {args.min_accuracy:.1%}",
              file=sys.stderr)
        failed = True

    if os.path.exists(args.baseline):
        with open(args.baseline, "r", encoding="utf-8") as fh:
            baseline = json.load(fh)
        if report["accuracy"] < baseline["accuracy"]:
            print(f"\nFAIL: accuracy {report['accuracy']:.1%} regressed below baseline "
                  f"{baseline['accuracy']:.1%} (use --update-baseline if intentional)",
                  file=sys.stderr)
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
