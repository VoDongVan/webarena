#!/usr/bin/env python3
"""
Analyze WebArena baseline results.

Usage:
    python memorybank/analyze_results.py [results_dir] [--json output.json] [--csv output.csv]

Default results_dir: memorybank/results_baseline_27b
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJ = Path(__file__).parent.parent
LOG_DIR = PROJ / "memorybank" / "logs"
CONFIG_DIR = PROJ / "config_files"


# ---------------------------------------------------------------------------
# Parse log files → {task_id: "PASS" | "FAIL"}
# ---------------------------------------------------------------------------

def parse_logs(log_dir: Path) -> dict[int, str]:
    results: dict[int, str] = {}
    log_files = sorted(log_dir.glob("wa_*.out"))
    for log_path in log_files:
        last_task = None
        for line in log_path.read_text(errors="replace").splitlines():
            m = re.search(r"\[Config file\].*?/(\d+)\.json", line)
            if m:
                last_task = int(m.group(1))
            m = re.search(r"\[Result\] \((PASS|FAIL)\)", line)
            if m and last_task is not None:
                results[last_task] = m.group(1)
    return results



# ---------------------------------------------------------------------------
# Parse render HTML → step count + stop reason
# ---------------------------------------------------------------------------

def parse_render(html_path: Path) -> dict:
    text = html_path.read_text(errors="replace")

    steps = text.count("<h2>New Page</h2>")

    # Last parsed_action div contains the final action
    stop_reason = None
    stop_answer = None
    # Find the last stop action
    stop_matches = re.findall(
        r"ActionTypes\.STOP[^}]*'answer':\s*'([^']*)'", text
    )
    if stop_matches:
        stop_answer = stop_matches[-1]
        if stop_answer.startswith("Early stop:"):
            stop_reason = stop_answer
        else:
            stop_reason = "agent_stop"

    return {"steps": steps, "stop_reason": stop_reason, "stop_answer": stop_answer}


# ---------------------------------------------------------------------------
# Load task config → sites, intent, eval_types
# ---------------------------------------------------------------------------

def load_task_config(task_id: int) -> dict:
    cfg_path = CONFIG_DIR / f"{task_id}.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        cfg = json.load(f)
    return {
        "sites": cfg.get("sites", []),
        "intent": cfg.get("intent", ""),
        "eval_types": cfg.get("eval", {}).get("eval_types", []),
        "intent_template_id": cfg.get("intent_template_id"),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(results_dir: Path) -> list[dict]:
    log_results = parse_logs(LOG_DIR)

    render_files = sorted(
        results_dir.glob("render_*.html"),
        key=lambda p: int(re.search(r"render_(\d+)\.html", p.name).group(1)),
    )

    records = []
    for html_path in render_files:
        task_id = int(re.search(r"render_(\d+)\.html", html_path.name).group(1))

        outcome = log_results.get(task_id, "UNKNOWN")

        # A render file exists but no [Result] log entry means the evaluator
        # crashed on the same run that produced the render file (the exception
        # is caught before the [Result] line is logged, but after render_helper
        # is initialised so close() still runs).  This is the only reliable
        # signal for a true eval failure — error.txt is cumulative across all
        # job runs and conflates prior-run setup failures with eval failures.
        has_eval_error = (outcome == "UNKNOWN")

        render_info = parse_render(html_path)
        task_cfg = load_task_config(task_id)

        records.append({
            "task_id": task_id,
            "outcome": outcome,             # PASS / FAIL / UNKNOWN
            "has_eval_error": has_eval_error,
            "steps": render_info["steps"],
            "stop_reason": render_info["stop_reason"],
            "stop_answer": render_info["stop_answer"],
            "sites": task_cfg.get("sites", []),
            "intent": task_cfg.get("intent", ""),
            "eval_types": task_cfg.get("eval_types", []),
            "intent_template_id": task_cfg.get("intent_template_id"),
        })

    return records


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(records: list[dict]) -> None:
    total = len(records)
    passed = sum(1 for r in records if r["outcome"] == "PASS")
    failed = sum(1 for r in records if r["outcome"] == "FAIL")
    unknown = sum(1 for r in records if r["outcome"] == "UNKNOWN")
    with_eval_error = sum(1 for r in records if r["has_eval_error"])

    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"  Tasks analyzed : {total}")
    print(f"  PASS           : {passed}  ({100*passed/total:.1f}%)")
    print(f"  FAIL           : {failed}  ({100*failed/total:.1f}%)")
    if unknown:
        print(f"  UNKNOWN        : {unknown}")
    print(f"  Eval errors    : {with_eval_error}  ({100*with_eval_error/total:.1f}%)")

    # Steps stats
    step_counts = [r["steps"] for r in records]
    print(f"\n  Avg steps/task : {sum(step_counts)/len(step_counts):.1f}")
    print(f"  Min steps      : {min(step_counts)}")
    print(f"  Max steps      : {max(step_counts)}")

    # Stop reason breakdown
    stop_counts: dict[str, int] = defaultdict(int)
    for r in records:
        stop_counts[r["stop_reason"] or "unknown"] += 1
    print("\n  Stop reasons:")
    for reason, count in sorted(stop_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason:<45} {count:>4}  ({100*count/total:.1f}%)")

    # Per-site breakdown
    site_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "pass": 0, "eval_error": 0})
    for r in records:
        for site in r["sites"]:
            site_stats[site]["total"] += 1
            if r["outcome"] == "PASS":
                site_stats[site]["pass"] += 1
            if r["has_eval_error"]:
                site_stats[site]["eval_error"] += 1

    print("\n" + "=" * 60)
    print("PER-SITE BREAKDOWN")
    print("=" * 60)
    print(f"  {'Site':<20} {'Total':>6} {'Pass':>6} {'Pass%':>7} {'EvalErr':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*7} {'-'*8}")
    for site, s in sorted(site_stats.items()):
        t, p, e = s["total"], s["pass"], s["eval_error"]
        print(f"  {site:<20} {t:>6} {p:>6} {100*p/t:>6.1f}% {e:>8}")

    # Per eval_type breakdown
    eval_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "pass": 0})
    for r in records:
        for et in r["eval_types"]:
            eval_stats[et]["total"] += 1
            if r["outcome"] == "PASS":
                eval_stats[et]["pass"] += 1

    print("\n" + "=" * 60)
    print("PER EVAL-TYPE BREAKDOWN")
    print("=" * 60)
    print(f"  {'Eval type':<30} {'Total':>6} {'Pass':>6} {'Pass%':>7}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*7}")
    for et, s in sorted(eval_stats.items()):
        t, p = s["total"], s["pass"]
        print(f"  {et:<30} {t:>6} {p:>6} {100*p/t:>6.1f}%")

    # Per eval-type error breakdown
    # For each eval_type, count how many tasks had an eval error
    eval_error_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "eval_error": 0})
    for r in records:
        for et in r["eval_types"]:
            eval_error_stats[et]["total"] += 1
            if r["has_eval_error"]:
                eval_error_stats[et]["eval_error"] += 1

    print("\n" + "=" * 60)
    print("PER EVAL-TYPE ERROR BREAKDOWN")
    print("=" * 60)
    print(f"  {'Eval type':<30} {'Total':>6} {'EvalErr':>8} {'EvalErr%':>9}")
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*9}")
    for et, s in sorted(eval_error_stats.items()):
        t, e = s["total"], s["eval_error"]
        print(f"  {et:<30} {t:>6} {e:>8} {100*e/t:>8.1f}%")

    # Passed tasks list
    passed_ids = sorted(r["task_id"] for r in records if r["outcome"] == "PASS")
    print("\n" + "=" * 60)
    print(f"PASSED TASKS ({len(passed_ids)})")
    print("=" * 60)
    print("  " + ", ".join(map(str, passed_ids)))

    # Tasks with eval errors
    error_ids = sorted(r["task_id"] for r in records if r["has_eval_error"])
    print("\n" + "=" * 60)
    print(f"TASKS WITH EVAL ERRORS ({len(error_ids)})")
    print("=" * 60)
    print("  " + ", ".join(map(str, error_ids)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze WebArena results")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=str(PROJ / "memorybank" / "results_baseline_27b"),
        help="Path to results directory (default: memorybank/results_baseline_27b)",
    )
    parser.add_argument("--json", metavar="FILE", help="Save per-task records as JSON")
    parser.add_argument("--csv", metavar="FILE", help="Save per-task records as CSV")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing: {results_dir}")
    records = analyze(results_dir)
    print(f"Found {len(records)} render files\n")

    print_summary(records)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\nPer-task JSON saved to: {args.json}")

    if args.csv:
        import csv
        fieldnames = [
            "task_id", "outcome", "has_eval_error",
            "steps", "stop_reason", "stop_answer", "sites",
            "intent", "eval_types", "intent_template_id",
        ]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                row = dict(r)
                row["sites"] = "|".join(r["sites"])
                row["eval_types"] = "|".join(r["eval_types"])
                writer.writerow(row)
        print(f"Per-task CSV saved to: {args.csv}")


if __name__ == "__main__":
    main()
