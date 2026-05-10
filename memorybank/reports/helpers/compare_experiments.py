#!/usr/bin/env python3
"""
Compare multiple WebArena experiment result directories side-by-side.

Usage:
    python compare_experiments.py DIR[:LABEL] [DIR[:LABEL] ...] [options]

    DIR    — path to a results directory (absolute or relative to repo root)
    LABEL  — display name for the experiment (defaults to the directory name)

Options:
    --output FILE    write a Markdown report to FILE
    --model TEXT     model name to embed in the report header
    --notes TEXT     freeform notes to append at the end of the report

Examples:
    # Console comparison of two experiments
    python compare_experiments.py \\
        memorybank/results_baseline_27b:Baseline \\
        memorybank/results_memory_bm25_27b_300tasks:BM25

    # Three-way comparison saved to a Markdown file
    python compare_experiments.py \\
        memorybank/results_baseline_27b:Baseline \\
        memorybank/results_memory_bm25_27b_300tasks:BM25 \\
        memorybank/results_memory_dense_27b_300tasks:Dense \\
        --output memorybank/reports/my_comparison.md \\
        --model "Qwen3-30B-A3B-FP8"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo root and shared helpers
# ---------------------------------------------------------------------------

PROJ = Path(__file__).resolve().parents[3]   # .../webarena
CONFIG_DIR = PROJ / "config_files"

# Reuse load_outcomes from analyze_results.py so outcome-parsing logic stays
# in one place.
sys.path.insert(0, str(PROJ / "memorybank"))
from analyze_results import load_outcomes, analyze  # noqa: E402


# ---------------------------------------------------------------------------
# Task config cache
# ---------------------------------------------------------------------------

_task_cfg_cache: dict[int, dict] = {}

def task_cfg(task_id: int) -> dict:
    if task_id not in _task_cfg_cache:
        p = CONFIG_DIR / f"{task_id}.json"
        if p.exists():
            try:
                cfg = json.loads(p.read_text())
                _task_cfg_cache[task_id] = {
                    "site": cfg.get("sites", ["unknown"])[0],
                    "eval_types": cfg.get("eval", {}).get("eval_types", []),
                }
            except Exception:
                _task_cfg_cache[task_id] = {"site": "unknown", "eval_types": []}
        else:
            _task_cfg_cache[task_id] = {"site": "unknown", "eval_types": []}
    return _task_cfg_cache[task_id]


# ---------------------------------------------------------------------------
# Memory trace statistics
# ---------------------------------------------------------------------------

def memory_trace_stats(results_dir: Path) -> dict | None:
    trace_dir = results_dir / "memory_traces"
    if not trace_dir.exists():
        return None

    tasks = 0
    total_calls = 0
    tasks_with_retrieval = 0
    nonempty_calls = 0

    for f in trace_dir.glob("*.json"):
        try:
            t = json.loads(f.read_text())
        except Exception:
            continue
        tasks += 1
        calls = t.get("calls", [])
        total_calls += len(calls)
        has_r = any(c.get("retrieved") for c in calls)
        if has_r:
            tasks_with_retrieval += 1
            nonempty_calls += sum(1 for c in calls if c.get("retrieved"))

    if tasks == 0:
        return None
    return {
        "tasks": tasks,
        "avg_calls": total_calls / tasks,
        "tasks_with_retrieval": tasks_with_retrieval,
        "retrieval_rate": tasks_with_retrieval / tasks,
        "nonempty_calls": nonempty_calls,
    }


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def pct(n: int, d: int, decimals: int = 1) -> str:
    return f"{100 * n / d:.{decimals}f}%" if d else "—"


def compare(experiments: list[tuple[Path, str]]) -> dict[str, Any]:
    """Load outcomes for each experiment and build comparison data."""
    exp_outcomes: dict[str, dict[int, str]] = {}
    exp_records:  dict[str, list[dict]]     = {}

    for results_dir, label in experiments:
        outcomes = load_outcomes(results_dir)
        records  = analyze(results_dir)
        exp_outcomes[label] = outcomes
        exp_records[label]  = records

    # Common task set — tasks present in every experiment
    common_ids = set.intersection(*(set(o.keys()) for o in exp_outcomes.values()))

    # Per-task pass sets
    pass_sets: dict[str, set[int]] = {
        label: {t for t in common_ids if outcomes[t] == "PASS"}
        for label, outcomes in exp_outcomes.items()
    }

    # Per-site breakdown
    sites = sorted({task_cfg(t)["site"] for t in common_ids})
    site_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t in common_ids:
        site = task_cfg(t)["site"]
        site_stats[site]["total"] += 1
        for label, outcomes in exp_outcomes.items():
            if outcomes.get(t) == "PASS":
                site_stats[site][label] += 1

    # Memory trace stats (only for experiments that have a traces dir)
    trace_stats: dict[str, dict | None] = {
        label: memory_trace_stats(results_dir)
        for results_dir, label in experiments
    }

    # Stop-reason and step stats from records
    stop_stats: dict[str, dict[str, int]] = {}
    step_stats: dict[str, dict] = {}
    for label, records in exp_records.items():
        common_records = [r for r in records if r["task_id"] in common_ids]
        stop_counts: dict[str, int] = defaultdict(int)
        steps = []
        for r in common_records:
            stop_counts[r["stop_reason"] or "unknown"] += 1
            steps.append(r["steps"])
        stop_stats[label] = dict(stop_counts)
        step_stats[label] = {
            "avg": sum(steps) / len(steps) if steps else 0,
            "min": min(steps) if steps else 0,
            "max": max(steps) if steps else 0,
        }

    return {
        "labels":      [label for _, label in experiments],
        "dirs":        {label: str(d) for d, label in experiments},
        "common_ids":  common_ids,
        "exp_outcomes": exp_outcomes,
        "pass_sets":   pass_sets,
        "site_stats":  site_stats,
        "sites":       sites,
        "trace_stats": trace_stats,
        "stop_stats":  stop_stats,
        "step_stats":  step_stats,
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_comparison(data: dict) -> None:
    labels     = data["labels"]
    common_ids = data["common_ids"]
    pass_sets  = data["pass_sets"]
    site_stats = data["site_stats"]
    trace_stats = data["trace_stats"]
    step_stats  = data["step_stats"]
    stop_stats  = data["stop_stats"]
    N = len(common_ids)

    print("=" * 65)
    print("EXPERIMENT COMPARISON")
    print("=" * 65)
    print(f"Common tasks: {N}\n")

    # Overall
    col = 10
    header = f"{'Experiment':<25}" + "".join(f"{l:>{col*2}}" for l in labels)
    print(header)
    subheader = f"{'':25}" + "".join(f"{'Pass':>{col}}{'Rate':>{col}}" for _ in labels)
    print(subheader)
    print("-" * (25 + col * 2 * len(labels)))
    for label in labels:
        n = len(pass_sets[label])
        print(f"{label:<25}" + f"{n:>{col}}{pct(n, N):>{col}}")

    # Delta vs first experiment
    baseline_label = labels[0]
    baseline_set   = pass_sets[baseline_label]
    if len(labels) > 1:
        print(f"\nDelta vs '{baseline_label}':")
        for label in labels[1:]:
            s = pass_sets[label]
            gained = len(s - baseline_set)
            lost   = len(baseline_set - s)
            net    = len(s) - len(baseline_set)
            print(f"  {label:<22}  +{gained} gained  -{lost} lost  (net {net:+d})")

    # Per-site
    print(f"\n{'Site':<20}" + "".join(f"{l:>12}" for l in labels) + f"{'N':>5}")
    print("-" * (20 + 12 * len(labels) + 5))
    for site in data["sites"]:
        s = site_stats[site]
        t = s["total"]
        row = f"  {site:<18}"
        for label in labels:
            row += f"{pct(s.get(label, 0), t):>12}"
        row += f"{t:>5}"
        print(row)

    # Memory stats
    has_traces = any(v is not None for v in trace_stats.values())
    if has_traces:
        print(f"\n{'Memory retrieval':}")
        print(f"  {'':30}" + "".join(f"{l:>12}" for l in labels))
        print("  " + "-" * (30 + 12 * len(labels)))
        metrics = [
            ("avg_calls",        "Avg calls/task",       ".1f"),
            ("retrieval_rate",   "Tasks w/ retrieval",   ".0%"),
            ("nonempty_calls",   "Non-empty calls",      "d"),
        ]
        for key, label_str, fmt in metrics:
            row = f"  {label_str:<30}"
            for exp_label in labels:
                ts = trace_stats.get(exp_label)
                if ts is None:
                    row += f"{'—':>12}"
                else:
                    val = ts[key]
                    if fmt == ".0%":
                        row += f"{val:>11.0%} "
                    elif fmt == "d":
                        row += f"{int(val):>12}"
                    else:
                        row += f"{val:>12.1f}"
            print(row)

    # Overlap (only when ≥2 experiments)
    if len(labels) >= 2:
        print("\nOverlap analysis:")
        all_pass = set.intersection(*pass_sets.values())
        none_pass = common_ids - set.union(*pass_sets.values())
        print(f"  All experiments pass : {len(all_pass)}")
        print(f"  No experiment passes : {len(none_pass)}")
        for label in labels:
            only = pass_sets[label] - set.union(*(pass_sets[l] for l in labels if l != label))
            print(f"  {label} only : {len(only)}  {sorted(only)}")

    # Step stats
    print(f"\n{'Avg steps/task':<20}" + "".join(f"{step_stats[l]['avg']:>12.1f}" for l in labels))


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def build_markdown(data: dict, model: str = "", notes: str = "") -> str:
    labels     = data["labels"]
    dirs       = data["dirs"]
    common_ids = data["common_ids"]
    pass_sets  = data["pass_sets"]
    site_stats = data["site_stats"]
    trace_stats = data["trace_stats"]
    step_stats  = data["step_stats"]
    stop_stats  = data["stop_stats"]
    N = len(common_ids)
    baseline_label = labels[0]
    baseline_set   = pass_sets[baseline_label]

    lines: list[str] = []
    a = lines.append

    a("# Experiment Comparison Report")
    a("")
    a(f"**Date:** {date.today()}  ")
    if model:
        a(f"**Model:** {model}  ")
    a(f"**Common tasks:** {N}  ")
    a("")

    # Result directories
    a("## Result directories")
    a("")
    a("| Experiment | Directory |")
    a("|---|---|")
    for label in labels:
        a(f"| {label} | `{dirs[label]}` |")
    a("")

    # Overall pass rate
    a("## Overall pass rate")
    a("")
    header_cols = " | ".join(["Pass", "Total", "Rate"])
    a(f"| Experiment | {header_cols} |")
    a("|---|---|---|---|")
    for label in labels:
        n = len(pass_sets[label])
        a(f"| {label} | {n} | {N} | **{pct(n, N)}** |")
    a("")

    # Per-site
    a("## Per-site breakdown")
    a("")
    site_header = " | ".join(labels) + " | N"
    a(f"| Site | {site_header} |")
    a("|---|" + "---|" * (len(labels) + 1))
    for site in data["sites"]:
        s = site_stats[site]
        t = s["total"]
        cells = " | ".join(pct(s.get(label, 0), t) for label in labels)
        a(f"| {site} | {cells} | {t} |")
    a("")

    # Delta vs baseline
    if len(labels) > 1:
        a(f"## Delta vs '{baseline_label}'")
        a("")
        a(f"| Experiment | Gained | Lost | Net |")
        a("|---|---|---|---|")
        for label in labels[1:]:
            s = pass_sets[label]
            gained = len(s - baseline_set)
            lost   = len(baseline_set - s)
            net    = len(s) - len(baseline_set)
            a(f"| {label} | +{gained} | −{lost} | {net:+d} |")
        a("")

        # Unique wins / losses
        a("### Task-level overlap")
        a("")
        all_pass  = set.intersection(*pass_sets.values())
        none_pass = common_ids - set.union(*pass_sets.values())
        a(f"- **All experiments pass ({len(all_pass)}):** {', '.join(map(str, sorted(all_pass)))}")
        a(f"- **No experiment passes ({len(none_pass)}):** {N - len(none_pass)} tasks pass in at least one")
        a("")
        for label in labels:
            only = pass_sets[label] - set.union(*(pass_sets[l] for l in labels if l != label))
            a(f"- **{label} unique wins ({len(only)}):** {', '.join(map(str, sorted(only))) or '—'}")
        a("")
        only_baseline = baseline_set - set.union(*(pass_sets[l] for l in labels[1:]))
        a(f"- **{baseline_label} only — lost with memory ({len(only_baseline)}):** "
          f"{', '.join(map(str, sorted(only_baseline))) or '—'}")
        a("")

    # Memory retrieval stats
    has_traces = any(v is not None for v in trace_stats.values())
    if has_traces:
        a("## Memory retrieval activity")
        a("")
        metric_header = " | ".join(labels)
        a(f"| Metric | {metric_header} |")
        a("|---|" + "---|" * len(labels))
        metrics = [
            ("tasks",            "Tasks traced"),
            ("avg_calls",        "Avg calls / task"),
            ("retrieval_rate",   "Tasks w/ ≥1 retrieval"),
            ("nonempty_calls",   "Non-empty retrieval calls"),
        ]
        for key, metric_label in metrics:
            cells = []
            for exp_label in labels:
                ts = trace_stats.get(exp_label)
                if ts is None:
                    cells.append("—")
                elif key == "retrieval_rate":
                    cells.append(f"{ts[key]:.0%}")
                elif key == "avg_calls":
                    cells.append(f"{ts[key]:.1f}")
                else:
                    cells.append(str(ts[key]))
            a(f"| {metric_label} | {' | '.join(cells)} |")
        a("")

    # Step stats
    a("## Steps per task")
    a("")
    step_header = " | ".join(labels)
    a(f"| Metric | {step_header} |")
    a("|---|" + "---|" * len(labels))
    for key, label_str in [("avg", "Avg"), ("min", "Min"), ("max", "Max")]:
        vals = " | ".join(
            f"{step_stats[l][key]:.1f}" if key == "avg" else str(step_stats[l][key])
            for l in labels
        )
        a(f"| {label_str} | {vals} |")
    a("")

    # Stop reason table for each experiment
    a("## Stop reason breakdown")
    a("")
    for label in labels:
        a(f"### {label}")
        a("")
        a("| Stop reason | Count | % |")
        a("|---|---|---|")
        total = sum(stop_stats[label].values())
        for reason, count in sorted(stop_stats[label].items(), key=lambda x: -x[1]):
            a(f"| {reason} | {count} | {pct(count, total)} |")
        a("")

    if notes:
        a("## Notes")
        a("")
        a(notes)
        a("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple WebArena experiment result directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        metavar="DIR[:LABEL]",
        help="Results directory, optionally followed by :Label",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Write Markdown report to this file",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name to include in the report header",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Freeform notes to append at the end of the Markdown report",
    )
    args = parser.parse_args()

    experiments: list[tuple[Path, str]] = []
    for spec in args.experiments:
        if ":" in spec:
            raw_dir, label = spec.rsplit(":", 1)
        else:
            raw_dir = spec
            label   = Path(spec).name
        results_dir = Path(raw_dir)
        if not results_dir.is_absolute():
            results_dir = PROJ / results_dir
        if not results_dir.exists():
            print(f"ERROR: directory not found: {results_dir}", file=sys.stderr)
            sys.exit(1)
        experiments.append((results_dir, label))

    if len(experiments) < 2:
        print("ERROR: provide at least two experiment directories to compare.", file=sys.stderr)
        sys.exit(1)

    data = compare(experiments)
    print_comparison(data)

    if args.output:
        md = build_markdown(data, model=args.model, notes=args.notes)
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJ / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        print(f"\nMarkdown report saved to: {out_path}")


if __name__ == "__main__":
    main()
