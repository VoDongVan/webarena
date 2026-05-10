"""
Analyzes per-step memory query behavior from memory traces + render HTMLs.

Usage:
    python analysis_helper/query_behavior.py [result_dir]

Default result_dir: memorybank/results_memory_bm25_27b_60tasks
"""

import json
import os
import sys

def count_steps(render_path):
    if not os.path.exists(render_path):
        return None
    with open(render_path) as f:
        content = f.read()
    return content.count("<h2>New Page</h2>")


def analyze(result_dir):
    traces_dir = os.path.join(result_dir, "memory_traces")
    if not os.path.isdir(traces_dir):
        print(f"No memory_traces/ in {result_dir}")
        return

    rows = []
    for fname in sorted(os.listdir(traces_dir), key=lambda x: int(x.replace(".json", ""))):
        task_id = int(fname.replace(".json", ""))
        with open(os.path.join(traces_dir, fname)) as f:
            t = json.load(f)

        steps = count_steps(os.path.join(result_dir, f"render_{task_id}.html"))
        total = len(t["calls"])
        non_empty = sum(1 for c in t["calls"] if c["query"])
        empty = total - non_empty
        hits = sum(1 for c in t["calls"] if c.get("retrieved"))
        rows.append(dict(
            task_id=task_id,
            outcome=t["outcome"],
            steps=steps,
            total=total,
            non_empty=non_empty,
            empty=empty,
            hits=hits,
        ))

    # --- Per-task table ---
    print(f"{'Task':>4}  {'Out':4}  {'Steps':>5}  {'Calls':>5}  {'Non-∅':>5}  {'Empty':>5}  {'Hits':>4}  {'Calls/Step':>10}  Pattern")
    print("-" * 80)
    for r in rows:
        steps = r["steps"]
        ratio = f"{r['total']/steps:.1f}" if steps else "  ?"
        if steps is None:
            pattern = "no render"
        elif r["empty"] == 0 and r["total"] <= steps + 2:
            pattern = "normal"
        elif r["empty"] > r["non_empty"] * 2:
            pattern = "*** STUCK (empty loop)"
        else:
            pattern = "multi-call/step"
        print(f"{r['task_id']:>4}  {r['outcome']:4}  {str(steps) if steps is not None else '?':>5}  "
              f"{r['total']:>5}  {r['non_empty']:>5}  {r['empty']:>5}  {r['hits']:>4}  "
              f"{ratio:>10}  {pattern}")

    # --- Aggregates ---
    total_calls = sum(r["total"] for r in rows)
    total_nonempty = sum(r["non_empty"] for r in rows)
    total_empty = sum(r["empty"] for r in rows)
    total_hits = sum(r["hits"] for r in rows)
    stuck = [r for r in rows if r["empty"] > r["non_empty"] * 2]
    normal = [r for r in rows if r["empty"] == 0]

    print("\n=== Summary ===")
    print(f"Tasks analyzed:       {len(rows)}")
    print(f"Total memory calls:   {total_calls}")
    print(f"  Non-empty queries:  {total_nonempty}  ({100*total_nonempty/total_calls:.0f}%)")
    print(f"  Empty queries:      {total_empty}  ({100*total_empty/total_calls:.0f}%)")
    print(f"  Retrieval hits:     {total_hits}  ({100*total_hits/total_nonempty:.0f}% of non-empty)")
    print(f"Tasks with no empty queries (normal):  {len(normal)}")
    print(f"Tasks stuck in empty loop:             {len(stuck)}  → task IDs: {[r['task_id'] for r in stuck]}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default = os.path.join(base, "results_memory_bm25_27b_60tasks")
    result_dir = sys.argv[1] if len(sys.argv) > 1 else default
    analyze(result_dir)
