#!/usr/bin/env python3
"""
Inspect memory queries and retrieved memories from a WebArena memory run.

For each task shows:
  - Intent and outcome (PASS/FAIL)
  - Every non-empty retrieve_memory query the model issued
  - How many characters each query returned
  - The full content of every memory that was in-context for that task

Usage:
    # Default: 27B smoke-test run
    python memorybank/inspect_memory_calls.py

    # 9B run
    python memorybank/inspect_memory_calls.py \\
        --log  memorybank/logs/wa_56406178.out \\
        --memories  memorybank/memories/bm25_qwen3_9b/memories.json \\
        --provenance  memorybank/results_memory_bm25_9b_test/memory_provenance.json

    # Only show tasks 1 and 5
    python memorybank/inspect_memory_calls.py --tasks 1 5

    # Print full memory bank at the end
    python memorybank/inspect_memory_calls.py --bank
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent

DEFAULT_LOG        = REPO / "memorybank/logs/wa_56406437.out"
DEFAULT_MEMORIES   = REPO / "memorybank/memories/bm25_qwen3_27b/memories.json"
DEFAULT_PROVENANCE = REPO / "memorybank/results_memory_bm25_27b_test/memory_provenance.json"


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

def parse_log(log_path: Path) -> list[dict]:
    """
    Parse wa_*.out into a list of task records.

    Each record:
        task_id          : int
        intent           : str
        outcome          : "PASS" | "FAIL" | "UNKNOWN"
        extraction_count : int  (memories saved after this task)
        calls            : list of {"query": str, "chars": int}
    """
    tasks: list[dict] = []
    current: dict | None = None
    pending_query: str | None = None

    for line in log_path.read_text(errors="replace").splitlines():
        # ── New task ──────────────────────────────────────────────────────
        m = re.search(r"\[Config file\].*?/(\d+)\.json", line)
        if m:
            current = {
                "task_id": int(m.group(1)),
                "intent": "",
                "outcome": "UNKNOWN",
                "extraction_count": 0,
                "calls": [],
            }
            tasks.append(current)
            pending_query = None
            continue

        if current is None:
            continue

        # ── Intent ───────────────────────────────────────────────────────
        m = re.search(r"\[Intent\]: (.+)", line)
        if m:
            current["intent"] = m.group(1).strip()
            continue

        # ── Result ───────────────────────────────────────────────────────
        m = re.search(r"\[Result\] \((PASS|FAIL)\)", line)
        if m:
            current["outcome"] = m.group(1)
            continue

        # ── Memory extraction count ───────────────────────────────────────
        m = re.search(r"\[MemoryExtraction\] Saved (\d+) memories", line)
        if m:
            current["extraction_count"] = int(m.group(1))
            continue

        # ── MEMORY_CALL: capture query ────────────────────────────────────
        m = re.search(r"\[MEMORY_CALL\] query=(.*)", line)
        if m:
            raw = m.group(1).strip()
            # strip surrounding quotes, then unescape inner backslash-escapes
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
                raw = raw[1:-1]
            raw = raw.replace("\\'", "'").replace('\\"', '"')
            pending_query = raw
            continue

        # ── MEMORY_RESULT: pair with pending query ────────────────────────
        m = re.search(r"\[MEMORY_RESULT\] returned (\d+) chars", line)
        if m and pending_query is not None:
            current["calls"].append({
                "query": pending_query,
                "chars": int(m.group(1)),
            })
            pending_query = None

    return tasks


# ---------------------------------------------------------------------------
# Memory bank loader
# ---------------------------------------------------------------------------

def load_memories(path: Path) -> dict[int, dict]:
    """Return {memory_id: memory_dict} from memories.json."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    mem_list = data.get("memories", data) if isinstance(data, dict) else data
    return {m["id"]: m for m in mem_list}


def load_provenance(path: Path) -> dict[int, list[int]]:
    """
    Return {new_memory_id: [parent_memory_ids]}.

    Parent IDs are the IDs retrieved during the task that produced
    new_memory_id (i.e., memory_client._session_retrieved_ids at the
    time add_memories was called).
    """
    if not path.exists():
        return {}
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


# ---------------------------------------------------------------------------
# Map task → retrieved memory IDs
# ---------------------------------------------------------------------------

def build_task_retrieved(
    tasks: list[dict],
    provenance: dict[int, list[int]],
) -> dict[int, list[int]]:
    """
    Return {task_id: sorted unique list of memory IDs retrieved during that task}.

    We assign new memory IDs to tasks in creation order (IDs 0, 1, 2, …)
    using each task's extraction_count to determine the boundary.  Then the
    retrieved IDs for a task are the union of parent_ids for all new memories
    produced in that task.
    """
    next_id = 0
    result: dict[int, list[int]] = {}

    for task in tasks:
        count = task["extraction_count"]
        retrieved: set[int] = set()
        for new_id in range(next_id, next_id + count):
            retrieved.update(provenance.get(new_id, []))
        result[task["task_id"]] = sorted(retrieved)
        next_id += count

    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def fmt_mem(m: dict, indent: int = 8) -> str:
    pad = " " * indent
    lines = []
    title = m.get("title") or "(no title)"
    lines.append(f"[ID {m['id']}] {title}")
    ctx = m.get("context", "").strip()
    cnt = m.get("content", "").strip()
    if ctx:
        lines.append(f"{pad}Context : {ctx}")
    if cnt:
        lines.append(f"{pad}Content : {cnt}")
    return "\n".join(lines)


def print_task(
    task: dict,
    memories: dict[int, dict],
    retrieved_ids: list[int],
    show_empty: bool,
) -> None:
    calls = task["calls"]
    non_empty = [c for c in calls if c["query"]]
    empty_count = len(calls) - len(non_empty)
    hit_count = sum(1 for c in calls if c["chars"] > 0)

    outcome_sym = {"PASS": "✓", "FAIL": "✗"}.get(task["outcome"], "?")
    print(f"\n{'='*70}")
    print(f"TASK {task['task_id']}  [{task['outcome']}] {outcome_sym}")
    print(f"Intent : {task['intent']}")
    print(
        f"Calls  : {len(calls)} total"
        f"  |  {len(non_empty)} non-empty"
        f"  |  {empty_count} empty"
        f"  |  {hit_count} returned content"
        f"  |  {task['extraction_count']} memories extracted"
    )

    # Non-empty queries
    print()
    if non_empty:
        print("  ── Queries issued ──────────────────────────────────────────")
        for i, c in enumerate(non_empty, 1):
            status = f"{c['chars']} chars" if c["chars"] > 0 else "0 chars  (no match)"
            q = c["query"]
            if len(q) > 220:
                q = q[:217] + "..."
            print(f"  [{i:>2}] ({status})")
            print(f"        {q}")
            print()
    else:
        print("  (no non-empty queries)")
        print()

    # Empty-query note
    if show_empty and empty_count > 0:
        print(f"  ── Empty queries: {empty_count}×  (all return 0 chars — degenerate loop) ──")
        print()

    # Retrieved memories
    unique_ids = sorted(set(retrieved_ids))
    if unique_ids:
        print(f"  ── Memories retrieved ({len(unique_ids)} unique IDs: {unique_ids}) ──────────────")
        for mid in unique_ids:
            m = memories.get(mid)
            if m is None:
                print(f"  [ID {mid}]  (not found in memories.json)")
                print()
                continue
            for line in fmt_mem(m, indent=10).splitlines():
                print(f"  {line}")
            print()
    elif hit_count > 0:
        # Calls returned content but we have no provenance data (task extracted
        # 0 memories, so no provenance entry was written for this task).
        print(f"  (memory IDs unknown — {hit_count} call(s) returned content but this task")
        print(f"   extracted 0 memories so no provenance was recorded)")
        print()
    else:
        print("  (no memories retrieved — bank was empty at start of this task)")
        print()


def print_bank(memories: dict[int, dict]) -> None:
    print(f"\n{'='*70}")
    print(f"FULL MEMORY BANK  ({len(memories)} memories)")
    print("=" * 70)
    for mid in sorted(memories):
        print()
        for line in fmt_mem(memories[mid], indent=6).splitlines():
            print(f"  {line}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Show memory queries and retrieved memories from a WebArena run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--log",        default=str(DEFAULT_LOG),
                    help="Path to wa_*.out SLURM log")
    ap.add_argument("--memories",   default=str(DEFAULT_MEMORIES),
                    help="Path to memories.json")
    ap.add_argument("--provenance", default=str(DEFAULT_PROVENANCE),
                    help="Path to memory_provenance.json")
    ap.add_argument("--tasks",  nargs="+", type=int, default=None,
                    help="Only show these task IDs")
    ap.add_argument("--no-empty", action="store_true",
                    help="Hide the empty-query count per task")
    ap.add_argument("--bank", action="store_true",
                    help="Print full memory bank at the end")
    args = ap.parse_args()

    log_path  = Path(args.log)
    mem_path  = Path(args.memories)
    prov_path = Path(args.provenance)

    if not log_path.exists():
        sys.exit(f"ERROR: log not found: {log_path}")

    print(f"Log       : {log_path}")
    print(f"Memories  : {mem_path}")
    print(f"Provenance: {prov_path}")

    tasks    = parse_log(log_path)
    memories = load_memories(mem_path)
    prov     = load_provenance(prov_path)
    retrieved_by_task = build_task_retrieved(tasks, prov)

    displayed = [t for t in tasks if args.tasks is None or t["task_id"] in args.tasks]

    print(
        f"\nFound {len(tasks)} tasks in log"
        f"  |  {len(memories)} memories in bank"
        f"  |  showing {len(displayed)} tasks"
    )

    for task in displayed:
        print_task(
            task,
            memories,
            retrieved_by_task.get(task["task_id"], []),
            show_empty=not args.no_empty,
        )

    if args.bank:
        print_bank(memories)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    hdr = f"  {'Task':>4}  {'Out':>4}  {'Total':>5}  {'NonEmp':>6}  {'Empty':>5}  {'Hits':>4}  {'Extr':>4}  Intent"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for task in displayed:
        calls     = task["calls"]
        non_empty = sum(1 for c in calls if c["query"])
        empty     = len(calls) - non_empty
        hits      = sum(1 for c in calls if c["chars"] > 0)
        extr      = task["extraction_count"]
        intent    = task["intent"][:42] + "..." if len(task["intent"]) > 42 else task["intent"]
        sym       = {"PASS": "✓", "FAIL": "✗"}.get(task["outcome"], "?")
        print(f"  {task['task_id']:>4}  {sym:>4}  {len(calls):>5}  {non_empty:>6}  {empty:>5}  {hits:>4}  {extr:>4}  {intent}")


if __name__ == "__main__":
    main()
