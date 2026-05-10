#!/usr/bin/env python3
"""
Retroactively build memory_traces/{task_id}.json from an existing memory run.

Reads:
  - SLURM log      (queries truncated at 300 chars, char counts per call)
  - memories.json  (full memory content)
  - memory_provenance.json  (new_memory_id → [parent_ids retrieved that task])

Writes:
  - <result_dir>/memory_traces/{task_id}.json

Each file matches the schema written by run.py during live runs:
  {
    "task_id": int,
    "intent": str,
    "outcome": "PASS" | "FAIL" | "UNKNOWN",
    "note": str,          # only present in retroactive files
    "calls": [
      {"query": str, "retrieved": str},
      ...
    ]
  }

Limitation: queries are truncated at 300 chars (SLURM log limitation).
Retrieved text is reconstructed per-task from provenance; all calls within a
task that returned >0 chars get the same reconstructed text (the full set of
memories retrieved for that task).  Calls with 0 chars (empty-query loops)
get retrieved="".

Usage:
    python memorybank/build_retroactive_traces.py

    python memorybank/build_retroactive_traces.py \\
        --log        memorybank/logs/wa_56406178.out \\
        --memories   memorybank/memories/bm25_qwen3_9b/memories.json \\
        --provenance memorybank/results_memory_bm25_9b_test/memory_provenance.json \\
        --result-dir memorybank/results_memory_bm25_9b_test
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
DEFAULT_RESULT_DIR = REPO / "memorybank/results_memory_bm25_27b_test"


# ---------------------------------------------------------------------------
# Reuse parsers from inspect_memory_calls
# ---------------------------------------------------------------------------

def parse_log(log_path: Path) -> list[dict]:
    tasks: list[dict] = []
    current: dict | None = None
    pending_query: str | None = None

    for line in log_path.read_text(errors="replace").splitlines():
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

        m = re.search(r"\[Intent\]: (.+)", line)
        if m:
            current["intent"] = m.group(1).strip()
            continue

        m = re.search(r"\[Result\] \((PASS|FAIL)\)", line)
        if m:
            current["outcome"] = m.group(1)
            continue

        m = re.search(r"\[MemoryExtraction\] Saved (\d+) memories", line)
        if m:
            current["extraction_count"] = int(m.group(1))
            continue

        m = re.search(r"\[MEMORY_CALL\] query=(.*)", line)
        if m:
            raw = m.group(1).strip()
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
                raw = raw[1:-1]
            raw = raw.replace("\\'", "'").replace('\\"', '"')
            pending_query = raw
            continue

        m = re.search(r"\[MEMORY_RESULT\] returned (\d+) chars", line)
        if m and pending_query is not None:
            current["calls"].append({"query": pending_query, "chars": int(m.group(1))})
            pending_query = None

    return tasks


def load_memories(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    mem_list = data.get("memories", data) if isinstance(data, dict) else data
    return {m["id"]: m for m in mem_list}


def load_provenance(path: Path) -> dict[int, list[int]]:
    if not path.exists():
        return {}
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


def build_task_retrieved(tasks: list[dict], provenance: dict[int, list[int]]) -> dict[int, list[int]]:
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


def format_retrieved(memory_ids: list[int], memories: dict[int, dict]) -> str:
    """Reconstruct the retrieved string in the same format as memory_client.retrieve()."""
    lines = []
    for i, mid in enumerate(memory_ids, 1):
        m = memories.get(mid)
        if m is None:
            continue
        lines.append(f"[{i}] {m.get('title', '')}: {m.get('content', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build retroactive memory_traces/ from an existing memory run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--log",        default=str(DEFAULT_LOG))
    ap.add_argument("--memories",   default=str(DEFAULT_MEMORIES))
    ap.add_argument("--provenance", default=str(DEFAULT_PROVENANCE))
    ap.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    args = ap.parse_args()

    log_path  = Path(args.log)
    mem_path  = Path(args.memories)
    prov_path = Path(args.provenance)
    out_dir   = Path(args.result_dir) / "memory_traces"

    if not log_path.exists():
        sys.exit(f"ERROR: log not found: {log_path}")

    tasks    = parse_log(log_path)
    memories = load_memories(mem_path)
    prov     = load_provenance(prov_path)
    retrieved_by_task = build_task_retrieved(tasks, prov)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing memory traces to {out_dir}/")

    for task in tasks:
        tid = task["task_id"]
        retrieved_ids = retrieved_by_task.get(tid, [])
        retrieved_text = format_retrieved(retrieved_ids, memories)

        calls = []
        for c in task["calls"]:
            calls.append({
                "query": c["query"],
                "retrieved": retrieved_text if c["chars"] > 0 else "",
            })

        record = {
            "task_id": tid,
            "intent": task["intent"],
            "outcome": task["outcome"],
            "note": (
                "Retroactively generated. Queries truncated at 300 chars (SLURM log). "
                "Retrieved text is the full set of memories for this task (not per-call)."
            ),
            "calls": calls,
        }

        out_path = out_dir / f"{tid}.json"
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        n_calls = len(calls)
        n_hits  = sum(1 for c in task["calls"] if c["chars"] > 0)
        print(f"  task {tid:>3}: {n_calls} calls ({n_hits} hits, {len(retrieved_ids)} unique memory IDs) → {out_path.name}")

    print(f"\nDone. {len(tasks)} files written.")


if __name__ == "__main__":
    main()
