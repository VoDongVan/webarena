#!/usr/bin/env python3
"""Merge memory_traces/{task_id}.json into render_{task_id}.html.

Each predict_action block in the HTML corresponds to one agent step (0-indexed).
The memory trace records which steps triggered a retrieval call and what was
returned. This script injects a green "Memory Retrieved" panel between the
prev_action div and the predict_action div for any step that has a retrieval.

Usage:
    # Enrich all tasks in a result dir (writes render_N_memory.html):
    python memorybank/enrich_renders.py memorybank/results_memory_dense_diver_27b_300tasks_v3

    # Overwrite existing enriched files:
    python memorybank/enrich_renders.py <result_dir> --overwrite

    # Single task:
    python memorybank/enrich_renders.py <result_dir> --task-id 102
"""

import argparse
import html as html_lib
import json
import re
from pathlib import Path

# ── CSS injected into the <head> ──────────────────────────────────────────────

_EXTRA_CSS = """
    .memory_panel {
        border-left: 4px solid #388e3c;
        background: #f1f8e9;
        margin: 6px 0 2px 0;
        padding: 8px 12px;
        font-family: monospace;
        font-size: 0.88em;
    }
    .memory_panel .mem_header {
        font-weight: bold;
        color: #2e7d32;
        margin-bottom: 4px;
    }
    .memory_panel details summary {
        cursor: pointer;
        color: #1b5e20;
        margin-bottom: 4px;
    }
    .memory_panel pre {
        background: #e8f5e9;
        padding: 6px;
        border-radius: 3px;
        white-space: pre-wrap;
        word-wrap: break-word;
        margin: 4px 0 0 0;
    }
    .memory_panel .no_mem {
        color: #888;
        font-style: italic;
    }
"""

# ── HTML block for one retrieval call ─────────────────────────────────────────

def _memory_block(step: int, calls: list[dict]) -> str:
    """Build an HTML panel for all retrieval calls at a given step."""
    parts = [f"<div class='memory_panel'>"]
    label = "retrieval" if len(calls) == 1 else f"{len(calls)} retrievals"
    parts.append(f"<div class='mem_header'>🔍 Memory {label} — step {step}</div>")

    for i, call in enumerate(calls):
        if len(calls) > 1:
            parts.append(f"<div style='margin-top:6px'><b>Call {i+1}</b></div>")

        query = call.get("query", "")
        retrieved = call.get("retrieved", "")

        # Query (collapsible — can be long)
        parts.append("<details>")
        parts.append(f"<summary><b>Query</b></summary>")
        parts.append(f"<pre>{html_lib.escape(query)}</pre>")
        parts.append("</details>")

        # Retrieved memories
        parts.append("<div><b>Retrieved:</b></div>")
        if retrieved and retrieved.strip():
            parts.append(f"<pre>{html_lib.escape(retrieved)}</pre>")
        else:
            parts.append("<span class='no_mem'>(no memories retrieved)</span>")

    parts.append("</div>")
    return "\n".join(parts)


# ── Core enrichment logic ──────────────────────────────────────────────────────

_PREDICT_ACTION_RE = re.compile(r"<div class='predict_action'>")
_HEAD_CLOSE_RE = re.compile(r"</head>", re.IGNORECASE)


def enrich_render(html_path: Path, trace_path: Path, output_path: Path) -> bool:
    """Inject memory panels into html_path using trace_path; write to output_path.

    Returns True if any panels were injected.
    """
    html = html_path.read_text(encoding="utf-8")
    with open(trace_path, encoding="utf-8") as f:
        trace = json.load(f)

    calls = trace.get("calls", [])
    if not calls:
        output_path.write_text(html, encoding="utf-8")
        return False

    # Group calls by step (a step can have multiple calls in theory)
    step_calls: dict[int, list[dict]] = {}
    for call in calls:
        step = call["step"]
        step_calls.setdefault(step, []).append(call)

    # Inject extra CSS into <head>
    css_tag = f"<style>{_EXTRA_CSS}</style>"
    html = _HEAD_CLOSE_RE.sub(f"{css_tag}\n</head>", html, count=1)

    # Walk through predict_action divs and insert panels before matching ones
    result: list[str] = []
    last_end = 0
    step_idx = 0
    injected = 0

    for m in _PREDICT_ACTION_RE.finditer(html):
        result.append(html[last_end : m.start()])
        if step_idx in step_calls:
            result.append(_memory_block(step_idx, step_calls[step_idx]))
            result.append("\n")
            injected += 1
        result.append(html[m.start() : m.end()])
        last_end = m.end()
        step_idx += 1

    result.append(html[last_end:])
    output_path.write_text("".join(result), encoding="utf-8")
    return injected > 0


# ── CLI ────────────────────────────────────────────────────────────────────────

def process_result_dir(
    result_dir: Path,
    overwrite: bool = False,
    task_id: int | None = None,
) -> None:
    trace_dir = result_dir / "memory_traces"
    if not trace_dir.exists():
        print(f"No memory_traces/ found in {result_dir}")
        return

    if task_id is not None:
        trace_files = [trace_dir / f"{task_id}.json"]
    else:
        trace_files = sorted(trace_dir.glob("*.json"), key=lambda p: int(p.stem))

    total = ok = skipped = 0
    for trace_file in trace_files:
        if not trace_file.exists():
            print(f"  Trace not found: {trace_file}")
            continue
        tid = trace_file.stem
        render_file = result_dir / f"render_{tid}.html"
        if not render_file.exists():
            continue  # task crashed before render was written
        output_file = result_dir / f"render_{tid}_memory.html"
        if output_file.exists() and not overwrite:
            skipped += 1
            continue
        injected = enrich_render(render_file, trace_file, output_file)
        status = "injected" if injected else "no calls"
        print(f"  Task {tid:>4}: {output_file.name}  [{status}]")
        ok += 1
        total += 1

    print(f"\nDone. {ok} written, {skipped} skipped (use --overwrite to redo).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge memory traces into render HTMLs."
    )
    parser.add_argument("result_dir", help="Path to experiment result directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_memory.html files",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        metavar="N",
        help="Enrich a single task instead of all",
    )
    args = parser.parse_args()

    process_result_dir(Path(args.result_dir), overwrite=args.overwrite, task_id=args.task_id)


if __name__ == "__main__":
    main()
