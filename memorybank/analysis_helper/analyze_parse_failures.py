"""
analyze_parse_failures.py

Reads render HTML files for tasks that ended with "Failed to parse actions"
and categorizes what the model actually output in each failed attempt.

Usage:
    python memorybank/analysis_helper/analyze_parse_failures.py \
        memorybank/results_memory_bm25_27b_300tasks [--limit N]
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "memorybank"))
from analyze_results import load_outcomes  # noqa: E402


# ── HTML parsing helpers ──────────────────────────────────────────────────────

# Each step in the render HTML has a predict_action div that contains an
# action_object div with a repr() of the action dict.  We extract every
# raw_prediction value from those dicts.
_ACTION_BLOCK_RE = re.compile(
    r"class='action_object'[^>]*><pre>(.*?)</pre>",
    re.DOTALL,
)
_RAW_PRED_RE = re.compile(r"'raw_prediction':\s*'((?:[^'\\]|\\.)*)'", re.DOTALL)
# Python repr uses double-quoted value when the string contains a single quote (apostrophe).
# The key is always single-quoted in Python dict repr, so we match 'raw_prediction': "..."
_RAW_PRED_DQ_RE = re.compile(r"""'raw_prediction':\s*"((?:[^"\\]|\\.)*)"(?=[,}])""", re.DOTALL)
_NONE_TYPE_RE = re.compile(r"ActionTypes\.NONE")


def _unescape(s: str) -> str:
    """Reverse Python repr() escaping (\\n → \n, \\' → ', etc.)."""
    try:
        return bytes(s, "utf-8").decode("unicode_escape")
    except Exception:
        return s


def extract_none_predictions(html: str) -> list[str]:
    """Return raw_prediction strings for every NONE action in the HTML."""
    results = []
    for m in _ACTION_BLOCK_RE.finditer(html):
        block = m.group(1)
        if not _NONE_TYPE_RE.search(block):
            continue
        pm = _RAW_PRED_RE.search(block) or _RAW_PRED_DQ_RE.search(block)
        if pm:
            results.append(_unescape(pm.group(1)))
    return results


# ── Categorisation ────────────────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(r"<tool_call>", re.IGNORECASE)
_BACKTICK_ACTION_RE = re.compile(r"```(\w[\w\s\[\]\-\.:=,/\"']*?)```")
_ANSWER_PHRASE = "In summary, the next action I will perform is"
_VALID_ACTIONS = {
    "click", "hover", "type", "press", "scroll", "goto",
    "tab_focus", "new_tab", "go_back", "go_forward", "close_tab",
    "stop", "retrieve_memory",
}


def categorise(raw: str) -> str:
    """Assign one of five mutually exclusive categories."""
    stripped = raw.strip()
    if not stripped:
        return "empty"
    if _TOOL_CALL_RE.search(stripped):
        return "xml_tool_call"
    # Does it have backtick-delimited text?
    m = _BACKTICK_ACTION_RE.search(stripped)
    if m:
        verb = m.group(1).strip().split()[0].lower()
        if verb in _VALID_ACTIONS:
            return "valid_format_but_failed"  # parsed action but still raised error
        return "wrong_action_format"
    if _ANSWER_PHRASE in stripped:
        return "missing_backticks"  # has the phrase but no backticks
    return "other_text"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Path to results directory (relative to repo root or absolute)")
    parser.add_argument("--limit", type=int, default=0, help="Analyse at most N failed tasks (0=all)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = PROJ / results_dir

    if not results_dir.exists():
        print(f"ERROR: {results_dir} not found", file=sys.stderr)
        sys.exit(1)

    outcomes = load_outcomes(results_dir)

    failed_task_ids = [
        tid for tid, out in outcomes.items()
        if out == "FAIL"
    ]
    print(f"Total FAIL outcomes: {len(failed_task_ids)}")

    # Narrow to parse-failure tasks by reading stop reason from render HTML
    parse_fail_ids: list[int] = []
    for tid in failed_task_ids:
        html_path = results_dir / f"render_{tid}.html"
        if not html_path.exists():
            continue
        text = html_path.read_text(errors="replace")
        if "Failed to parse actions" in text:
            parse_fail_ids.append(tid)

    print(f"Tasks that stopped due to parse failure: {len(parse_fail_ids)}")

    if args.limit:
        parse_fail_ids = parse_fail_ids[: args.limit]
        print(f"(Analysing first {args.limit})")

    # Collect all NONE-action raw predictions from those tasks
    all_preds: list[str] = []
    task_cats: Counter = Counter()  # dominant category per task

    for tid in parse_fail_ids:
        html_path = results_dir / f"render_{tid}.html"
        text = html_path.read_text(errors="replace")
        preds = extract_none_predictions(text)
        cats = [categorise(p) for p in preds]
        all_preds.extend(preds)
        if cats:
            task_cats[cats[0]] += 1  # first failure is most diagnostic

    total_preds = len(all_preds)
    cat_counts: Counter = Counter(categorise(p) for p in all_preds)

    print(f"\nTotal NONE-action raw predictions collected: {total_preds}")
    print("\n── Category breakdown (per prediction) ──────────────────────────────")
    for cat, count in cat_counts.most_common():
        pct = 100 * count / total_preds if total_preds else 0
        print(f"  {cat:<30s}  {count:4d}  ({pct:5.1f}%)")

    print("\n── Dominant category of first failure per task ──────────────────────")
    for cat, count in task_cats.most_common():
        pct = 100 * count / len(parse_fail_ids) if parse_fail_ids else 0
        print(f"  {cat:<30s}  {count:4d}  ({pct:5.1f}%)")

    # Show a sample from the most common category
    most_common_cat = cat_counts.most_common(1)[0][0] if cat_counts else None
    if most_common_cat:
        samples = [p for p in all_preds if categorise(p) == most_common_cat][:3]
        print(f"\n── Sample raw predictions for '{most_common_cat}' ─────────────────────")
        for i, s in enumerate(samples, 1):
            preview = s[:300].replace("\n", "↵")
            print(f"\n  [{i}] {preview}")


if __name__ == "__main__":
    main()
