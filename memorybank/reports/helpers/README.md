# compare_experiments.py

Compares two or more WebArena experiment result directories side-by-side. Prints a console summary and optionally writes a Markdown report.

## Usage

```bash
python memorybank/reports/helpers/compare_experiments.py \
    DIR[:LABEL] DIR[:LABEL] [...] \
    [--output FILE] [--model TEXT] [--notes TEXT]
```

Each positional argument is a results directory path (absolute or relative to the repo root), optionally followed by `:Label` for display. At least two experiments are required.

| Flag | Description |
|---|---|
| `--output FILE` | Write a Markdown report to `FILE` |
| `--model TEXT` | Model name to embed in the report header |
| `--notes TEXT` | Freeform text appended at the end of the Markdown report |

## Examples

```bash
# Two-way comparison, console only
python memorybank/reports/helpers/compare_experiments.py \
    memorybank/results_baseline_27b:Baseline \
    memorybank/results_memory_bm25_27b_300tasks:BM25

# Three-way comparison saved to a Markdown file
python memorybank/reports/helpers/compare_experiments.py \
    memorybank/results_baseline_27b:Baseline \
    memorybank/results_memory_bm25_27b_300tasks:BM25 \
    memorybank/results_memory_dense_27b_300tasks:Dense \
    --model "Qwen3-30B-A3B-FP8" \
    --output memorybank/reports/my_comparison.md
```

## What it outputs

**Overall pass rate** — pass count and percentage for each experiment on the common task set (tasks present in every experiment with a definitive PASS/FAIL outcome).

**Per-site breakdown** — pass rate by site (GitLab, Reddit, Shopping, Shopping Admin).

**Delta vs baseline** — how many tasks each memory experiment gained or lost relative to the first experiment listed.

**Task-level overlap** — which tasks all experiments pass, which tasks only one experiment passes, and which no experiment passes.

**Memory retrieval activity** — for experiments that have a `memory_traces/` directory: average memory calls per task, fraction of tasks with at least one non-empty retrieval, and total non-empty retrieval calls. Experiments without traces (e.g., baseline) show `—`.

**Steps per task** — average, min, and max agent steps across the common task set.

**Stop reason breakdown** — distribution of why each task ended (agent stop, parse failure, max steps, etc.).

## Notes

- **Outcome sources.** The script reuses `load_outcomes` from `analyze_results.py`. It first looks for per-task `result_N.json` files (written by newer runs), then falls back to parsing `log_files.txt` (present for all runs). Tasks with an `UNKNOWN` outcome (eval errors) are excluded from the common set.
- **Common task set.** Only tasks present with a definitive outcome in *every* experiment are included. Runs covering different task ranges will naturally produce a smaller common set.
- **Paths.** Relative paths are resolved from the repo root (`webarena/`). The `--output` path is also relative to the repo root unless absolute.
