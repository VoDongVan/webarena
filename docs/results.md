# Results Folder — How to Read and Use Output Files

This document explains every file produced in a results directory (e.g.
`memorybank/results_baseline_27b/`) and how to work with them.

---

## Directory layout

```
results_baseline_27b/
├── config.json               CLI arguments snapshot (reproducibility)
├── log_files.txt             Path to the SLURM .out log for this run
├── error.txt                 Exceptions that crashed individual tasks
├── render_0.html             Step-by-step episode replay for task 0
├── render_1.html             ...
├── render_641.html
└── traces/
    ├── 0.zip                 Playwright browser trace for task 0
    ├── 1.zip
    └── ...
```

---

## `render_{task_id}.html`

### What it is

One HTML file per task. It is a **complete visual replay of the agent
episode**: every browser state the agent saw, every action it chose, and
every screenshot. It is the primary artifact for debugging and manual
inspection.

### How it is produced

`RenderHelper` in `browser_env/helper_functions.py` writes this file
incrementally during the episode. The file is created at the start of each
task (even if the task later crashes) and flushed after every step, so a
partial file means the agent was interrupted mid-episode.

### Structure

The file is plain HTML. Open it in any browser. It is organised as a
sequence of **steps**, each labelled `New Page`:

```
[Task metadata block]          — task_id, intent, sites, eval config

[Step 1] New Page
  URL bar                      — current page URL (clickable link)
  Accessibility tree           — text representation of the DOM the agent saw
  Screenshot (PNG, inline)     — actual browser screenshot at this state
  Previous action (pink)       — the action that led to this state
  LLM raw output (grey)        — full chain-of-thought + action string
  Parsed action object (grey)  — structured action dict (type, element_id, …)
  Resolved action (yellow)     — human-readable: e.g. click [42] where [42] is button 'Submit'

[Step 2] New Page
  ...

[Final step]
  stop [answer]                — agent's final answer, or early-stop reason
```

### What each colour means

| Colour | Content |
|--------|---------|
| Pink   | The **previous action** — what the agent did to arrive at this state |
| Grey   | Raw LLM output (top box) and the parsed `Action` dict (bottom box) |
| Yellow | Human-readable resolved action, e.g. `click [42] where [42] is link 'Add to Cart'` |

### Reading the final action

The last yellow box tells you how the episode ended:

| Final action | Meaning |
|---|---|
| `stop [some text]` | Agent chose to stop and submitted `some text` as its answer |
| `stop [N/A]` | Agent gave up — could not complete the task |
| `stop [Early stop: Failed to parse actions for 3 times]` | Agent produced 3 consecutive malformed outputs; run.py forced a stop |
| `stop [Early stop: Same action for 3 times]` | Agent repeated the same action 3 times |
| `stop [Early stop: Same typing action for 3 times]` | Agent typed the same text 3 times |
| `stop [Early stop: Reach max steps 30]` | Episode hit the 30-step limit |

### Common things to look for

- **Why did the agent fail?** Compare the agent's `stop` answer (last yellow
  box) against the reference answer in the task metadata block at the top.
- **Where did the agent go wrong?** Scan the URL bar per step — if the agent
  navigated to the wrong page early on, it will never recover.
- **Did the agent get stuck?** Three consecutive identical yellow boxes
  (same action) or three consecutive grey boxes with parse errors indicate
  the early-stop triggers fired.
- **Did the agent find the right information but answer incorrectly?**
  Compare the accessibility tree text near the end of the episode against
  the reference answer. This distinguishes navigation failures from
  answer-formatting failures.

---

## `traces/{task_id}.zip`

### What it is

A **Playwright browser trace** — a complete recording of browser-level
activity for the episode: every network request, every DOM snapshot, every
screenshot, and precise timing for each action.

### How it is produced

`env.save_trace()` is called in `run.py` after the episode finishes
(controlled by `--save_trace_enabled`). Traces are only written for tasks
that complete without crashing before the save call.

### How to view it

```bash
# Install Playwright if not already installed
playwright install chromium

# Open the interactive trace viewer in your browser
playwright show-trace memorybank/results_baseline_27b/traces/0.zip
```

The viewer shows a timeline with:
- A filmstrip of screenshots at each action
- A network panel (requests and responses)
- A DOM snapshot panel (inspect any element at any point)
- Action timing (how long each click/type/navigation took)

### When to use traces vs HTML renders

| Question | Use |
|---|---|
| What did the agent think and why? | `render_{task_id}.html` |
| Did the page actually load correctly? | `traces/{task_id}.zip` |
| Did a network request fail or return unexpected data? | `traces/{task_id}.zip` |
| Did a click land on the wrong element? | `traces/{task_id}.zip` (DOM snapshot) |
| How long did each step take? | `traces/{task_id}.zip` (timeline) |
| Quick overview of the full episode | `render_{task_id}.html` |

In practice, start with the HTML render to understand what happened at the
agent level. Only open the Playwright trace when you need to diagnose a
browser-level issue (page not loading, wrong element clicked, JS error).

---

## `error.txt`

### What it is

A log of **unhandled exceptions** that crashed individual tasks. Each entry
has the task config path, the exception message, and a full stack trace.

### Important caveat

`error.txt` is opened in **append mode** and accumulates entries from **all
job runs** that wrote to this results directory. If a task crashed in an
early job run (e.g. due to a cookie setup failure) but was retried and
succeeded in a later run, its error entry will still be present alongside
its `render_{task_id}.html`. Do not use `error.txt` alone to determine
whether a task's final result is reliable.

### Reliable eval health signal

A task's evaluation is reliable if and only if a `[Result] (PASS/FAIL)`
line appears for that task ID in the SLURM log (`memorybank/logs/wa_*.out`).
If a render file exists but no `[Result]` line was ever logged, the
evaluator crashed on the final run — that task's outcome is unknown.

The analysis script (`memorybank/analyze_results.py`) uses this signal
correctly.

---

## `config.json`

A snapshot of all CLI arguments passed to `run.py` for this run. Use it to
reproduce the exact experiment:

```bash
cat memorybank/results_baseline_27b/config.json
```

Key fields: `model`, `provider`, `temperature`, `top_p`, `max_tokens`,
`test_start_idx`, `test_end_idx`, `instruction_path`.

---

## Analysing results programmatically

### `analyze_results.py` — pass-rate summary

```bash
# Print summary to stdout (default: results_baseline_27b)
python memorybank/analyze_results.py

# Analyse a different results directory
python memorybank/analyze_results.py memorybank/results_memory_bm25_27b_test/

# Export per-task data
python memorybank/analyze_results.py --json results.json --csv results.csv
```

Produces:
- Overall pass rate
- Per-site pass rate
- Per eval-type pass rate and error rate
- Agent behaviour breakdown (stop reasons, step counts)
- List of passed task IDs
- List of tasks where the evaluator crashed (UNKNOWN outcome)

**Known limitation**: `parse_logs()` reads all `wa_*.out` files in `memorybank/logs/`
globally. When results from multiple runs cover the same task IDs, the latest log file
(highest job ID) wins. Analysing a 9B run's result directory while a later 27B run
exists will incorrectly show the 27B outcomes for shared task IDs.

---

### `inspect_memory_calls.py` — per-task memory query inspector (memory runs only)

Shows the `retrieve_memory` queries the agent issued each task, how many characters
each returned, and the full content of every memory the agent received (reconstructed
from `memory_provenance.json`).

```bash
# Default: 27B smoke-test run (all tasks)
python memorybank/inspect_memory_calls.py

# Only tasks 1 and 5
python memorybank/inspect_memory_calls.py --tasks 1 5

# Hide the empty-query count annotation
python memorybank/inspect_memory_calls.py --no-empty

# Print full memory bank at the end
python memorybank/inspect_memory_calls.py --bank

# 9B run
python memorybank/inspect_memory_calls.py \
    --log        memorybank/logs/wa_56406178.out \
    --memories   memorybank/memories/bm25_qwen3_9b/memories.json \
    --provenance memorybank/results_memory_bm25_9b_test/memory_provenance.json
```

Per-task output:
- Intent, outcome (PASS/FAIL), call counts (total / non-empty / empty / hits / extracted)
- Each non-empty query with its returned character count
- Full title + context + content of every memory the model received (derived from
  `memory_provenance.json`: parent IDs of new memories produced that task)

Summary table columns: `Task | Out | Total | NonEmp | Empty | Hits | Extr | Intent`

Required files (all produced automatically by `run_memory_experiment.sh`):
- `memorybank/logs/wa_<JOB_ID>.out` — SLURM log with `[MEMORY_CALL]` / `[MEMORY_RESULT]` lines
- `memorybank/memories/<run>/memories.json` — accumulated memory bank
- `memorybank/results_<run>/memory_provenance.json` — provenance graph written at run end
