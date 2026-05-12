# Memory-Augmented WebArena: Experiment Summary

**Date:** 2026-05-12  
**Model:** Qwen3-27B via vLLM  
**Experiments:** Baseline · BM25 v1 · BM25 v2 · Dense v1 · Dense v2  
**Scope:** 213 tasks (test indices 0–299; 213 unique task IDs in this range)

---

## 1. What We Did

### 1.1 Experiments

We ran five experiment configurations, all on the same 213-task subset for fair comparison:

| Experiment | Retriever | Fix applied | Config file |
|---|---|---|---|
| **Baseline** | None | — | (original) |
| **BM25 v1** | BM25 (port 8020) | None | `webarena_memory_bm25_27b_300tasks.yaml` |
| **BM25 v2** | BM25 (port 8020) | Reminder message | `webarena_memory_bm25_27b_300tasks_v2.yaml` |
| **Dense v1** | Dense / BGE-large-en-v1.5 | None | `webarena_memory_dense_27b_300tasks.yaml` |
| **Dense v2** | Dense / BGE-large-en-v1.5 | Reminder message | `webarena_memory_dense_27b_300tasks_v2.yaml` |

### 1.2 The Reminder Message Fix

Both v2 experiments applied a single change to `agent/agent.py`. After appending the
`retrieve_memory` tool result to the conversation history and before the second `call_llm`
(which must produce a browser action), a user-side reminder message is injected:

```python
messages.append({
    "role": "user",
    "content": (
        "Memory retrieved. Now output your next browser action — "
        "do NOT call any tools. End your response with:\n"
        "In summary, the next action I will perform is "
        "```action [args]```"
    ),
})
```

**Motivation:** Qwen3's tool-call format is XML (`<tool_call><function=...>`). After the first
LLM call (which produces an XML tool call), the conversation history contains XML. On the second
call (no tools provided), the model often outputs XML again — producing output that neither the
tool-call parser nor the action parser can handle. The reminder explicitly breaks this priming.

---

## 2. Performance

### 2.1 Overall

| Experiment | Tasks | PASS | **PASS%** | Δ vs Baseline | Parse-fail stop | Same-action stop |
|---|---|---|---|---|---|---|
| Baseline | 213 | 45 | **21.1%** | — | 46 (21.6%) | 41 (19.2%) |
| BM25 v1 | 213 | 42 | **19.7%** | −1.4 pp | 86 (40.4%) | 19 (8.9%) |
| BM25 v2 | 213 | 47 | **22.1%** | **+1.0 pp** | 80 (37.6%) | 9 (4.2%) |
| Dense v1 | 213 | 39 | **18.3%** | −2.8 pp | 84 (39.4%) | 16 (7.5%) |
| Dense v2 | 213 | 42 | **19.7%** | −1.4 pp | 80 (37.6%) | 9 (4.2%) |

Key observations:
- **Memory v1 hurts performance.** Both BM25 v1 and Dense v1 underperform baseline because XML
  bleed-through nearly doubles the parse-fail stop rate (22% → 40%).
- **The reminder fix partially recovers performance.** BM25 v2 (+1.0 pp above baseline) is the
  only configuration that beats baseline. Dense v2 recovers to match it.
- **Memory cuts same-action stops in half.** Baseline: 19.2%, all memory runs: 4–9%. The
  retrieved context appears to help the model avoid repetitive actions, which is the one clear
  benefit of memory retrieval in these runs.

### 2.2 Per-Site Breakdown

| Site | Baseline | BM25 v1 | BM25 v2 | Dense v1 | Dense v2 |
|---|---|---|---|---|---|
| **GitLab** (39 tasks) | 9 (23.1%) | 7 (17.9%) | **10 (25.6%)** | 9 (23.1%) | 9 (23.1%) |
| **Reddit** (9 tasks) | 1 (11.1%) | 1 (11.1%) | 1 (11.1%) | 0 (0.0%) | 1 (11.1%) |
| **Shopping** (83 tasks) | **18 (21.7%)** | **18 (21.7%)** | 17 (20.5%) | 16 (19.3%) | 16 (19.3%) |
| **Shopping Admin** (82 tasks) | 17 (20.7%) | 16 (19.5%) | **19 (23.2%)** | 14 (17.1%) | 16 (19.5%) |

- BM25 v2 gains are concentrated in **GitLab (+7.7 pp vs BM25 v1)** and **Shopping Admin (+3.7 pp)**.
- Shopping results are flat or slightly negative across all memory runs — memory retrieval may be
  less helpful for shopping tasks, or the XML bleed-through disproportionately affects them.
- Dense v1 reddit result (0%) is an outlier (9 tasks, likely noise).

### 2.3 Full Stop Reason Breakdown

| Stop reason | Baseline | BM25 v1 | BM25 v2 | Dense v1 | Dense v2 |
|---|---|---|---|---|---|
| `agent_stop` (natural completion) | 105 (49.3%) | 79 (37.1%) | 92 (43.2%) | 80 (37.6%) | 94 (44.1%) |
| `parse_fail` (3× failed parse) | 46 (21.6%) | 86 (40.4%) | 80 (37.6%) | 84 (39.4%) | 80 (37.6%) |
| `same_action` (3× repeated action) | 41 (19.2%) | 19 (8.9%) | 9 (4.2%) | 16 (7.5%) | 9 (4.2%) |
| `max_steps` (30 steps reached) | 9 (4.2%) | 19 (8.9%) | 22 (10.3%) | 22 (10.3%) | 15 (7.0%) |
| `unknown` / other | 12 (5.6%) | 10 (4.7%) | 10 (4.7%) | 11 (5.2%) | 15 (7.0%) |

---

## 3. Error Analysis

### 3.1 Two Distinct Parse-Failure Modes

Parse failures (NONE actions) come from two independent sources:

#### Mode A — `scroll [direction=down]` (pre-existing model behavior)

Qwen3-27B consistently uses a named-parameter scroll format:

```
scroll [direction=down]
```

WebArena expects positional arguments:

```
scroll [element_id] [direction]      # e.g. scroll [0] [down]
```

This is a **baseline failure** — it exists without memory retrieval and accounts for 92.5% of all
baseline NONE predictions. In all memory runs it becomes the dominant *prediction-level* failure
after the XML bleed-through is partially suppressed by the reminder fix.

#### Mode B — XML tool-call bleed-through (memory-specific)

The agent uses OpenAI-style tool calling for `retrieve_memory`. Per-step flow:

1. **First LLM call** (tools provided): model outputs `<tool_call><function=retrieve_memory>...`
   → vLLM parses it as `msg.tool_calls` → memory retrieved.
2. **Second LLM call** (no tools): model should output backtick action format.

After seeing XML in conversation history, the model often produces hybrid output on the second call:

```
<tool_call><function=click>[2371]```
```

Neither parser can handle this → NONE action → retry → task termination.

This failure is **entirely absent in the baseline** (0%) and is the dominant *first-failure* cause
in all memory runs (63–69% of first-failure tasks).

### 3.2 NONE Prediction Breakdown

**Per prediction (all NONE actions in parse-fail tasks):**

| Category | Baseline | BM25 v1 | BM25 v2 | Dense v1 | Dense v2 |
|---|---|---|---|---|---|
| `valid_format_but_failed` (scroll bug) | 92.5% | 48.7% | 60.9% | 50.1% | 66.5% |
| `xml_tool_call` (bleed-through) | 0% | 43.3% | 37.6% | 42.1% | 33.2% |
| `empty` | 0% | 5.9% | 1.2% | 6.7% | 0.3% |
| `other_text` | 6.4% | 1.1% | 0.3% | 0.7% | 0% |
| `missing_backticks` | 1.1% | 1.1% | 0% | 0.2% | 0% |

**First-failure category per parse-fail task:**

| Category | Baseline | BM25 v1 | BM25 v2 | Dense v1 | Dense v2 |
|---|---|---|---|---|---|
| `xml_tool_call` | 0% | **68.6%** | **62.5%** | **63.1%** | **62.0%** |
| `valid_format_but_failed` | 91.9% | 23.3% | 37.5% | 31.0% | 38.0% |
| `empty` | 1.2% | 8.1% | 0% | 6.0% | 0% |

### 3.3 Effect of the Reminder Fix

| Metric | BM25 v1 → v2 | Dense v1 → v2 |
|---|---|---|
| PASS rate | 19.7% → 22.1% (+2.4 pp) | 18.3% → 19.7% (+1.4 pp) |
| Parse-fail stops | 86 → 80 (−7%) | 84 → 80 (−5%) |
| XML predictions (% of all NONE) | 43.3% → 37.6% (−5.7 pp) | 42.1% → 33.2% (−8.9 pp) |
| Empty predictions | 5.9% → 1.2% | 6.7% → 0.3% |
| XML first-failure rate (per task) | 68.6% → 62.5% (−6.1 pp) | 63.1% → 62.0% (−1.1 pp) |

**What the fix does well:**
- Eliminates empty responses almost entirely (6–7% → <1%).
- Modestly reduces total XML predictions (−6 to −9 pp at prediction level).

**What the fix does not solve:**
- XML is still the dominant *first failure* within a task (62% of parse-fail tasks). The reminder
  reduces XML on retries but cannot reliably prevent the first XML output.
- The scroll format bug is completely unaffected — it is not related to memory/tool-call history.
- No parse error feedback is given to the model: when a NONE action occurs, the next step's
  observation is unchanged, so the model has no signal to change behavior.

### 3.4 The No-Feedback Loop

When `ActionParsingError` is raised:
- **Within a step (retry):** The bad output is silently discarded; the LLM is called again with
  the same history and no error message. (`max_retry=1` → at most two attempts per step.)
- **Across steps:** A NONE action executes with no browser effect. The model receives the same
  page observation on the next step with no explanation, so it typically repeats the same wrong
  format, eventually triggering the 3-consecutive-NONE termination condition.

---

## 4. Memories Saved

| Experiment | Memories saved |
|---|---|
| BM25 v1 | `memorybank/memories/bm25_qwen3_27b_300tasks/memories.json` |
| BM25 v2 | `memorybank/memories/bm25_qwen3_27b_300tasks_v2/memories.json` |
| Dense v1 | `memorybank/memories/dense_qwen3_27b_300tasks/memories.json` |
| Dense v2 | `memorybank/memories/dense_qwen3_27b_300tasks_v2/memories.json` (546 memories) |

Both v2 runs use independent memory stores (start fresh), so they represent clean re-runs with
the fix applied — not incremental additions to v1 memory.

---

## 5. Completed Fixes (Applied After v1/v2 Experiments)

### Fix A — Correct scroll same-action early-stop *(applied 2026-05-12)*

**File:** `run.py` — `early_stop()`

The original early-stop for `scroll` was: terminate after 3 identical scroll directions.
This is incorrect: consecutive `scroll [down]` actions are valid while the page is still
moving. The agent should only be stopped if it's scrolling into a boundary (no content change).

**Change:** For scroll actions, the same-action check now also verifies that the text
observation before the k-scroll window equals the observation after. If content changed,
scrolling was productive and the run continues. If not, the boundary was hit — stop as before.

**Why this matters:** In baseline v2, 29/89 tasks (32.6%) were stopped by same-action
triggers, all on long-content pages. These tasks should have continued.

**Expected impact on v3+ runs:** Same-action stop rate drops from ~32% toward ~0% (only
tasks that genuinely hit a boundary will stop). Affected tasks will instead run to
`max_steps` (30), giving the agent more steps to find the answer.

---

## 6. To-Do for Future Work

The following improvements are ordered by estimated impact.

### Fix 1 — Correct the scroll action format *(highest impact)*

**Target:** Baseline + all memory runs  
**Mechanism:** Post-process the LLM output before action parsing. Detect
`scroll [direction=down]` (or `scroll [direction=up]`) and convert to the WebArena positional format.

```python
import re

_SCROLL_NAMED_RE = re.compile(
    r'```scroll\s+\[direction=(down|up)\]```', re.IGNORECASE
)

def fix_scroll_format(response: str) -> str:
    return _SCROLL_NAMED_RE.sub(
        lambda m: f"```scroll [0] [{m.group(1)}]```", response
    )
```

**Expected gain:** Eliminates ~92% of baseline parse failures and ~60–67% of memory-run
prediction-level failures. Could raise the baseline from 21.1% to ~26–27% and memory runs
proportionally. This is the single highest-leverage fix available.

### Fix 2 — Inject parse error feedback into the next step's observation *(medium impact)*

**Target:** All runs  
**Mechanism:** When the previous action was NONE (parse failure), prepend a brief error message
to the next step's observation before passing it to the model:

```python
if prev_action_type == ActionTypes.NONE:
    observation = (
        "[Error: your previous action could not be parsed. "
        "Use the format: ```action [args]```]\n\n" + observation
    )
```

**Expected gain:** Breaks the repetition loop that causes 3-consecutive-NONE termination. Tasks
that currently terminate on parse failure may recover and complete. Particularly effective for
the scroll bug where the model is stuck in a loop with no external signal to change.

### Fix 3 — Stronger XML bleed-through suppression *(medium impact, memory-specific)*

**Target:** BM25 and Dense memory runs  
**Mechanism:** After the second `call_llm`, post-process the response to strip leading
`<tool_call>...</tool_call>` XML blocks before passing to the action parser. The model sometimes
outputs an XML preamble followed by a valid backtick action in the same response; stripping the
XML would recover these cases.

```python
import re
_XML_PREFIX_RE = re.compile(r'^<tool_call>.*?</tool_call>\s*', re.DOTALL)
response = _XML_PREFIX_RE.sub('', response)
```

**Expected gain:** Reduces XML first-failure rate from ~62% toward 0% for the "XML preamble +
valid action" pattern. Combined with Fix 2 (feedback on remaining pure-XML failures), XML
bleed-through could be fully neutralized.

### Fix 4 — Stronger format reminder or system prompt injection *(low–medium impact)*

**Target:** Memory runs  
**Mechanism:** The current reminder is a user-turn message. Alternatives:
- Append the format requirement to the **system prompt** so it is always in context.
- Use a stronger phrasing that names the XML format explicitly: *"Do not use `<tool_call>` XML.
  You must end with: ` ```action [args]``` `."*

### Fix 5 — Add more diverse task coverage for evaluation *(process improvement)*

The 213-task subset (test indices 0–299) has only 9 Reddit tasks and no multi-site tasks. A
broader evaluation (all 812 tasks, or a stratified random sample) would give more reliable
per-site estimates and reduce noise in the headline pass rate.

---

## 7. Summary

| | Baseline | BM25 v1 | BM25 v2 | Dense v1 | Dense v2 |
|---|---|---|---|---|---|
| **Pass rate** | 21.1% | 19.7% | **22.1%** | 18.3% | 19.7% |
| **Parse-fail stop** | 21.6% | 40.4% | 37.6% | 39.4% | 37.6% |
| **XML bleed-through** | 0% | 43% of NONE | 38% of NONE | 42% of NONE | 33% of NONE |
| **Same-action stop** | 19.2% | 8.9% | 4.2% | 7.5% | 4.2% |

The current state: memory retrieval is **net neutral to slightly negative** at the task level.
The reminder fix closes most of the gap, and BM25 v2 narrowly beats the baseline, but the
fundamental bottlenecks — the scroll format bug and XML bleed-through — remain. Fix 1 (scroll
format post-processing) is the highest-leverage next step and would benefit every experiment
including the baseline.
