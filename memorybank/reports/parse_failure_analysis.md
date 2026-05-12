# Parse Failure Analysis Report (v2 — Corrected)

**Date:** 2026-05-11  
**Experiments analyzed:** Baseline (27B), BM25 v1 (300 tasks), BM25 v2 (300 tasks, reminder fix)  
**Script:** `memorybank/analysis_helper/analyze_parse_failures.py`

> **Note:** This report supersedes the v1 analysis (2026-05-10). Two bugs were found and fixed in
> the analysis script before the numbers below were produced. See §1 for details.

---

## 1. Bugs Fixed in the Analysis Script

### Bug 1 — Wrong quote style in `_RAW_PRED_DQ_RE` (silent data loss)

Python's `repr()` always writes dict keys with single quotes: `'raw_prediction': ...`.
When the prediction string contains an apostrophe (e.g., `I'm on the page`), Python switches
the *value* to double quotes: `'raw_prediction': "...I'm..."`.
The old regex looked for `"raw_prediction": "..."` (double-quoted key), which never appears.

**Impact:** 73 of 335 NONE predictions in BM25 v2 (21.8%) were silently dropped.

**Fix:** Changed `_RAW_PRED_DQ_RE` to match `'raw_prediction': "..."` (single-quoted key,
double-quoted value).

### Bug 2 — Backtick regex excluded `=` (systematic misclassification)

`_BACKTICK_ACTION_RE` used character class `[\w\s\[\]\-\.:/]`, which excludes `=`.
The model's most common wrong action format is `scroll [direction=down]` — which contains `=`
and was therefore not recognized as a backtick-delimited action. These predictions fell
through to the `missing_backticks` category instead of `valid_format_but_failed`.

**Impact:** The `missing_backticks` category was inflated (was 37–54% of predictions; corrected
to near 0%), and `valid_format_but_failed` was heavily undercounted.

**Fix:** Added `=` to the character class in `_BACKTICK_ACTION_RE`.

---

## 2. Corrected Results

### 2.1 Baseline (no memory retrieval)

| Category | Predictions | % | First-failure tasks | % |
|---|---|---|---|---|
| `valid_format_but_failed` | 494 | **92.5%** | 148 | **91.9%** |
| `other_text` | 34 | 6.4% | 11 | 6.8% |
| `missing_backticks` | 6 | 1.1% | 2 | 1.2% |
| `xml_tool_call` | 0 | 0% | 0 | 0% |

161 parse-failure tasks. Total NONE predictions: 534.

### 2.2 BM25 v1 (no fix)

| Category | Predictions | % | First-failure tasks | % |
|---|---|---|---|---|
| `valid_format_but_failed` | 182 | **48.7%** | 20 | 23.3% |
| `xml_tool_call` | 162 | 43.3% | 59 | **68.6%** |
| `empty` | 22 | 5.9% | 7 | 8.1% |
| `other_text` | 4 | 1.1% | — | — |
| `missing_backticks` | 4 | 1.1% | — | — |

86 parse-failure tasks / 300 tasks run. Total NONE predictions: 374.

### 2.3 BM25 v2 (with format reminder fix)

| Category | Predictions | % | First-failure tasks | % |
|---|---|---|---|---|
| `valid_format_but_failed` | 204 | **60.9%** | 30 | 37.5% |
| `xml_tool_call` | 126 | 37.6% | 50 | **62.5%** |
| `empty` | 4 | 1.2% | — | — |
| `other_text` | 1 | 0.3% | — | — |

80 parse-failure tasks / 213 tasks run. Total NONE predictions: 335.

---

## 3. Two Distinct Failure Modes

Parse failures in these experiments come from two independent sources:

### Failure Mode A — Wrong scroll format (pre-existing model behavior)

The model (Qwen3-27B) consistently outputs `scroll [direction=down]` instead of the
positional format WebArena expects (`scroll [element_id] [down]`). This uses a named-parameter
style that likely comes from the model's training data or from the XML tool-call format
(`<parameter=direction>down</parameter>`) bleeding into its positional-argument output.

This failure exists in the **baseline** and accounts for 92.5% of baseline NONE predictions.
In BM25 v2 it is the dominant prediction-level failure at 60.9%, virtually all from
`scroll [direction=down]` (203 of 204 cases).

### Failure Mode B — XML tool-call bleed-through (memory-specific)

The agent uses OpenAI-style tool calling for `retrieve_memory`. Qwen3's tool call format is XML:

```
<tool_call>
<function=retrieve_memory>
<parameter=query>...</parameter>
</function>
</tool_call>
```

Per-step flow with memory:
1. **First LLM call** (tools provided): model outputs `<tool_call>` XML → vLLM parses it → memory retrieved.
2. **Second LLM call** (no tools): model is expected to output the backtick action format.

After seeing XML in the conversation history, the model often outputs XML again on the second
call. This produces output like `<tool_call><function=click>[2371]` — neither the tool-call
parser nor the action parser can handle it — resulting in a NONE action.

This failure is **entirely absent in the baseline** (0%) and is the dominant *first-failure*
mode in both BM25 v1 (68.6%) and BM25 v2 (62.5%).

---

## 4. Effect of the Reminder Message Fix (v1 → v2)

The fix injects a user-side message between the tool result and the second `call_llm`:

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

**What changed (first-failure per task):**

| Category | BM25 v1 | BM25 v2 | Change |
|---|---|---|---|
| `xml_tool_call` | 68.6% | 62.5% | −6.1 pp |
| `valid_format_but_failed` | 23.3% | 37.5% | +14.2 pp |
| `empty` | 8.1% | 0% | −8.1 pp |

The reminder eliminates empty responses and modestly reduces XML bleed-through. However,
some tasks that previously failed on XML now fail on wrong scroll format instead — the
reminder redirects the first failure from xml to wrong-format-scroll without preventing the
overall task failure. The net improvement in pass rate is small (~1–2 pp).

---

## 5. The Model Receives No Parse Error Feedback

When `ActionParsingError` is raised, the agent does **not** inform the model of the failure:

- **Within a step (retry path, `agent.py:294–299`):** The bad output is silently discarded.
  The LLM is called again with the **same conversation history** — no error message appended.
  (`max_retry=1`, so each step gets at most two attempts.)

- **Across steps:** A NONE action executes with no browser effect. On the next step, the model
  receives the **unchanged page observation** with no explanation. From the model's perspective
  the page simply did not change, and it typically infers it needs to continue the same action
  (e.g., scroll again) — producing the same wrong format again.

This no-feedback loop is why the "Failed to parse actions for 3 times" stop condition is
triggered by repetition: the model has no signal that its previous output was invalid and
defaults to repeating it.

---

## 6. Summary and Next Steps

### What the corrected analysis reveals

| | Baseline | BM25 v1 | BM25 v2 |
|---|---|---|---|
| Parse-fail stop rate | ~22% | ~40% | ~38% |
| Dominant first-failure | `scroll [direction=down]` | `xml_tool_call` | `xml_tool_call` |
| Dominant prediction-level | `scroll [direction=down]` | `scroll` + XML (split) | `scroll [direction=down]` |

The baseline's parse failure rate is already 22% due solely to the scroll format issue.
Memory retrieval adds XML bleed-through on top, pushing the rate to ~38–40%.

### Highest-impact fixes

**Fix 1 — Correct the scroll action format (targets both baseline and memory runs)**  
Post-process the LLM output before action parsing: detect `scroll [direction=down]` and
convert it to a valid format (e.g., `scroll [0] [down]` to scroll the viewport). This alone
could eliminate 90%+ of baseline parse failures and 60% of BM25 prediction-level failures.

**Fix 2 — Inject parse error feedback into the next step's observation (targets all runs)**  
When the previous action was NONE, prepend a brief error message to the next observation
before passing it to the model: e.g., `"[Error: your previous action could not be parsed.
Use the format: \`\`\`action [args]\`\`\`]"`. This breaks the repetition loop that causes
3-consecutive-NONE termination.

**Fix 3 — Stronger XML bleed-through suppression (targets memory runs)**  
The current reminder message partially suppresses XML bleed-through (−6 pp on first-failure
rate). A post-processing step that strips leading `<tool_call>...</tool_call>` XML before
action parsing would recover cases where the model outputs XML preamble followed by a valid
backtick action in the same response.
