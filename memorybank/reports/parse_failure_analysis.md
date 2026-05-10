# Parse Failure Analysis Report

**Date:** 2026-05-10  
**Experiments analyzed:** BM25 (300 tasks), Dense (300 tasks), Baseline (635 tasks)

---

## Problem

Memory retrieval experiments (BM25 and Dense) have a ~40% "Failed to parse actions" stop rate — nearly double the 22% baseline rate. This is the primary cause of underperformance relative to baseline.

## Root Cause

The agent uses OpenAI-style tool calling to invoke `retrieve_memory`. Qwen3's tool call format is XML:

```
<tool_call>
<function=retrieve_memory>
<parameter=query>...</parameter>
</function>
</tool_call>
```

The per-step flow when memory is used:
1. **First call** (with tools): model outputs `<tool_call>` XML → vLLM parses it as `msg.tool_calls` → memory is retrieved
2. **Second call** (no tools): model is expected to output the backtick action format

After seeing XML tool-call format in the conversation history, the model often outputs XML on the second call too — for example `<tool_call><function=click>[2371]` — a format that neither the tool-call parser nor the action parser can handle. This produces a NONE action, and after 3 consecutive NONE actions, the task fails with "Failed to parse actions for 3 times."

## Diagnostic Results

Script: `memorybank/analysis_helper/analyze_parse_failures.py`

### BM25

| Category | Predictions | % | Tasks (first failure) | % |
|---|---|---|---|---|
| xml_tool_call | 137 | 47.4% | 53 | **61.6%** |
| missing_backticks | 109 | 37.7% | 15 | 17.4% |
| empty | 22 | 7.6% | 9 | 10.5% |
| valid_format_but_failed | 13 | 4.5% | 3 | 3.5% |
| other_text | 8 | 2.8% | 1 | 1.2% |

86 of 171 failed tasks (50%) ended due to parse failure.  
Total NONE-action predictions collected: 289.

### Dense

| Category | Predictions | % | Tasks (first failure) | % |
|---|---|---|---|---|
| xml_tool_call | 156 | 45.2% | 52 | **61.9%** |
| missing_backticks | 151 | 43.8% | 26 | 31.0% |
| empty | 27 | 7.8% | 5 | 6.0% |
| valid_format_but_failed | 6 | 1.7% | 0 | — |
| other_text | 5 | 1.4% | 0 | — |

84 of 172 failed tasks (49%) ended due to parse failure.  
Total NONE-action predictions collected: 345.

### Baseline (no memory, for comparison)

| Category | Predictions | % | Tasks (first failure) | % |
|---|---|---|---|---|
| missing_backticks | 385 | 90.6% | 135 | 83.9% |
| other_text | 34 | 8.0% | 11 | 6.8% |
| valid_format_but_failed | 6 | 1.4% | 3 | 1.9% |
| **xml_tool_call** | **0** | **0%** | **0** | **0%** |

161 of 543 failed tasks (30%) ended due to parse failure.  
Total NONE-action predictions collected: 425.

**Key finding:** XML tool call format is entirely absent in the baseline and accounts for ~62% of first-failure tasks in both memory experiments. This is the incremental cause of the doubled parse failure rate.

## Fix Applied

**File:** `agent/agent.py` (lines 268–277)

After the tool result is appended to the conversation and before the second `call_llm`, we insert a user-side reminder message that explicitly tells the model to switch back to the backtick action format:

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

This breaks the XML-format priming introduced by the tool-call exchange in the conversation history.

## Expected Impact

- Eliminates or greatly reduces the 61–62% of parse-failure tasks caused by XML tool call bleed-through
- Should bring the "Failed to parse actions" rate closer to the 22% baseline level
- Net effect: meaningful improvement to overall task pass rate for both BM25 and Dense retrievers

## Next Step

Submit a re-run of BM25 or Dense (tasks 0–300) with the fix applied. Compare the parse failure stop reason distribution and overall pass rate against the original results in `comparison_baseline_bm25_dense.md`.
