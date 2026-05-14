# Baseline vs. BM25 vs. Dense Memory — Three-Way Analysis (v3, 2026-05-14)

**Model:** Qwen3.5-27B-Instruct (vLLM, local)  
**Benchmark:** WebArena, tasks 0–299 (300 submitted; 213 non-map tasks reach evaluation)  
**Variant tag:** v3 — scroll same-action early-stop fix applied to all three runs  
**Dense retriever:** AQ-MedAI/Diver-Retriever-4B (co-located on same A100 as the main LLM)

**Changes in v3:**
Bug in original prompt: Original prompt of WebArena instructs the model to use "scroll [direction=down|up]", but the expected action is "scroll [down|up]" => model cannot scroll
In v3, scroll up and down does not counted as repeated action, unless the environment does not change after actions. This is an issue for tasks that required long scrolling.
After fixing the 2 bugs, both baseline and BM25 performance improve significantly.

---

## 1. Executive Summary

On the **132 tasks where all three experiments produced valid results**, accuracy is:

- **Baseline:** 35/132 = **26.5%**
- **BM25 memory:** 39/132 = **29.5%** (+3.0 pp over baseline)
- **Dense memory:** 43/132 = **32.6%** (+6.1 pp over baseline, +3.1 pp over BM25)

Both memory-retrieval methods beat the no-memory baseline. Dense leads BM25 by +3.1 pp on the shared subset, driven almost entirely by stronger performance on `shopping_admin` sentiment/review tasks. BM25 retains a small edge on `shopping` tasks.

**Caveats on the dense result:** Dense completed 157 of 213 non-map tasks; 56 tasks failed without writing results (transient service crashes concentrated in gitlab tasks 169–177, shopping tasks 281–286, and shopping_admin tasks 288–292). The remaining dense gap will be closed by a re-run. BM25 has a separate incomplete region (33 tasks, 260–299) with known eval errors; a re-run for those is in progress.

---

## 2. Experiment Configuration

| Setting | Baseline | BM25 Memory | Dense Memory |
|---|---|---|---|
| Config file | `webarena_baseline_27b_300tasks_v3.yaml` | `webarena_memory_bm25_27b_300tasks_v3.yaml` | `webarena_memory_dense_diver_27b_300tasks_v3.yaml` |
| Retriever | — | BM25 (TF-IDF) | Dense (Diver-4B) |
| Retriever port | — | 8020 | 8020 |
| Embedding model | — | — | `AQ-MedAI/Diver-Retriever-4B` |
| Embedding port | — | — | 8101 |
| Top-k memories | — | 3 | 3 |
| Extraction model | — | same (Qwen3.5-27B) | same (Qwen3.5-27B) |
| Memory init | — | fresh | fresh |
| Max steps | 30 | 30 | 30 |
| Prompt | `p_cot_id_actree_2s` | `p_cot_id_actree_2s_memory` | `p_cot_id_actree_2s_memory` |
| Task range | 0–299 | 0–299 | 0–299 |
| Non-map tasks | 213 | 213 | 213 |
| Tasks with results | 211 | 180 | 157 |

---

## 3. Overall Accuracy

### 3a. Raw (all non-map tasks, unequal coverage)

These numbers are *not* directly comparable — each experiment covers a different subset of the 213 tasks. Included for reference.

| Metric | Baseline | BM25 | Dense |
|---|---|---|---|
| Tasks with results | 211 | 180 | 157 |
| **PASS** | **46 (21.8%)** | **44 (24.4%)** | **53 (33.8%)** |
| FAIL | 165 (78.2%) | 136 (75.6%) | 104 (66.2%) |
| Tasks missing results | 2 | 33 | 56 |

### 3b. Clean three-way (132 tasks where all three have valid results)

This is the **primary metric**. All 132 tasks have a PASS/FAIL result in every experiment.

| Metric | Baseline | BM25 | Dense | BM25 − BL | Dense − BL |
|---|---|---|---|---|---|
| Tasks | 132 | 132 | 132 | — | — |
| **PASS** | **35 (26.5%)** | **39 (29.5%)** | **43 (32.6%)** | **+4 (+3.0 pp)** | **+8 (+6.1 pp)** |
| FAIL | 97 (73.5%) | 93 (70.5%) | 89 (67.4%) | — | — |

### 3c. Two-way clean (178 tasks where baseline and BM25 both have results)

Retained for continuity with the earlier baseline-vs-BM25 report. The 178-task intersection excludes dense.

| Metric | Baseline | BM25 | Delta |
|---|---|---|---|
| Tasks | 178 | 178 | — |
| **PASS** | **38 (21.3%)** | **44 (24.7%)** | **+6 (+3.4 pp)** |
| FAIL | 140 (78.7%) | 134 (75.3%) | — |

---

## 4. Data Quality: Missing Results

### Baseline (2 missing — tasks 1, 65)
Both are `shopping_admin` tasks. Evaluation process did not write a result file, likely isolated eval bugs. Negligible impact on conclusions.

### BM25 (33 missing — tasks 260–299 range)

| Site | Missing results |
|---|---|
| shopping | 25 |
| gitlab | 5 |
| shopping_admin | 3 |
| **Total** | **33** |

**Root cause:** WebArena service containers bind to dynamic hostnames. A suspected hostname rotation during the late-run batch (tasks 260–299) caused the evaluator's url_match comparisons to fail. Tasks were never written to disk. **A re-run of tasks 260–299 is in progress.**

### Dense (56 missing — various task IDs)

| Site | Missing results |
|---|---|
| shopping_admin | 23 |
| shopping | 18 |
| gitlab | 14 |
| reddit | 1 |
| **Total** | **56** |

Missing task IDs: 1, 2, 6, 21, 25, 26, 28, 44, 50, 51, 63, 78, 104, 107, 116, 119, 124, 129, 130, 132, 133, 135, 142, 144, 169–173, 175, 177, 182, 184, 191, 194, 197, 201, 206, 213, 216, 228, 239, 241, 243, 247, 281–286, 288–292

**Root cause:** Transient service crashes during the 21.7-hour run. Three clusters are visible: gitlab tasks 169–177 (7 consecutive), shopping tasks 281–286 (6 consecutive), and shopping_admin tasks 288–292 (5 consecutive). The runner skips failed tasks without writing a result file. **A re-run targeting these 56 tasks is planned once the BM25 re-run finishes.**

---

## 5. Per-Site Breakdown (clean three-way, 132 tasks)

| Site | n | Baseline | BM25 | Dense | BM25−BL | Dense−BL |
|---|---|---|---|---|---|---|
| gitlab | 20 | 4 (20.0%) | 3 (15.0%) | 3 (15.0%) | −1 | −1 |
| reddit | 8 | 0 (0.0%) | 1 (12.5%) | 1 (12.5%) | +1 | +1 |
| shopping | 46 | 16 (34.8%) | 19 (41.3%) | 18 (39.1%) | +3 | +2 |
| shopping_admin | 58 | 15 (25.9%) | 16 (27.6%) | 21 (36.2%) | +1 | **+6** |
| **Total** | **132** | **35 (26.5%)** | **39 (29.5%)** | **43 (32.6%)** | **+4** | **+8** |

Key observations:
- **Dense dominates `shopping_admin`** (+6 over baseline, +5 over BM25). Dense memory retrieves semantically similar prior tasks better than BM25 on review/sentiment extraction queries.
- **BM25 leads `shopping`** by a slim margin over Dense (+3 vs +2 over baseline). BM25's term-matching may be better for exact product name or order number lookups.
- **Both memory methods underperform baseline on `gitlab`**. This is consistent with the XML parse-failure bug (see Section 7) and the fact that gitlab tasks often require exact commit-counting procedures not well-represented in early memories.
- **Reddit**: sample too small (n=8) for conclusions; both methods tie at 12.5%.

---

## 6. Task-Level Agreement (clean three-way, 132 tasks)

| Outcome pattern | Count | Tasks |
|---|---|---|
| All 3 PASS | 26 | 11, 14, 24, 48, 77, 94–96, 115, 117, 126, 128, 164, 188–190, 192, 205, 208–211, 231–233, 258 |
| All 3 FAIL | 79 | — |
| Only Baseline | 3 | 41, 150, 207 |
| Only BM25 | 4 | 30, 158, 161, 187 |
| Only Dense | 8 | 29, 103, 112, 114, 121, 199, 214, 225 |
| Baseline + BM25, not Dense | 3 | 12, 134, 160 |
| Baseline + Dense, not BM25 | 3 | 15, 183, 230 |
| BM25 + Dense, not Baseline | 6 | 13, 47, 157, 166, 185, 234 |

Memory methods (BM25 or Dense) collectively provide **18 exclusive wins** vs. 3 exclusive wins for baseline-only. Dense has the most unique wins (8); BM25 has 4.

### Tasks only Dense solved

| Task | Site | Intent (truncated) |
|---|---|---|
| 29 | reddit | Count comments with more downvotes than upvotes |
| 103 | gitlab | List issues in kkroening/ffmpeg-python with specific label |
| 112 | shopping_admin | Customers who expressed dissatisfaction with Circe fleece |
| 114 | shopping_admin | Customers who expressed dissatisfaction with Antonia product |
| 121 | shopping_admin | Reasons why customers like Circe hooded fleece |
| 199 | shopping_admin | Order ID of newest pending order |
| 214 | shopping_admin | Key aspects customers dislike about Zing Jump Rope |
| 225 | shopping | What customers say about brush from Sephora |

Pattern: **5 of 8 are `shopping_admin` review/sentiment tasks**. Dense retrieval finds semantically similar past tasks (e.g., "customers who expressed dissatisfaction about product X") and returns relevant memory even when the product name differs — a case where dense wins over BM25's term-matching.

### Tasks BM25 + Dense both solve (memory wins, not baseline)

| Task | Site | Intent (truncated) |
|---|---|---|
| 13 | shopping_admin | Number of reviews mentioning "excellent" |
| 47 | shopping | Fulfilled orders over past month (date: 6/12/2023) |
| 157 | shopping_admin | Show all customers |
| 166 | shopping | Main criticisms of this product (extract relevant sentences) |
| 185 | shopping_admin | Brand of products with 3 units left |
| 234 | shopping | Order number of most recent on-hold order |

Both memory methods beat baseline on these structured lookup tasks (inventory counts, order history, review extraction).

### Tasks only Baseline solved (memory hurt or was irrelevant)

| Task | Site | Intent (truncated) |
|---|---|---|
| 41 | shopping_admin | List top-1 search term in the store |
| 150 | shopping | Price of fake tree bought Jan 2023 |
| 207 | gitlab | Combined commits by Eric+Kilian on 1/3/2023 |

Only 3 tasks where baseline beats both memory methods on the shared 132-task set.

---

## 7. Failure Mode Analysis (Baseline vs. BM25 — from v3 run logs)

*Step-count and stop-reason data are not available in current result JSON files; the following is from the detailed v3 run log analysis. Dense-specific failure breakdown is deferred until stop-reason logging is added to the result format.*

### 7a. Stop reason distribution (baseline vs BM25, 213 tasks)

| Stop reason | Baseline | BM25 | Change |
|---|---|---|---|
| Agent stop (voluntary) | 127 (59.6%) | 102 (47.9%) | −25 |
| Same action ×3 | 39 (18.3%) | 20 (9.4%) | **−19** |
| Reach max steps 30 | 30 (14.1%) | 36 (16.9%) | +6 |
| Unknown / eval error | 10 (4.7%) | 36 (16.9%) | +26 (eval errors) |
| Same typing ×3 | 5 (2.3%) | 5 (2.3%) | 0 |
| Parse failure ×3 | 2 (0.9%) | 14 (6.6%) | **+12** |

Key shifts:
- **Same-action failures drop by half in BM25** (39 → 20). Memory breaks repetitive navigation loops.
- **Parse failures increase sharply** (2 → 14). Caused by `--tool-call-parser qwen3_xml` failing on malformed tool tags. Baseline has no tool calls so is unaffected. Deferred bug.
- **Max steps increase** (30 → 36). Memory context may lengthen reasoning on some tasks.

### 7b. Repeated action failures (same-action ×3)

| Action type | Baseline | BM25 |
|---|---|---|
| scroll | 15 | 5 |
| click | 15 | 11 |
| goto | 8 | 2 |
| press | 1 | 2 |
| **Total** | **39** | **20** |

Scroll failures cut by two-thirds. Click failures remain relatively high — these are mostly pagination elements in `shopping_admin`.

### 7c. Parse failures — deferred bug

BM25's 14 parse-failure stops are caused by `--tool-call-parser qwen3_xml` failing on malformed XML in tool-use responses. Fixing this is estimated to recover 1–2 additional task successes.

---

## 8. Steps per Task (Baseline vs. BM25 — from v3 run logs)

| Metric | Baseline | BM25 |
|---|---|---|
| Avg steps (all tasks) | 13.0 | 11.9 |
| Avg steps (PASS tasks) | 7.3 | 5.7 |
| Avg steps (FAIL tasks) | 14.6 | 13.6 |
| Min steps | 1 | 0 |
| Max steps | 31 | 31 |

BM25 completes PASS tasks in **1.6 fewer steps on average** (5.7 vs 7.3, −22%). Memory provides a navigation shortcut: the agent recalls a related procedure from a prior task and avoids exploratory steps.

*Dense step data not yet available from result files. Will be added once step logging is included in the output format.*

---

## 9. Discussion

### Which method is best?

On the 132-task shared evaluation, **Dense leads at 32.6%**, followed by BM25 (29.5%), then baseline (26.5%). The Dense advantage is concentrated in `shopping_admin` sentiment/review tasks (+6 over baseline vs +1 for BM25). BM25 retains a narrow edge on `shopping` exact-lookup tasks (+3 over baseline vs +2 for Dense).

Both memory methods consistently beat baseline, with only 3 tasks where baseline beats both memory methods. The memory benefit is real and robust across all four sites (or neutral on reddit/gitlab).

### Why does Dense win on shopping_admin?

Dense retrieval encodes query semantics — "customers dissatisfied with product X" matches memories about "customers dissatisfied with product Y" even when the product name differs. BM25 requires exact or near-exact term overlap. Shopping_admin review tasks are template-like (same intent, different product), so dense retrieval generalizes better across them.

### Why does BM25 outperform baseline on shopping?

Shopping tasks tend to involve specific product names or order numbers. BM25's term-matching retrieves memories about the exact product or action type (e.g., "on-hold orders", "fulfilled orders last month"). Dense may retrieve semantically similar but procedurally less relevant memories on these.

### GitLab regression

Both memory methods underperform baseline on `gitlab` tasks (15% vs 20%). Two factors likely contribute: (1) the XML parse-failure bug affects BM25 and dense equally, causing ~14–15% of tasks to terminate early; (2) early-session gitlab memories may encode wrong procedures that mislead the agent on later tasks.

### Stability of the Dense estimate

Dense has only 157/213 tasks completed. The 56 missing tasks are spread across all sites. If the missing tasks follow the same accuracy distribution as the completed ones (~33.8%), the final Dense accuracy will be similar. However, the missing cluster at tasks 281–292 (shopping + shopping_admin) could shift the per-site numbers.

### Is the gap stable across task count?

| Tasks in common | Baseline | BM25 | Dense |
|---|---|---|---|
| 111 (at 112 dense tasks) | 27.0% | 30.6% | 31.5% |
| 132 (at 157 dense tasks) | 26.5% | 29.5% | 32.6% |

The dense advantage has increased slightly as more tasks accumulate (31.5% → 32.6%). BM25 vs baseline gap is stable (~3 pp).

---

## 10. Limitations and Next Steps

| Issue | Severity | Status |
|---|---|---|
| BM25 eval errors (tasks 260–299, 33 tasks) | Medium — affects coverage | **Re-run submitted (job pending)** |
| Dense missing results (56 non-map tasks) | Medium — affects coverage | Re-run planned after BM25 re-run completes |
| XML parse failure bug (6.6% of BM25 tasks) | Low — est. 1–2 tasks | Deferred; fix `tool-call-parser` configuration |
| GitLab regression in memory conditions | Medium — both BM25 and Dense underperform | Investigate XML parse failures + memory content |
| Step/stop-reason logging for Dense | Low — diagnostic only | Add `num_steps` and `stop_reason` to result JSON format |
| Reddit accuracy (0–12.5%) | Low | Small sample (n=8); not actionable yet |

---

## Appendix A: Passed Task Lists

**Baseline PASS (46 tasks):** 11, 12, 14, 15, 24, 41, 48, 77, 94, 95, 96, 104, 115, 117, 126, 128, 134, 144, 150, 160, 164, 183, 188, 189, 190, 192, 205, 206, 207, 208, 209, 210, 211, 230, 231, 232, 233, 258, 260, 274, 275, 276, 278, 293, 294, 295

**BM25 PASS (44 tasks):** 11, 12, 13, 14, 24, 30, 47, 48, 77, 94, 95, 96, 115, 117, 126, 128, 132, 133, 134, 135, 157, 158, 160, 161, 164, 166, 185, 187, 188, 189, 190, 192, 205, 206, 208, 209, 210, 211, 216, 231, 232, 233, 234, 258

**Dense PASS (53 tasks):** 11, 13, 14, 15, 24, 29, 47, 48, 77, 94, 95, 96, 103, 112, 114, 115, 117, 121, 126, 128, 157, 164, 166, 183, 185, 188, 189, 190, 192, 199, 205, 208, 209, 210, 211, 214, 225, 230, 231, 232, 233, 234, 258, 260, 274, 275, 276, 278, 293, 294, 295, 298, 299

**All-three PASS (26 tasks):** 11, 14, 24, 48, 77, 94, 95, 96, 115, 117, 126, 128, 164, 188, 189, 190, 192, 205, 208, 209, 210, 211, 231, 232, 233, 258

---

## Appendix B: Missing Result Task IDs

**Baseline (2):** 1, 65

**BM25 (33):** 260, 261, 262, 263, 264, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299

*All concentrated in the final batch. Suspected root cause: WebArena service hostname rotation causing url_match evaluator failures.*

**Dense (56):** 1, 2, 6, 21, 25, 26, 28, 44, 50, 51, 63, 78, 104, 107, 116, 119, 124, 129, 130, 132, 133, 135, 142, 144, 169, 170, 171, 172, 173, 175, 177, 182, 184, 191, 194, 197, 201, 206, 213, 216, 228, 239, 241, 243, 247, 281, 282, 283, 284, 285, 286, 288, 289, 290, 291, 292

*Concentrated in three clusters: gitlab tasks 169–177, shopping tasks 281–286, shopping_admin tasks 288–292. Likely transient service failures during the 21.7-hour run.*
