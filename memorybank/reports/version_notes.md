# Experiment Version Notes

All experiments use **Qwen3-27B** via vLLM on UMass Unity HPC (superpod-a100, vram80).  
All v2+ experiments cover **tasks 0–299** (213 unique tasks in that range) for fair comparison.

---

## Baseline (no memory retrieval)

### v1 — `results_baseline_27b`
- Full run: all 812 tasks (635 completed, task IDs 0–751).
- **No scroll prompt fix.** Prompt says `` `scroll [direction=down|up]` ``; parser only accepts `scroll [down]` / `scroll [up]` → model follows the prompt and always fails to parse scroll actions.
- Results (first 213 tasks for comparison): **45/213 PASS (21.1%)**, parse-fail stop 21.6%.

### v2 — `results_baseline_27b_300tasks_v2`
- First 300 tasks only (test indices 0–299).
- **Scroll prompt fix applied**: prompt changed to `` `scroll [down|up]` `` in both `p_cot_id_actree_2s.json` and `p_cot_id_actree_2s_memory.json` (and all other prompt files).
- Expected: parse-fail stop rate should drop from ~22% toward ~0%; overall pass rate should improve.

---

## BM25 Memory Retrieval

Retriever: keyword BM25 on port 8020. `top_k=3`. Memory initialized fresh each run.

### v1 — `results_memory_bm25_27b_300tasks`
- First 300 tasks.
- **No scroll prompt fix. No XML reminder fix.**
- XML tool-call bleed-through: model outputs `<tool_call>` XML on the second LLM call (which should produce a browser action), because the conversation history contains a prior XML tool exchange. Causes ~40% parse-fail stop rate.
- Results: **42/213 PASS (19.7%)**, parse-fail stop 40.4%. Worse than baseline.

### v2 — `results_memory_bm25_27b_300tasks_v2`
- First 300 tasks.
- **XML reminder fix applied** (`agent/agent.py`): after appending the `retrieve_memory` tool result, a user-side message is injected before the second `call_llm` — *"Memory retrieved. Now output your next browser action..."* — to suppress XML bleed-through.
- **No scroll prompt fix** (not yet applied at the time of this run).
- Results: **47/213 PASS (22.1%)**, parse-fail stop 37.6%. Beats baseline v1 by +1.0 pp.

---

## Dense Memory Retrieval

Retriever: dense embeddings via an embedding model served on port 8101 (vLLM pooling runner, `--runner pooling`). Retrieval server on port 8020. `top_k=3`. Memory initialized fresh each run.

**Infrastructure change (2026-05-13):** The embedding server now runs on a **separate GPU node** (submitted as `run_embedding_server.sh` on `gpu` partition, L40S by default) rather than sharing the A100 with the 27B LLM. The embedding node's hostname is written to `$NODEDIR/.embedding_node`; the GPU job reads it and passes `http://<host>:8101/v1/embeddings` to the retrieval server. This frees the main GPU from memory pressure and allows any embedding model size to be used without tuning `--gpu-memory-utilization`.

### v1 — `results_memory_dense_27b_300tasks`
- First 300 tasks.
- **No scroll prompt fix. No XML reminder fix.**
- Same XML bleed-through problem as BM25 v1, plus higher empty-response rate (6.7%).
- Results: **39/213 PASS (18.3%)**, parse-fail stop 39.4%. Worst result overall.

### v2 — `results_memory_dense_27b_300tasks_v2`
- First 300 tasks.
- **XML reminder fix applied** (same as BM25 v2).
- **No scroll prompt fix** (not yet applied at the time of this run).
- Results: **42/213 PASS (19.7%)**, parse-fail stop 37.6%. Matches baseline v1 but does not beat it.

### v3 (bge-large) — `results_memory_dense_27b_300tasks_v3`
- First 300 tasks. All three fixes applied (scroll prompt, XML reminder, scroll same-action stop).
- Embedding model: `BAAI/bge-large-en-v1.5`. Embedding server on dedicated L40S node.
- Config: `webarena_memory_dense_27b_300tasks_v3.yaml`
- Results: TBD (not yet run as of 2026-05-13).

### v3 (Diver) — `results_memory_dense_diver_27b_300tasks_v3`
- First 300 tasks. All three fixes applied.
- Embedding model: `AQ-MedAI/Diver-Retriever-4B` — a reasoning-intensive retriever fine-tuned
  from Qwen3-Embedding-4B on math/coding/medical data with hard negatives. Achieves SOTA on BRIGHT
  benchmark (nDCG@10 = 46.8), outperforming ReasonIR-8B and RaDeR-7B. Paper: arXiv 2508.07995.
- Embedding server on dedicated L40S node (48 GB VRAM, `--gpu-memory-utilization 0.50`).
- Config: `webarena_memory_dense_diver_27b_300tasks_v3.yaml`
- Results: TBD (submitted 2026-05-13).

---

---

## Behavioral Fix: Scroll Same-Action Early Stop (2026-05-12)

**File:** `run.py` — `early_stop()`

**Problem:** The original same-action early-stop treated `scroll [down]` repeated 3× as a
stuck loop and terminated the task. This is wrong when the page is long: legitimate reading
behavior requires many consecutive downward scrolls, and each one reveals new content.

**Analysis:** In baseline v2 (89 tasks), 29 tasks (32.6%) were stopped by same-action. All
29 were on long content pages (shopping admin analytics, reddit posts, gitlab issue lists)
where the agent was genuinely progressing through the page.

**Fix:** For `ActionTypes.SCROLL` only, the same-action check now also requires that the
page text observation is identical before and after the scroll window. If content changed,
the scroll was productive and the run continues. If content is unchanged (boundary reached),
the run stops as before.

```
Before: stop after 3× scroll [down] regardless of page movement
After:  stop after 3× scroll [down] only if the page did not move
```

All other action types (click, goto, type, etc.) retain the original k-consecutive behavior.

**Expected impact:** Same-action stop rate should drop substantially (was 32.6% in baseline
v2, 19.2% in baseline v1). Tasks that previously terminated early due to scroll will now
run longer — up to max_steps (30) — giving the agent more time to find answers.

**Applies to:** All experiments from v3 onward.

---

## Summary Table

| Experiment | Tasks | PASS% | Scroll prompt | XML reminder | Scroll stop fix | Embedding model |
|---|---|---|---|---|---|---|
| Baseline v1 | 213 | 21.1% | ✗ | N/A | ✗ | — |
| **Baseline v2** | 213 | TBD | **✓** | N/A | ✗ | — |
| BM25 v1 | 213 | 19.7% | ✗ | ✗ | ✗ | — |
| BM25 v2 | 213 | 22.1% | ✗ | **✓** | ✗ | — |
| Dense v1 | 213 | 18.3% | ✗ | ✗ | ✗ | bge-large-en-v1.5 |
| Dense v2 | 213 | 19.7% | ✗ | **✓** | ✗ | bge-large-en-v1.5 |
| **Baseline v3** | running | TBD | **✓** | N/A | **✓** | — |
| **BM25 v3** | running | TBD | **✓** | **✓** | **✓** | — |
| **Dense v3 (bge-large)** | TBD | TBD | **✓** | **✓** | **✓** | bge-large-en-v1.5 |
| **Dense v3 (Diver)** | submitted | TBD | **✓** | **✓** | **✓** | Diver-Retriever-4B |

**Running/submitted as of 2026-05-13:**
- Baseline v3 and BM25 v3 are currently running (SLURM jobs 57560889 and 57561593).
- Dense Diver v3 submitted alongside; uses `AQ-MedAI/Diver-Retriever-4B` on a separate L40S node.
- Dense bge-large v3 not yet submitted (deferred until Diver results available for comparison).
