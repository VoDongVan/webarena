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

Retriever: dense embeddings via `BAAI/bge-large-en-v1.5` served on port 8101 (vLLM pooling runner). Retrieval server on port 8020. `top_k=3`. Memory initialized fresh each run.

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

---

## Summary Table

| Experiment | Tasks | PASS% | Scroll fix | XML reminder |
|---|---|---|---|---|
| Baseline v1 | 213 | 21.1% | ✗ | N/A |
| **Baseline v2** | 213 | TBD | **✓** | N/A |
| BM25 v1 | 213 | 19.7% | ✗ | ✗ |
| BM25 v2 | 213 | 22.1% | ✗ | **✓** |
| Dense v1 | 213 | 18.3% | ✗ | ✗ |
| Dense v2 | 213 | 19.7% | ✗ | **✓** |

**Next logical runs** (not yet started as of 2026-05-12):
- BM25 v3 and Dense v3: apply both the scroll fix and the XML reminder fix together.
- These would be the first runs with both fixes applied simultaneously.
