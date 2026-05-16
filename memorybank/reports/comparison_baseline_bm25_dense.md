# Experiment Comparison Report

**Date:** 2026-05-16  
**Model:** Qwen/Qwen3-32B-FP8 (27B active params) via vLLM  
**Task range:** 0–299, excluding map site (211 non-map tasks scored by all three)  
**Version:** v3 — thinking mode enabled, Diver-4B dense retriever

## Result directories

| Experiment | Directory |
|---|---|
| Baseline | `memorybank/results_baseline_27b_300tasks_v3` |
| BM25 | `memorybank/results_memory_bm25_27b_300tasks_v3` |
| Dense (Diver-4B) | `memorybank/results_memory_dense_diver_27b_300tasks_v3` |

## Overall pass rate

### All scored tasks (each experiment's full set)

| Experiment | Pass | Total | Rate |
|---|---|---|---|
| Baseline | 46 | 211 | 21.8% |
| BM25 | 52 | 213 | 24.4% |
| Dense | 60 | 212 | 28.3% |

### Common tasks only (211 tasks scored by all three — fair comparison)

| Experiment | Pass | Total | Rate | vs Baseline |
|---|---|---|---|---|
| Baseline | 46 | 211 | **21.8%** | — |
| BM25 | 52 | 211 | **24.6%** | +2.8 pp |
| Dense | 60 | 211 | **28.4%** | **+6.6 pp** |

## Per-site breakdown (common 211 tasks)

| Site | Baseline | BM25 | Dense | N |
|---|---|---|---|---|
| gitlab | 23.1% (9/39) | 17.9% (7/39) | 23.1% (9/39) | 39 |
| reddit | 0.0% (0/9) | 11.1% (1/9) | 22.2% (2/9) | 9 |
| shopping | 26.5% (22/83) | 32.5% (27/83) | 31.3% (26/83) | 83 |
| shopping_admin | 18.8% (15/80) | 21.2% (17/80) | 28.7% (23/80) | 80 |

Key observations:
- Dense leads on **shopping_admin** (+10 pp over baseline, +7.5 pp over BM25) — the largest site
- Dense doubles reddit pass rate vs baseline (0% → 22.2%)
- BM25 edges out dense on shopping by 1.2 pp; both beat baseline
- GitLab is flat across all three conditions

## Delta vs Baseline (common 211 tasks)

| Experiment | Gained | Lost | Net |
|---|---|---|---|
| BM25 | +17 | −11 | **+6** |
| Dense | +22 | −8 | **+14** |

Dense gains 14 net tasks vs baseline; BM25 gains 6.

### Task-level overlap

- **All three pass (32):** tasks solved by every condition — common ground ceiling
- **None pass (137):** 74 tasks pass in at least one condition
- **Baseline unique wins (5):** 41, 104, 144, 150, 207 — memory hurts on these
- **BM25 unique wins (6):** 30, 132, 158, 161, 187, 264
- **Dense unique wins (11):** 28, 29, 103, 112, 114, 121, 199, 213, 214, 225, 228

Dense has 11 unique wins (tasks it solves that neither baseline nor BM25 solves), the most of any condition.

### Tasks gained by dense (vs baseline)
28, 29, 47, 103, 112, 114, 121, 133, 135, 157, 166, 185, 199, 213, 214, 216, 225, 228, 234, + 3 more

### Tasks lost by dense (vs baseline)
12, 41, 104, 134, 144, 150, 160, 207 (8 tasks)

## Memory retrieval activity (common tasks)

| Metric | Baseline | BM25 | Dense |
|---|---|---|---|
| Tasks traced | — | 163 | 184 |
| Total calls | — | 1459 | 1536 |
| Non-empty calls | — | 1403 (96%) | 1515 (99%) |
| Avg calls / task | — | 9.0 | 8.3 |
| Tasks w/ ≥1 retrieval | — | 162/163 (99%) | 183/184 (99%) |

Empty query rate (thinking-mode artifact): BM25 ~4%, Dense ~1% (client-side guard added May 15).

## Steps per task (common 211 tasks)

| Metric | Baseline | BM25 | Dense |
|---|---|---|---|
| Avg steps | 12.9 | 13.9 | 13.0 |
| Min | 1 | 1 | 1 |
| Max | 31 | 31 | 31 |

BM25 uses ~1 more step on average than baseline, likely due to tool-call round-trips.
Dense is closer to baseline in avg steps despite having tool calls — may reflect more decisive retrieval.

## Notes on empty query behavior

With thinking mode enabled (all v3 runs), Qwen3-27B generates `retrieve_memory(query="")` at ~3–4% of steps. This is a model trait, not an implementation bug (confirmed by comparing with pre-thinking-mode runs that had 0 empty queries).

- **BM25:** handled silently by the retriever (returns 0 matches); no crashes
- **Dense (old run 57704098):** crashed 56 tasks via HTTP 500 (embedding server rejects empty strings)
- **Dense (current run 57733522):** fixed by server-side guard in `dense_retriever.py` + client-side guard in `agent.py`

## Comparison with earlier versions

| Version | Model | Baseline | BM25 | Dense | Common N |
|---|---|---|---|---|---|
| v1 (Apr) | Qwen3-30B-A3B-FP8 | 21.6% | 20.2% | 18.8% | 208 |
| v2 (May) | Qwen3-27B | — | — | — | — |
| **v3 (May 16)** | **Qwen3-27B** | **21.8%** | **24.6%** | **28.4%** | **211** |

v3 is the first version where memory retrieval consistently beats baseline. Key changes from v1:
- Model: switched from MoE (30B-A3B) to dense (27B) — better tool-call compliance
- Thinking mode enabled
- BM25 memory bank enriched with ~300 tasks of trajectories before run
- Dense retriever: Diver-4B (AQ-MedAI/Diver-Retriever-4B) instead of BGE
