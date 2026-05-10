# Experiment Comparison Report

**Date:** 2026-05-10  
**Model:** Qwen3-30B-A3B-FP8  
**Common tasks:** 208  

## Result directories

| Experiment | Directory |
|---|---|
| Baseline | `/scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena/memorybank/results_baseline_27b` |
| BM25 | `/scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena/memorybank/results_memory_bm25_27b_300tasks` |
| Dense | `/scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena/memorybank/results_memory_dense_27b_300tasks` |

## Overall pass rate

| Experiment | Pass | Total | Rate |
|---|---|---|---|
| Baseline | 45 | 208 | **21.6%** |
| BM25 | 42 | 208 | **20.2%** |
| Dense | 39 | 208 | **18.8%** |

## Per-site breakdown

| Site | Baseline | BM25 | Dense | N |
|---|---|---|---|---|
| gitlab | 24.3% | 18.9% | 24.3% | 37 |
| reddit | 12.5% | 12.5% | 0.0% | 8 |
| shopping | 21.7% | 21.7% | 19.3% | 83 |
| shopping_admin | 21.2% | 20.0% | 17.5% | 80 |

## Delta vs 'Baseline'

| Experiment | Gained | Lost | Net |
|---|---|---|---|
| BM25 | +10 | −13 | -3 |
| Dense | +4 | −10 | -6 |

### Task-level overlap

- **All experiments pass (29):** 11, 14, 77, 94, 95, 128, 157, 164, 188, 189, 190, 192, 205, 209, 210, 211, 231, 232, 233, 258, 260, 274, 275, 276, 278, 293, 294, 298, 299
- **No experiment passes (152):** 56 tasks pass in at least one

- **Baseline unique wins (7):** 15, 24, 30, 126, 133, 230, 295
- **BM25 unique wins (7):** 13, 28, 106, 150, 160, 245, 261
- **Dense unique wins (1):** 158

- **Baseline only — lost with memory (7):** 15, 24, 30, 126, 133, 230, 295

## Memory retrieval activity

| Metric | Baseline | BM25 | Dense |
|---|---|---|---|
| Tasks traced | — | 190 | 188 |
| Avg calls / task | — | 6.4 | 7.1 |
| Tasks w/ ≥1 retrieval | — | 99% | 99% |
| Non-empty retrieval calls | — | 1160 | 1310 |

## Steps per task

| Metric | Baseline | BM25 | Dense |
|---|---|---|---|
| Avg | 9.0 | 10.5 | 11.5 |
| Min | 1 | 1 | 1 |
| Max | 31 | 31 | 31 |

## Stop reason breakdown

### Baseline

| Stop reason | Count | % |
|---|---|---|
| agent_stop | 105 | 50.5% |
| Early stop: Failed to parse actions for 3 times | 46 | 22.1% |
| Early stop: Same action for 3 times | 39 | 18.8% |
| Early stop: Reach max steps 30 | 9 | 4.3% |
| unknown | 7 | 3.4% |
| Early stop: Same typing action for 3 times | 2 | 1.0% |

### BM25

| Stop reason | Count | % |
|---|---|---|
| Early stop: Failed to parse actions for 3 times | 84 | 40.4% |
| agent_stop | 79 | 38.0% |
| Early stop: Same action for 3 times | 19 | 9.1% |
| Early stop: Reach max steps 30 | 16 | 7.7% |
| unknown | 6 | 2.9% |
| Early stop: Same typing action for 3 times | 4 | 1.9% |

### Dense

| Stop reason | Count | % |
|---|---|---|
| Early stop: Failed to parse actions for 3 times | 83 | 39.9% |
| agent_stop | 80 | 38.5% |
| Early stop: Reach max steps 30 | 20 | 9.6% |
| Early stop: Same action for 3 times | 16 | 7.7% |
| unknown | 7 | 3.4% |
| Early stop: Same typing action for 3 times | 2 | 1.0% |
