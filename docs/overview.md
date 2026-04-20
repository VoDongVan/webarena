# WebArena — Project Overview

## What Is It?

WebArena is a **benchmark and evaluation framework** for autonomous web agents. It provides 812 real-world web tasks across self-hosted web applications, letting researchers build and test LLM-based agents that navigate websites like a human.

Paper: https://arxiv.org/abs/2307.13854

---

## High-Level Architecture

```
run.py  (orchestrator)
  │
  ├── agent/              LLM-based decision making
  │     ├── PromptAgent          calls LLM each step; optional memory_client
  │     ├── TeacherForcingAgent  replays fixed action sequence
  │     ├── memory_client.py     HTTP client for retriever server (memory feature)
  │     └── prompts/             prompt templates + constructors
  │           ├── CoTPromptConstructor
  │           ├── DirectPromptConstructor
  │           └── MemoryCoTPromptConstructor   (injects retrieved memories)
  │
  ├── browser_env/        Browser automation (Playwright + Gymnasium)
  │     ├── ScriptBrowserEnv     the Gym environment
  │     ├── actions.py           19 action types + execution
  │     ├── processors.py        DOM/accessibility tree → text
  │     └── auto_login.py        cookie-based pre-authentication
  │
  ├── llms/               LLM provider abstraction
  │     ├── lm_config.py         model configuration dataclass
  │     ├── utils.py             call_llm() dispatcher
  │     ├── tokenizers.py        tiktoken / LlamaTokenizer wrapper
  │     └── providers/           vLLM + OpenAI + HuggingFace integrations
  │
  ├── evaluation_harness/ Task success scoring
  │     ├── evaluators.py        StringEvaluator, URLEvaluator, HTMLContentEvaluator
  │     └── helper_functions.py  shopping/reddit/gitlab API helpers + LLM fuzzy match
  │
  └── config_files/       812 task JSON definitions
```

---

## Supported Web Environments

All services run locally via Docker. URLs are set via environment variables.

| Service | Env Var | Default Port | Purpose |
|---|---|---|---|
| OneStopShop (e-commerce) | `SHOPPING` | 7770 | Shopping tasks |
| Magento CMS (admin panel) | `SHOPPING_ADMIN` | 7780 | Admin operations |
| Reddit clone | `REDDIT` | 9999 | Forum tasks |
| GitLab | `GITLAB` | 8023 | Code/repo tasks |
| Wikipedia (Kiwix) | `WIKIPEDIA` | 8888 | Information retrieval |
| OpenStreetMap | `MAP` | 3000 | Map interaction |
| Homepage | `HOMEPAGE` | 4399 | Landing page |

---

## Task Config Format

Each of the 812 tasks is a JSON file with:

```json
{
  "task_id": 42,
  "intent": "Find the cheapest red shoes under $50",
  "start_url": "http://localhost:7770",
  "storage_state": "cookies/shopping.json",
  "eval": {
    "eval_types": ["string_match"],
    "reference_answers": {
      "exact_match": "$34.99"
    }
  }
}
```

`eval_types` determines which evaluator(s) run. Multiple types are ANDed (all must pass).

---

## End-to-End Data Flow

```
run.py
  │
  ├─ 1. Load task config JSON
  ├─ 2. auto_login.py → refresh cookies → Chromium starts pre-authenticated
  ├─ 3. env.reset() → initial observation (accessibility tree text + screenshot)
  │
  ├─ Loop (up to max_steps=30):
  │   ├─ early_stop() check (max steps / parse failures / repeating actions)
  │   ├─ agent.next_action(trajectory, intent, meta_data)
  │   │     PromptConstructor → formats obs into LLM prompt
  │   │     (MemoryCoTPromptConstructor also prepends RETRIEVED MEMORIES: block if set)
  │   │     call_llm() → vLLM / OpenAI / HuggingFace
  │   │     extract_action() → regex → action string
  │   │     create_id_based_action() → structured Action dict
  │   ├─ if RETRIEVE_MEMORY action:
  │   │     memory_client.retrieve(query) → store in meta_data["retrieved_memories"]
  │   │     (browser state unchanged; current state re-appended to trajectory)
  │   ├─ else: env.step(action) → Playwright executes in real browser → new obs
  │   └─ trajectory.append(state + action)
  │
  ├─ evaluator_router(config) → score 0.0 or 1.0 → PASS / FAIL
  └─ [optional] agent.extract_and_save_memories(trajectory, intent, score)
```

---

## How to Run

```bash
# Set service URLs (set by run_experiment.sh on HPC; set manually for local runs)
export SHOPPING="http://localhost:7770"
export GITLAB="http://localhost:8023"
export REDDIT="http://localhost:9999"
# ... etc.

# Baseline run (no memory) — tasks 0–9, vLLM backend
python run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
  --test_start_idx 0 \
  --test_end_idx 10 \
  --provider vllm \
  --model Qwen/Qwen3-8B \
  --mode chat \
  --exclude_sites map \
  --result_dir results/my_run/

# Memory-enabled run (requires retriever server running at port 8020)
python run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s_memory.json \
  --test_start_idx 0 \
  --test_end_idx 10 \
  --provider vllm \
  --model Qwen/Qwen3-8B \
  --mode chat \
  --exclude_sites map \
  --retriever_server_url http://localhost:8020 \
  --top_k 3 \
  --result_dir results/memory_run/
```

On the HPC cluster, use `memorybank/submit_experiment.sh` which handles vLLM startup, service readiness polling, and SLURM submission automatically.

---

## Key Design Decisions

- **Closed-loop evaluation**: real browser + real web apps, not mocked HTML.
- **Binary reward**: `1.0` if browser action executes without exception; task success is evaluated separately by `evaluation_harness/` after the episode ends.
- **No auto-termination**: episode runs until `STOP` action or `max_steps`.
- **Element ID system**: accessibility tree nodes are assigned integer IDs per observation step; the agent refers to them in actions (e.g., `click [42]`).
- **Resume support**: `get_unfinished()` skips tasks that already have an HTML result file.

---

## Output Files

```
result_dir/
├── config.json              all CLI args (reproducibility)
├── log_files.txt            path to the .log file
├── error.txt                tracebacks for unhandled exceptions
├── traces/{task_id}.zip     Playwright browser trace (if enabled)
└── render_{task_id}.html    step-by-step action + screenshot visualization
```
