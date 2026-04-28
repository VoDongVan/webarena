# HPC Deployment — UMass Unity Cluster

This project runs on the Unity HPC cluster. Docker is not available, so all
WebArena services run under **Apptainer** (formerly Singularity) and inference
runs via a local **vLLM** server. Everything is submitted through **SLURM**.

---

## Repository Layout (this fork)

```
webarena/               ← this repo (benchmark + agent code)
webarena_build/         ← sibling directory: Apptainer SIF images + SLURM launch scripts
```

`webarena_build/` is at `../webarena_build/` relative to this repo's root.

---

## Running an Experiment

### Quick start

```bash
cd webarena/memorybank
bash submit_experiment.sh configs/webarena_baseline.yaml
```

`submit_experiment.sh` reads the config YAML, resolves the model name from
`configs/llm_clients.yaml`, picks the right GPU constraint and wall-clock time,
then calls `sbatch run_experiment.sh`.

### What `run_experiment.sh` does

1. **Parallel startup**: launches WebArena services (`launch_all.sh`) and vLLM
   simultaneously as background processes, then `wait`s for both.
2. **Polls readiness**: checks each service's node-discovery file
   (`webarena_build/homepage/.{site}_node`) and HTTP 200 from each port.
   Times out after 45 minutes.
3. **Exports env vars**: reads node hostnames from the discovery files and
   exports both `WA_*` and legacy `SHOPPING/REDDIT/...` names.
4. **Runs the experiment**: calls `python scripts/generate_test_data.py` then
   `python run.py` with the configured model and task range.
5. **Cleanup trap**: kills vLLM on EXIT.

### Shutting down WebArena servers

Always stop servers before re-submitting:

```bash
bash webarena_build/stop_all.sh
```

This cancels all running WebArena SLURM jobs (homepage, reddit, gitlab,
wikipedia, shopping, shopping_admin).

---

## WebArena Services on Apptainer

Each service is a separate SLURM job running on a CPU node. All service jobs use a **24-hour** wall-clock limit (`#SBATCH -t 24:00:00`) to match the agent job's maximum run time. Startup times:

| Service | Port | Node variable | Typical startup |
|---|---|---|---|
| Shopping (Magento storefront) | 7770 | `.shopping_node` | ~3 min |
| Shopping Admin (Magento admin) | 7780 | `.shopping_admin_node` | ~6 min |
| Reddit | 9999 | `.reddit_node` | ~1 min |
| GitLab | 8023 | `.gitlab_node` | ~4 min |
| Wikipedia | 8888 | `.wikipedia_node` | ~1 min |
| Homepage | 4399 | `.homepage_node` | ~30 s |

**Map (OpenStreetMap) is excluded** — no map server is deployed. Always pass
`--exclude_sites map` to `run.py` or add `map` to the sites exclusion list.

### Dynamic hostnames

Nodes are assigned by SLURM at job start — hostnames are not static. Each
service script writes its assigned hostname to
`webarena_build/homepage/.{site}_node` **only after the service is fully
ready**. The experiment job polls these files before proceeding.

### Shopping Admin — known issue (fixed)

Magento's root URL (`http://NODE:7780/`) serves the **storefront**, not the
admin login. The admin login page is at `/admin/`. `auto_login.py` was fixed
(commit `22f0b8b`) to navigate to `{SHOPPING_ADMIN}/admin/` for the
shopping_admin login flow.

The `run_shopping_admin.sh` warmup checks `/admin/` specifically (not root)
before writing the node-discovery file, so the readiness signal is correct.

---

## vLLM Backend

vLLM is started on the **same GPU node** as the experiment job, serving on
`localhost:8010`. It is configured with:

```
--reasoning-parser qwen3     # strips <think>...</think> from output
--max-model-len 32768
--gpu-memory-utilization 0.85
```

### Provider = "vllm"

`lm_config.py` treats `provider="vllm"` identically to `provider="openai"` at
the API level — vLLM exposes an OpenAI-compatible `/v1/chat/completions`
endpoint. The distinction is that `construct_llm_config()` reads
`VLLM_API_BASE` from the environment and stores it in `gen_config["api_base"]`
so `openai_utils.py` can build the lazy singleton client correctly.

The client is created once per process via a module-level singleton in
`llms/providers/openai_utils.py`, initialized from `VLLM_API_BASE` /
`VLLM_API_KEY` env vars. `VLLM_API_KEY` is set to `"abc"` (any non-empty
string works).

### GPU constraints by model size

`submit_experiment.sh` auto-selects the SLURM `--constraint`:

| Model | Constraint | Wall time |
|---|---|---|
| Qwen3.5-0.8B | `vram16\|vram23\|vram40` | 1 h |
| Qwen3.5-4B | `vram23\|vram40` | 2 h |
| Qwen3-8B / Qwen3.5-9B | `vram40\|vram48` | 3 h |
| Qwen3.5-27B | `vram80` | 6 h |

The 27B model requires a VRAM-80 node (A100). vLLM falls back from
FlashAttention 2 to Triton attention on V100 (compute capability 7.0).

### HuggingFace model cache

```
export HF_HOME=/work/pi_rrahimi_umass_edu/vanvo/huggingface
```

Set in `run_experiment.sh`. Models are pre-downloaded here; vLLM reads them
from this path at startup.

---

## Log and Result Locations

```
memorybank/logs/
├── wa_<SLURM_JOB_ID>.out       # stdout/stderr of the experiment job
└── vllm_<SLURM_JOB_ID>.log     # vLLM server log

memorybank/results/
├── config.json                 # CLI args snapshot
├── error.txt                   # unhandled exception tracebacks
├── log_files.txt               # path to the .log file for this run
├── render_<task_id>.html       # step-by-step browser visualization
└── traces/                     # Playwright browser traces (if enabled)
```

---

## Available Models (`configs/llm_clients.yaml`)

| Config key | Model | Notes |
|---|---|---|
| `qwen3_0.8b` | Qwen/Qwen3.5-0.8B | Fast; good for smoke tests |
| `qwen3_4b` | Qwen/Qwen3.5-4B | |
| `qwen3` | Qwen/Qwen3-8B | Default |
| `qwen3_9b` | Qwen/Qwen3.5-9B | Thinking enabled |
| `qwen3_27b` | Qwen/Qwen3.5-27B | Best quality; needs A100 |

All Qwen3.5 models have `enable_thinking: true` and use the `qwen3`
reasoning parser in vLLM (strips chain-of-thought `<think>` blocks from the
output before action parsing).

---

## Task Selection

The 812 task JSON files live in `config_files/`. Task indices are 0-based.

```bash
# Smoke test (task 0 only)
--test_start_idx 0 --test_end_idx 1

# Full run (all non-map tasks — map is filtered by --exclude_sites)
--test_start_idx 0 --test_end_idx 812 --exclude_sites map
```

Task 0 is a shopping task. Use it as a quick integration check before
committing to a full run.

---

## Accounts & Credentials

Defined in `browser_env/env_config.py`. For this deployment:

| Site | Username | Password |
|---|---|---|
| Shopping | `emma.lopez@gmail.com` | `Password.123` |
| Shopping Admin | `admin` | `admin1234` |
| Reddit | `MarvelsGrantMan136` | `notarobot` |
| GitLab | `root` | `webarena1234!` |

---

## Research Extension: Memory Retrieval

Memory retrieval is implemented as **API-level tool calling**, not as a browser action.
The LLM calls `retrieve_memory(query)` as an OpenAI function-calling tool inside
`PromptAgent.next_action()`. The tool loop runs as many retrieval rounds as the model
wants before it commits to a browser action. After each task, an extraction LLM distills
1–3 generalizable lessons and saves them to the retriever server.

### Architecture

```
PromptAgent.next_action()
  tool loop:
    call_llm(..., tools=[_RETRIEVE_MEMORY_TOOL])
    if tool_call → memory_client.retrieve(query) → append result → repeat
    else         → parse browser action → return

run.py (after each task):
  extract_and_save_memories(trajectory, intent, score)
    → call_llm(extraction_lm_config, ...)  → parse _MemoryItem list
    → memory_client.add_memories(items)
    → memory_client.save_memories(memory_save_path)   ← periodic flush
```

Memories are flushed to disk after **every task** (not just at the end), so partial
results survive job cancellation.

### Running with memory

The `submit_memory_experiment.sh` script handles everything — use it instead of
`submit_experiment.sh`:

```bash
cd webarena/memorybank
bash submit_memory_experiment.sh configs/webarena_memory_bm25_27b_test.yaml
```

This submits a CPU services job that:
1. Spins up all 6 WebArena services as separate SLURM jobs
2. Polls until all services are healthy
3. Submits the GPU job (`run_memory_experiment.sh`) with vLLM + the experiment

The retriever server is started **inside** the GPU job automatically — no manual server
launch needed.

### vLLM flags required for tool calling

```bash
--enable-auto-tool-choice
--tool-call-parser qwen3_xml    # Qwen3.5 generates XML-format tool calls
--reasoning-parser qwen3        # strip <think> blocks
```

The parser must be `qwen3_xml` (underscore). `hermes` (JSON format) and `qwen3xml`
(missing underscore) both fail for Qwen3.5 models.

### Experiment configs

| Config | Model | Tasks | Purpose |
|---|---|---|---|
| `webarena_memory_bm25_9b_test.yaml` | Qwen3.5-9B | 0–9 | Smoke test |
| `webarena_memory_bm25_27b_test.yaml` | Qwen3.5-27B | 0–9 | Smoke test |
| `webarena_memory_bm25_27b.yaml` | Qwen3.5-27B | 0–50 | Full run |

### Memory storage

Memories accumulate in the retriever server's in-memory BM25 index throughout the run.
After each task's extraction, `save_memories` flushes the index to:

```
memorybank/memories/bm25_qwen3_27b/memories.json   # 27B run
memorybank/memories/bm25_qwen3_9b/memories.json    # 9B run
```

Pass `--memories_init_path` to pre-load memories from a previous run.

### Baseline (no memory)

Omit `--retriever_server_url` and use `p_cot_id_actree_2s.json` (no `_memory` suffix).

---

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `Playwright TimeoutError: waiting for get_by_placeholder("user name")` | `auto_login.py` navigated to storefront root instead of `/admin/` | Fixed in commit `22f0b8b` |
| `Executable doesn't exist at .../chromium-*/chrome-linux/chrome` | Playwright browsers not installed in conda env | Run `playwright install chromium` |
| `AttributeError: module 'openai' has no attribute 'error'` | Old `openai.error` import; openai SDK ≥ 1.0 changed the API | Fixed in commit `d8def64` |
| `vLLM failed` in job output | vLLM took > 30 min to start (rare on cold cache miss) | Re-submit; model weights should be cached in `HF_HOME` |
| Shopping Admin node file never written | Magento cache flush or Elasticsearch slow to start | Job will retry health checks for 45 min; usually resolves |
| `AssertionError` in evaluator / `ERR_CONNECTION_REFUSED` at end of run | WA service SLURM jobs (8h limit) expired before the agent job finished | Fixed: all service jobs now use 24h limit |
| `AssertionError` or `AttributeError: 'NoneType'` in `llm_fuzzy_match` | Evaluator LLM response didn't contain `"correct"`/`"incorrect"` verbatim (Qwen thinks freely) | Fixed: prompts now explicitly demand a single-word verdict; `None` and unrecognized responses default to `0.0` |
| `JSONDecodeError` in vLLM tool parser / tool calls never fire | Wrong `--tool-call-parser` flag. Qwen3.5 generates XML-format tool calls, not JSON. `hermes` expects JSON and silently drops calls | Use `--tool-call-parser qwen3_xml` (underscore required; `qwen3xml` is also invalid) |
| `[MemoryExtraction] Failed: No module named 'qdrant_client'` | `memory/__init__.py` imports qdrant at load time; not installed in webarena env | Fixed: `_MemoryItem` defined locally in `agent.py`; `models/prompts.py` loaded via `importlib` to bypass `__init__.py` |
