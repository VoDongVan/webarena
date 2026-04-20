# WebArena — Documentation Index

Quick-reference docs covering the full codebase. Read these instead of re-exploring the repo from scratch.

---

## Files

| File | Covers |
|---|---|
| [overview.md](overview.md) | What WebArena is, architecture diagram, supported web environments, task config format, how to run |
| [browser_env.md](browser_env.md) | `browser_env/` — Gymnasium environment, 19 action types (incl. `RETRIEVE_MEMORY`), DOM/accessibility-tree observation pipeline, auto-login |
| [agent.md](agent.md) | `agent/` — PromptAgent, TeacherForcingAgent, prompt constructors (Direct/CoT/MemoryCoT), memory_client.py, prompt template format, LLM call flow |
| [llms.md](llms.md) | `llms/` — LMConfig, call_llm() dispatcher, vLLM/OpenAI/HuggingFace providers, Tokenizer, retry logic |
| [evaluation_harness.md](evaluation_harness.md) | `evaluation_harness/` — StringEvaluator, URLEvaluator, HTMLContentEvaluator, site API helpers, LLM fuzzy match |
| [run.md](run.md) | `run.py` — CLI args (incl. memory args), startup sequence, per-task loop, RETRIEVE_MEMORY intercept, early stopping, trajectory structure, output files |
| [hpc_deployment.md](hpc_deployment.md) | UMass Unity HPC setup — Apptainer services, vLLM backend, SLURM workflow, model configs, memory retrieval usage, known issues |
| [results.md](results.md) | Results folder layout — how to read `render_{task_id}.html` files, use Playwright traces, interpret `error.txt`, and run the analysis script |

---

## One-Paragraph Summary

WebArena spins up 6 self-hosted websites (shop, admin, Reddit clone, GitLab, Wikipedia, homepage — map excluded in this deployment) and runs 812 natural-language tasks against them. An LLM-based `PromptAgent` sees the page as an accessibility-tree text string, calls an LLM via vLLM (OpenAI-compatible API) to decide the next action (`click [42]`, `type [164] [query] [1]`, `stop [answer]`, `retrieve_memory [query]`, etc.), and the `ScriptBrowserEnv` executes that action in a real Chromium browser via Playwright. After up to 30 steps, one or more evaluators (`StringEvaluator` / `URLEvaluator` / `HTMLContentEvaluator`) score the outcome as 0.0 or 1.0. The aggregate pass rate over all tasks is the final benchmark metric. A research extension adds agent-controlled memory retrieval: the agent issues `retrieve_memory` to query a BM25/dense retriever, and an extraction LLM saves new memories after each task.

---

## Key Concepts

### Element IDs
Each time the browser renders a new page, `TextObservationProcessor` assigns integer IDs to every DOM/accessibility node. These IDs appear in the observation text as `[42]` and are valid only for the current step. The agent includes them in actions (e.g., `click [42]`), and the processor looks up the bounding box to execute the click.

### Trajectory
`list[StateInfo | Action]` — alternates: `[state, action, state, action, ..., action]`. Always ends with an Action. The evaluator reads the last Action's `answer` field for text tasks, and uses `env.page` directly for URL/DOM tasks. `RETRIEVE_MEMORY` actions appear in the trajectory but do not advance the browser: the same `StateInfo` is re-appended immediately after, so the interleaving structure is preserved.

### Scoring
All evaluators score 0.0 or 1.0. Multiple evaluators are multiplied together — every condition must pass for a task to succeed.

### Prompt Format
The prompt is: system instructions → few-shot examples (obs/action pairs) → current observation filled into a template. For CoT, examples include a reasoning paragraph ending with the action inside backticks. The action is extracted by regex: ` ```action string``` `.
