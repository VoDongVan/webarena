# WebArena — Documentation Index

Quick-reference docs covering the full codebase. Read these instead of re-exploring the repo from scratch.

---

## Files

| File | Covers |
|---|---|
| [overview.md](overview.md) | What WebArena is, architecture diagram, supported web environments, task config format, how to run |
| [browser_env.md](browser_env.md) | `browser_env/` — Gymnasium environment, 18 action types, DOM/accessibility-tree observation pipeline, auto-login |
| [agent.md](agent.md) | `agent/` — PromptAgent, TeacherForcingAgent, prompt constructors (Direct/CoT), prompt template format, LLM call flow |
| [llms.md](llms.md) | `llms/` — LMConfig, call_llm() dispatcher, OpenAI/HuggingFace providers, Tokenizer, retry logic |
| [evaluation_harness.md](evaluation_harness.md) | `evaluation_harness/` — StringEvaluator, URLEvaluator, HTMLContentEvaluator, site API helpers, LLM fuzzy match |
| [run.md](run.md) | `run.py` — CLI args, startup sequence, per-task loop, early stopping, trajectory structure, output files |
| [hpc_deployment.md](hpc_deployment.md) | UMass Unity HPC setup — Apptainer services, vLLM backend, SLURM workflow, model configs, known issues |

---

## One-Paragraph Summary

WebArena spins up 7 self-hosted websites (shop, admin, Reddit clone, GitLab, Wikipedia, maps, homepage) and runs 812 natural-language tasks against them. An LLM-based `PromptAgent` sees the page as an accessibility-tree text string, calls an LLM (OpenAI or HuggingFace) to decide the next action (`click [42]`, `type [164] [query] [1]`, `stop [answer]`, etc.), and the `ScriptBrowserEnv` executes that action in a real Chromium browser via Playwright. After up to 30 steps, one or more evaluators (`StringEvaluator` / `URLEvaluator` / `HTMLContentEvaluator`) score the outcome as 0.0 or 1.0. The aggregate pass rate over all tasks is the final benchmark metric.

---

## Key Concepts

### Element IDs
Each time the browser renders a new page, `TextObservationProcessor` assigns integer IDs to every DOM/accessibility node. These IDs appear in the observation text as `[42]` and are valid only for the current step. The agent includes them in actions (e.g., `click [42]`), and the processor looks up the bounding box to execute the click.

### Trajectory
`list[StateInfo | Action]` — alternates: `[state, action, state, action, ..., action]`. Always ends with an Action. The evaluator reads the last Action's `answer` field for text tasks, and uses `env.page` directly for URL/DOM tasks.

### Scoring
All evaluators score 0.0 or 1.0. Multiple evaluators are multiplied together — every condition must pass for a task to succeed.

### Prompt Format
The prompt is: system instructions → few-shot examples (obs/action pairs) → current observation filled into a template. For CoT, examples include a reasoning paragraph ending with the action inside backticks. The action is extracted by regex: ` ```action string``` `.
