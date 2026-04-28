# run.py — Evaluation Orchestrator

The entry point. Wires together the agent, browser environment, and evaluator into a full evaluation loop.

---

## CLI Arguments (`config()`, lines 59–186)

### Browser / Environment

| Arg | Default | Purpose |
|---|---|---|
| `--render` | False | Show browser window (headless by default) |
| `--slow_mo` | 0 | Slow browser actions by N ms (debugging) |
| `--viewport_width` | 1280 | Browser window width |
| `--viewport_height` | 720 | Browser window height |
| `--save_trace_enabled` | False | Save Playwright trace zip per task |
| `--sleep_after_execution` | 0.0 | Wait after each action (overridden to 2.0 in main) |
| `--max_steps` | 30 | Max actions per episode |

### Observation / Action

| Arg | Default | Purpose |
|---|---|---|
| `--observation_type` | `"accessibility_tree"` | `"html"` / `"accessibility_tree"` / `"image"` |
| `--current_viewport_only` | False | Only show visible DOM nodes |
| `--action_set_tag` | `"id_accessibility_tree"` | `"id_accessibility_tree"` / `"playwright"` |

**Constraint**: `id_accessibility_tree` action set requires `accessibility_tree` observation type.

### Agent

| Arg | Default | Purpose |
|---|---|---|
| `--agent_type` | `"prompt"` | `"prompt"` or `"teacher_forcing"` |
| `--instruction_path` | (json path) | Prompt template JSON file |
| `--parsing_failure_th` | 3 | Stop after N consecutive NONE actions |
| `--repeating_action_failure_th` | 3 | Stop after N identical consecutive actions |

### LLM

| Arg | Default | Purpose |
|---|---|---|
| `--provider` | `"openai"` | `"vllm"` / `"openai"` / `"huggingface"` |
| `--model` | `"gpt-3.5-turbo-0613"` | Model name (e.g. `Qwen/Qwen3-8B` for vLLM) |
| `--mode` | `"chat"` | `"chat"` / `"completion"` |
| `--temperature` | 1.0 | Sampling temperature |
| `--top_p` | 0.9 | Nucleus sampling |
| `--max_tokens` | 384 | Max output tokens |
| `--max_obs_length` | 1920 | Truncate obs to this many tokens |
| `--max_retry` | 1 | Action parse retries per step |
| `--model_endpoint` | `""` | HuggingFace inference endpoint URL |

### Tasks & Output

| Arg | Default | Purpose |
|---|---|---|
| `--test_start_idx` | 0 | First task index to run |
| `--test_end_idx` | 1000 | Last task index (exclusive) |
| `--exclude_sites` | `["map"]` | Skip tasks involving these site names; map excluded by default |
| `--result_dir` | `""` | Where to save results (auto-named if empty) |

### Memory (optional)

| Arg | Default | Purpose |
|---|---|---|
| `--retriever_server_url` | `""` | URL of retriever server (empty = memory disabled; baseline mode) |
| `--top_k` | 3 | Number of memories to retrieve per `retrieve_memory` action |
| `--extraction_model` | `""` | Model name for the extraction LLM; uses main model if empty |
| `--memory_save_path` | `""` | Path to persist memory bank as JSON (passed to retriever server) |

All memory args default to disabled. Passing `--retriever_server_url http://localhost:8020` enables memory. The retriever server must be running before `run.py` starts (see HPC deployment doc).

---

## Startup Sequence (`main()`, lines 454–511)

```
1. args = config()
2. args.sleep_after_execution = 2.0      ← hardcoded override (page settle time)
3. prepare(args)                         ← convert prompt .py→.json, mkdir result_dir
4. test_file_list = config_files[start:end]
5. test_file_list = get_unfinished(...)  ← skip already-completed tasks
6. filter by --exclude_sites             ← drops map tasks by default
7. dump_config(args)                     ← write result_dir/config.json
8. agent = construct_agent(args)
9. if --retriever_server_url:
      memory_client = MemoryClient(url)
      agent.memory_client = memory_client
      agent.extraction_lm_config = (build from --extraction_model or reuse main lm_config)
10. test(args, agent, test_file_list, memory_client=memory_client)
```

---

## test() — The Core Loop (lines 245–405)

Signature: `test(args, agent, config_file_list, memory_client=None)`

### Environment Init

```python
env = ScriptBrowserEnv(
    headless=not args.render,
    observation_type=args.observation_type,
    viewport_size={"width": args.viewport_width, "height": args.viewport_height},
    sleep_after_execution=args.sleep_after_execution,
    ...
)
```

### Per-Task Steps

**1. Load config + refresh cookies**

Reads task JSON. If a `storage_state` path is present, calls `auto_login.py` as a subprocess to verify/renew cookies, then patches the config with the fresh cookie path.

**2. Reset**

```python
agent.reset(config_file)
obs, info = env.reset(options={"config_file": config_file})
trajectory = [{"observation": obs, "info": info}]
meta_data = {"action_history": ["None"]}
```

**3. Action loop**

```
while True:
    a. early_stop(trajectory, max_steps, thresholds)
          if triggered → inject STOP action with reason; break

    b. action = agent.next_action(trajectory, intent, meta_data)
          if ValueError → inject STOP action with error message

    c. trajectory.append(action)

    d. render_helper.render(action, ...)   → update HTML output file

    e. meta_data["action_history"].append(action_str)

    f. if action_type == STOP: break

    g. obs, _, terminated, _, info = env.step(action)
       state_info = {"observation": obs, "info": info}
       trajectory.append(state_info)

    h. if terminated: inject STOP; break
```

Note: `RETRIEVE_MEMORY` is no longer a browser action. Memory retrieval now happens
entirely inside `agent.next_action()` via the OpenAI tool-calling API — `run.py` has
no retrieval branch.

**4. Evaluate**

```python
evaluator = evaluator_router(config_file)
score = evaluator(trajectory, config_file, env.page, client)
```

**5. Memory extraction (if enabled)**

```python
if memory_client and isinstance(agent, PromptAgent) and agent.extraction_lm_config:
    agent.extract_and_save_memories(trajectory, intent, score)
    if args.memory_save_path:
        memory_client.save_memories(args.memory_save_path)  # periodic flush
```

Memories are saved to disk after **every task**, not just at the end of the run.
This ensures partial results survive job cancellation or unexpected failures.

**6. Log & save**

- `scores.append(score)`
- Log `[Result] (PASS)` or `[Result] (FAIL)`
- If trace enabled: `save_trace(result_dir/traces/{task_id}.zip)`
- Catch `OpenAIError` → log and continue
- Catch any other `Exception` → log + write traceback to `error.txt` and continue

**7. Summary**

```python
logger.info(f"Average score: {sum(scores)/len(scores) if scores else 'N/A (no completed tasks)'}")
```

---

## early_stop() (lines 161–214)

Called before every action. Returns `(should_stop: bool, reason: str)`.

| Condition | Check | Reason string |
|---|---|---|
| Max steps | `(len(trajectory) - 1) / 2 >= max_steps` | `"Reach max steps N"` |
| Parse failures | last `k` actions are all `ActionTypes.NONE` | `"Failed to parse actions for k times"` |
| Repeating actions | last `k` non-TYPE actions are equivalent, or total TYPE actions > `k` | `"Same action for k times"` |

Action equivalence is checked via `is_equivalent()` from `browser_env/actions.py`.

---

## Trajectory Structure

```
[StateInfo_0,
 Action_0,    StateInfo_1,
 Action_1,    StateInfo_2,
 ...
 Action_N]                 ← last element is always an Action (STOP or forced stop)
```

- **StateInfo**: `{"observation": {"text": ..., "image": ...}, "info": {...}}`
- **Action**: TypedDict with `action_type`, `element_id`, `answer`, `raw_prediction`, etc.

The evaluator reads:
- `trajectory[-1]["answer"]` for `StringEvaluator`
- `env.page.url` for `URLEvaluator`
- Live page DOM for `HTMLContentEvaluator`

---

## meta_data — One-Step Memory

```python
meta_data = {
    "action_history": [],     # human-readable string per past action
    "force_prefix": "```",    # optional, Llama prefix forcing
    ...
}
```

`action_history[-1]` maps to `{previous_action}` in the prompt template so the agent knows what it just did.

---

## Helper Functions

| Function | Purpose |
|---|---|
| `config()` | Parse + validate all CLI args |
| `early_stop()` | 3 stopping criteria per step |
| `test()` | Main evaluation loop |
| `prepare()` | Create output dirs, convert prompts to JSON |
| `get_unfinished()` | Filter already-completed tasks (resume support) |
| `dump_config()` | Save `config.json` for reproducibility |

---

## Output Files

```
result_dir/
├── config.json              all CLI args
├── log_files.txt            path to the .log file for this run
├── error.txt                full tracebacks for any unhandled exceptions
├── traces/{task_id}.zip     Playwright browser trace (if --save_trace_enabled)
└── render_{task_id}.html    step-by-step visualization (action + screenshot per step)
```

---

## Resume Support

`get_unfinished(config_files, result_dir)` checks for existing `.html` render files in `result_dir`. Tasks that already have an output file are skipped. This allows safely re-running after a crash without re-doing completed work.

Exception: if `"debug"` is in `result_dir`, all tasks run regardless.

---

## Full System Call Graph

```
main()
  ├── prepare()
  ├── construct_agent()              → PromptAgent or TeacherForcingAgent
  ├── [optional] MemoryClient(url)   → wired into agent.memory_client
  └── test(args, agent, tasks, memory_client)
        ├── ScriptBrowserEnv(...)
        └── for task in task_list:
              ├── auto_login.py subprocess   → fresh cookies
              ├── agent.reset()
              ├── env.reset()                → initial obs
              ├── loop:
              │     ├── early_stop()
              │     ├── agent.next_action()
              │     │     ├── PromptConstructor.construct()
              │     │     ├── [memory] tool loop:
              │     │     │     call_llm(..., tools=[_RETRIEVE_MEMORY_TOOL])
              │     │     │     if tool_call → memory_client.retrieve() → repeat
              │     │     ├── [baseline] call_llm() → response string
              │     │     └── extract_action() + create_id_based_action()
              │     ├── env.step(action)             → Playwright → Chromium
              │     └── trajectory.append(...)
              ├── evaluator_router()(trajectory, page, ...)
              │     ├── StringEvaluator    (answer text)
              │     ├── URLEvaluator       (page.url)
              │     └── HTMLContentEvaluator (DOM content)
              └── [optional] agent.extract_and_save_memories(trajectory, intent, score)
                    ├── memory_client.add_memories(items)
                    └── memory_client.save_memories(path)   ← after every task
```
