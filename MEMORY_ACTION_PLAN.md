# Plan: Memory Retrieval via Explicit Agent Action

## Goal
Extend `webarena/` to support agent-controlled memory retrieval. Instead of automatic
per-step retrieval (like memorybank/), the agent issues an explicit `retrieve_memory [query]`
action when it decides it needs past experience. This is the novel research contribution.

---

## How It Works

```
Agent sees observation
  → decides it needs past experience
  → issues: retrieve_memory [how do I add items to cart on this site?]
  → system queries retriever server (BM25 or dense)
  → memories appear in NEXT observation under "RETRIEVED MEMORIES:" section
  → agent uses them to decide its next browser action
  → memories cleared after that one step (no accumulation)
```

- Agent pays 1 step per retrieval → must be strategic
- After each task ends: extraction LLM reads trajectory → saves memories to retriever
- Over a full run, memories accumulate across tasks

---

## Open Question (needs answer before implementation)

When `retrieve_memory` fires, the browser state doesn't change.
Should the next turn show:
- **(A, recommended)** Same page observation (last AXTree, no re-render) + memories
- **(B)** Just the memories, no observation

Recommendation: A — agent needs full context to decide next action.

---

## Files to Change

| File | What changes |
|------|-------------|
| `browser_env/actions.py` | Add `RETRIEVE_MEMORY = 18` to `ActionTypes`; `create_retrieve_memory_action(query)`; handle in `action2str()` |
| `agent/prompts/raw/p_cot_id_actree_2s_memory.py` | New prompt variant: add `retrieve_memory [query]` to action list in intro; add `{memories}` field to template |
| `agent/prompts/prompt_constructor.py` | In `construct()`: inject `meta_data["retrieved_memories"]` as `RETRIEVED MEMORIES:` block above observation; clear after injecting |
| `run.py` | Intercept `RETRIEVE_MEMORY` action (no `env.step()`); store memories in `meta_data`; add extraction call after evaluator; new CLI args |
| `agent/memory_client.py` | New file: sync HTTP client for retriever server `/retrieve` and `/add_memories` endpoints |
| `agent/agent.py` | Add optional `memory_client` + `extraction_lm_config` to `PromptAgent`; add `extract_and_save_memories()` |

---

## New CLI Args for `run.py`

```
--retriever_server_url   (default: "" → retrieval disabled, baseline mode)
--top_k                  (default: 3)
--extraction_model       (optional separate model for extraction LLM)
--memory_save_path       (path to persist memory bank as JSON)
```

---

## Main Loop Change in `run.py`

```python
if action["action_type"] == ActionTypes.RETRIEVE_MEMORY:
    memories = memory_client.retrieve(action["answer"], top_k)
    meta_data["retrieved_memories"] = memories
    # do NOT call env.step() — browser state unchanged
    # trajectory records the action; state_info stays as-is
else:
    obs, _, terminated, _, info = env.step(action)   # existing path
    state_info = {"observation": obs, "info": info}
    trajectory.append(state_info)
```

---

## Prompt Template (new variant)

```
RETRIEVED MEMORIES:
{memories}

OBSERVATION:
{observation}
URL: {url}
OBJECTIVE: {objective}
PREVIOUS ACTION: {previous_action}
```

When `{memories}` is empty, the "RETRIEVED MEMORIES:" section is omitted entirely.

New action in intro:
```
Memory Action:
`retrieve_memory [query]`: Search your memory bank for relevant past experiences.
The query should describe what kind of information you are looking for.
Retrieved memories will appear in your next observation. Use this when you are
unsure how to proceed on a site you may have encountered before.
```

---

## Extraction (after each task)

After `score = evaluator(...)`:
```python
if memory_client and args.extraction_model:
    memories = extract_memories(trajectory, intent, score, extraction_lm_config)
    memory_client.add_memories(memories)
```

Extraction uses a separate LLM call with a prompt that reads the full trajectory
and extracts 1–3 generalizable `MemoryItem`s (title / context / content / polarity).
Reuses `MemoryItem` dataclass and extraction prompts from `memorybank/memory/` and
`memorybank/models/prompts.py`.

---

## Retriever Server

Run externally before starting `run.py`:
```bash
cd /scratch3/workspace/vdvo_umass_edu-CS696_S26/memorybank
python retrieval/server.py --retriever_type bm25 --port 8020
```

Pass `--retriever_server_url http://localhost:8020` to `run.py`.
Dense retriever needs vLLM embedding server (`--retriever_type dense`).

---

## Status

- [ ] Confirm open question (A vs B above)
- [ ] Implement
