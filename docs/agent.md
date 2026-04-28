# agent/ — Agent Architecture

The decision-making layer. Takes observations from the browser and produces actions using an LLM.

---

## Class Hierarchy

```
Agent  (abstract base, agent.py:28)
  ├── TeacherForcingAgent   replays a fixed action sequence (debugging)
  └── PromptAgent           calls an LLM each step (main agent)
```

**Factory:** `construct_agent(args)` reads CLI args and returns the appropriate agent.

---

## Agent (base class, `agent.py:28`)

Defines the interface:

```python
def next_action(trajectory: Trajectory, intent: str, meta_data: Any) -> Action
def reset(test_config_file: str) -> None
```

---

## TeacherForcingAgent (`agent.py:47`)

Used for debugging and replaying reference solutions.

- **`reset(config_file)`** — loads the reference action sequence from the task JSON.
- **`set_actions(action_seq)`** — parses each action string into a structured `Action`. Failed parses become `create_none_action()`.
- **`next_action(...)`** — returns `self.actions.pop(0)`. No LLM involved.

---

## PromptAgent (`agent.py:100`)

The main LLM-driven agent.

### Constructor

```python
PromptAgent(
    action_set_tag: str,                    # "id_accessibility_tree" or "playwright"
    lm_config: LMConfig,                    # model + generation settings
    prompt_constructor: PromptConstructor,  # formats obs → prompt, parses response → action
    memory_client: Any = None,              # MemoryClient instance (optional)
    extraction_lm_config: Any = None,       # LMConfig for memory extraction (optional)
)
```

`memory_client` and `extraction_lm_config` are wired in by `run.py` after `construct_agent()` returns (not passed to the factory), because they depend on CLI args processed after agent construction.

### `next_action()` Step-by-Step

**Baseline path** (`memory_client is None`):

```
1. prompt_constructor.construct(trajectory, intent, meta_data)
      → OpenAI-format messages list

2. Retry loop (up to max_retry):
      call_llm(lm_config, prompt)     → response string
      prepend force_prefix if set
      extract_action(response)        → "click [42]"
      create_id_based_action(...)     → Action dict
      on ActionParsingError: retry
```

**Memory path** (`memory_client` is set — tool-calling loop):

```
1. prompt_constructor.construct(...)  → messages list (mutable copy)

2. Tool loop:
      call_llm(lm_config, messages, tools=[_RETRIEVE_MEMORY_TOOL])
        → returns full message object (not a string)

      if msg.tool_calls:
          append assistant turn (with tool_calls) to messages
          for each tool call:
              parse query from tc.function.arguments
              result = memory_client.retrieve(query, top_k)
              append tool result message
          continue (call LLM again with updated messages)

      else:  ← no tool calls, model chose a browser action
          response = force_prefix + msg.content
          extract_action(response) → browser action string
          break

3. return Action
```

`call_llm` passes `tools` to `generate_from_openai_chat_completion`, which returns the full
message object when tools are provided and a plain string otherwise. The tool loop can run
multiple rounds — the LLM may call `retrieve_memory` several times before committing to an action.

### `extract_and_save_memories(trajectory, intent, score)` (`agent.py:313`)

Called by `run.py` after each task if `memory_client` and `extraction_lm_config` are set.

```
1. Builds a text summary of the trajectory (objective + obs snippets + raw actions)
2. Selects prompt key: "success_extraction" if score==1, else "failure_extraction"
3. Loads webarena_prompts from memorybank/models/prompts.py via importlib.util
   (direct file load — bypasses models/__init__.py which imports fastembed/qdrant,
    neither of which is installed in the webarena conda env)
4. Calls call_llm(extraction_lm_config, messages)  → response string
5. Parses _MemoryItem objects via _MemoryItem.from_string(response)
   (_MemoryItem is defined locally in agent.py for the same dep-isolation reason)
6. Calls memory_client.add_memories(items)
```

Failures are silently swallowed — extraction never crashes the main loop.

**Extraction prompts** (`memorybank/models/prompts.py → webarena_prompts`):
- `"success_extraction"` — asks for transferable navigation strategies from a passing trajectory
- `"failure_extraction"` — asks for failure lessons and warning signs from a failing trajectory
- Both follow the SWE-Bench prompt style: "what makes a good memory / bad memory", analysis steps, and a concrete output format with `<extracted_memories>` tags
- Output format: `# Memory Item N` → `## Title` / `## Context` / `## Content` blocks (no `## Polarity`)

### `construct_agent(args)` — Factory Function

```
1. lm_config = construct_llm_config(args)
2. if agent_type == "teacher_forcing": return TeacherForcingAgent()
3. if agent_type == "prompt":
      load instruction JSON → get prompt_constructor class name string
      tokenizer = Tokenizer(provider, model)
      constructor = eval(class_name)(instruction_path, lm_config, tokenizer)
      return PromptAgent(action_set_tag, lm_config, constructor)
      # memory_client + extraction_lm_config wired in run.py afterward
```

Note: the class name is evaluated with `eval()` — it must be one of the `PromptConstructor` subclasses importable in scope.

---

## Prompt Constructors (`prompts/prompt_constructor.py`)

### PromptConstructor (base, line 23)

Loaded from an instruction JSON with four fields:

```json
{
  "intro":     "You are an autonomous intelligent agent...",
  "examples":  [["<observation>", "<action>"], ...],
  "template":  "OBSERVATION:\n{observation}\nURL: {url}\nOBJECTIVE: {objective}\nPREVIOUS ACTION: {previous_action}",
  "meta_data": {
    "observation": "accessibility_tree",
    "action_type": "id_accessibility_tree",
    "prompt_constructor": "CoTPromptConstructor",
    "action_splitter": "```",
    "keywords": ["url", "objective", "observation", "previous_action"]
  }
}
```

**`get_lm_api_input(intro, examples, current)`** — formats for the target provider:

| Provider + mode | Output format |
|---|---|
| `vllm` chat | `list[{"role": ..., "content": ...}]` — system intro, alternating user/assistant, then user |
| `vllm` completion | Single formatted string |
| OpenAI chat | Same list format as vllm chat but with `example_user`/`example_assistant` system names |
| OpenAI completion | Single formatted string |
| HuggingFace Llama-2 chat | String with `[INST]`/`[/INST]` and `<<SYS>>`/`<</SYS>>` tokens |

**`extract_action(response)`** — calls `_extract_action()` then maps real-world URLs back to local test URLs.

### DirectPromptConstructor (line 148)

Agent predicts the action directly, no reasoning.

**`construct()`:**
1. Grabs `trajectory[-1]["observation"]["text"]` (latest accessibility tree)
2. Truncates to `max_obs_length` tokens using the tokenizer
3. Fills `{observation}`, `{url}`, `{objective}`, `{previous_action}` into the template
4. Calls `get_lm_api_input()` and returns the prompt

**`_extract_action(response)`:**  
Regex `r"```((.|\n)*?)```"` — finds text between backticks. Raises `ActionParsingError` if not found.

### CoTPromptConstructor (line 228)

Agent reasons step-by-step before the action. Identical `construct()`.

The key difference is in the examples: each one ends with:
> *"In summary, the next action I will perform is ```click [42]```"*

`_extract_action()` uses the same backtick regex, so it only captures the final action and ignores the reasoning text before it.

### MemoryCoTPromptConstructor (line 285)

Vestigial — no longer used for memory injection. Memory retrieval now happens inside
`PromptAgent.next_action()` via API-level tool calls, not through the prompt template.
This class remains in the codebase but its `construct()` is a no-op pass-through to
`CoTPromptConstructor`.

---

## Prompt Template Files (`prompts/raw/`)

Stored as Python dicts, converted to JSON via `to_json.py`. The JSON files land in `prompts/jsons/` and are regenerated at run-start by `prepare()`.

| File | Constructor | Shots | Notes |
|---|---|---|---|
| `p_direct_id_actree_2s.py` | `DirectPromptConstructor` | 2 | Standard |
| `p_cot_id_actree_2s.py` | `CoTPromptConstructor` | 2 | Standard; default for vLLM runs |
| `p_direct_id_actree_2s_no_na.py` | `DirectPromptConstructor` | 2 | No N/A option for impossible tasks |
| `p_cot_id_actree_2s_no_na.py` | `CoTPromptConstructor` | 2 | No N/A option for impossible tasks |
| `p_direct_id_actree_3s_llama.py` | `DirectPromptConstructor` | 3 | Llama-2, has `force_prefix: "```"` |
| `p_cot_id_actree_2s_memory.py` | `CoTPromptConstructor` | 2 | Adds `## Reasoning Memory` tool guidance; no `{memories}` slot — retrieval via tool call |

### What a Full Prompt Looks Like (CoT, OpenAI chat)

```
System: You are an autonomous intelligent agent...
         [action syntax and rules]

System (example_user):
OBSERVATION:
[1744] link 'HP ZBook Firefly...'
[1758] StaticText '$279.49'
URL: http://onestopshop.com/product/...
OBJECTIVE: What is the price of HP Inkjet Fax Machine?
PREVIOUS ACTION: None

System (example_assistant):
Let's think step-by-step. This page shows the HP Inkjet Fax Machine priced at $279.49.
I have achieved the objective. In summary, the next action I will perform is ```stop [$279.49]```

[... more examples ...]

User:
OBSERVATION:
[current accessibility tree]
URL: http://...
OBJECTIVE: [actual task]
PREVIOUS ACTION: [last action string]
```

---

## New File: `agent/memory_client.py`

Sync HTTP client for the retriever server (started externally before `run.py`).

```python
class MemoryClient:
    def __init__(self, base_url: str)
    def retrieve(query: str, top_k: int = 3) -> str
        # POST /retrieve → returns formatted "[1] title: content\n[2] ..." string
    def add_memories(items: list[MemoryItem]) -> None
        # POST /add_memories
```

`retrieve()` returns an empty string if no memories are found, so the prompt section is omitted cleanly.

---

## meta_data — Action History Sidecar

Passed to `next_action()` on every step. Provides one-step memory without parsing the trajectory:

```python
meta_data = {
    "action_history": ["click [42]", "type [164] [query] [1]", ...],
    # Set by run.py when agent issues retrieve_memory:
    "retrieved_memories": "...",   # consumed + cleared by MemoryCoTPromptConstructor
}
```

`action_history[-1]` maps to `{previous_action}` in the prompt template.

---

## Full Data Flow

```
run.py
  └── PromptAgent.next_action(trajectory, intent, meta_data)
        │
        ├─ PromptConstructor.construct()
        │       tokenizer.encode(obs)[:max_obs_length]    truncate
        │       fill template → get_lm_api_input()
        │       → list of messages (vLLM / OpenAI format)
        │
        ├─ [memory path] tool-call loop:
        │       call_llm(lm_config, messages, tools=[_RETRIEVE_MEMORY_TOOL])
        │         → msg object
        │       if msg.tool_calls:
        │           memory_client.retrieve(query, top_k) → memory string
        │           append tool result → repeat
        │       else:
        │           response = msg.content → proceed to action parse
        │
        ├─ [baseline path]
        │       call_llm(lm_config, messages)  → response string
        │
        ├─ extract_action(response) → "click [42]"
        │
        └─ create_id_based_action("click [42]")
                → Action{action_type=CLICK, element_id="42", raw_prediction="..."}
```
