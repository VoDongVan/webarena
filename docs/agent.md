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
    action_set_tag: str,            # "id_accessibility_tree" or "playwright"
    lm_config: LMConfig,            # model + generation settings
    prompt_constructor: PromptConstructor,  # formats obs → prompt, parses response → action
)
```

### `next_action()` Step-by-Step

```
1. prompt_constructor.construct(trajectory, intent, meta_data)
      → formatted LLM input (OpenAI messages list or string)

2. call_llm(lm_config, prompt)
      → raw LLM response string

3. prepend force_prefix if set (e.g. "```" for Llama models)

4. Retry loop (up to max_retry):
      extract_action(response)         → action string, e.g. "click [42]"
      create_id_based_action("click [42]")  → structured Action dict
      on ActionParsingError: retry
      on max retries exceeded: return create_none_action()

5. action["raw_prediction"] = response  (stored for logging)

6. return Action
```

### `construct_agent(args)` — Factory Function

```
1. lm_config = construct_llm_config(args)
2. if agent_type == "teacher_forcing": return TeacherForcingAgent()
3. if agent_type == "prompt":
      load instruction JSON → get prompt_constructor class name string
      tokenizer = Tokenizer(provider, model)
      constructor = eval(class_name)(instruction_path, lm_config, tokenizer)
      return PromptAgent(action_set_tag, lm_config, constructor)
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
| OpenAI chat | `list[{"role": ..., "content": ...}]` — system intro, alternating example_user/assistant, then user |
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

### CoTPromptConstructor (line 206)

Agent reasons step-by-step before the action. Identical `construct()`.

The key difference is in the examples: each one ends with:
> *"In summary, the next action I will perform is ```click [42]```"*

`_extract_action()` uses the same backtick regex, so it only captures the final action and ignores the reasoning text before it.

---

## Prompt Template Files (`prompts/raw/`)

Stored as Python dicts, converted to JSON via `to_json.py`.

| File | Type | Action space | Shots | Notes |
|---|---|---|---|---|
| `p_direct_id_actree_2s.py` | Direct | accessibility tree IDs | 2 | Standard GPT |
| `p_cot_id_actree_2s.py` | CoT | accessibility tree IDs | 2 | Standard GPT |
| `p_direct_id_actree_2s_no_na.py` | Direct | accessibility tree IDs | 2 | No N/A for impossible tasks |
| `p_cot_id_actree_2s_no_na.py` | CoT | accessibility tree IDs | 2 | No N/A for impossible tasks |
| `p_direct_id_actree_3s_llama.py` | Direct | accessibility tree IDs | 3 | Llama-2, has `force_prefix: "```"` |

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

## meta_data — Action History Sidecar

Passed to `next_action()` on every step. Provides one-step memory without parsing the trajectory:

```python
meta_data = {
    "action_history": ["click [42]", "type [164] [query] [1]", ...],
    "force_prefix": "```",   # optional, for Llama prefix forcing
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
        │       → list of OpenAI messages
        │
        ├─ call_llm(lm_config, messages)
        │       → "...```click [42]```"
        │
        ├─ extract_action() → "click [42]"
        │
        └─ create_id_based_action("click [42]")
                → Action{action_type=CLICK, element_id="42", raw_prediction="..."}
```
