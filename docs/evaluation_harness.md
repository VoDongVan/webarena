# evaluation_harness/ — Task Success Evaluation

Scores the agent's trajectory after the episode ends. Completely independent of the agent and browser environment.

---

## File Map

| File | Role |
|---|---|
| `evaluators.py` | Evaluator classes + `evaluator_router()` factory |
| `helper_functions.py` | Site-specific API helpers + LLM fuzzy match |

---

## Evaluator Hierarchy

```
Evaluator  (abstract base)
  ├── StringEvaluator       checks the agent's text answer
  ├── URLEvaluator          checks the browser's current URL
  └── HTMLContentEvaluator  checks DOM content via JavaScript

EvaluatorComb   wraps multiple evaluators, multiplies their scores
evaluator_router(config_file) → EvaluatorComb
```

**Scoring is multiplicative**: every evaluator in the combination must return `1.0` for the task to pass.

---

## Evaluator Base Class

```python
def __call__(self, trajectory, config_file, page, client) -> float
```

- `trajectory` — full action/state history
- `config_file` — path to task JSON
- `page` — live Playwright `Page` object
- `client` — CDP session
- Returns `0.0` (fail) or `1.0` (pass)

**`get_last_action(trajectory)`** — returns `trajectory[-1]` (always an Action).  
**`get_last_state(trajectory)`** — returns `trajectory[-2]` (the last StateInfo).

---

## StringEvaluator

Used when the task answer is a text string. The agent must emit `stop [answer]`.

### Matching Strategies (all multiplicative)

**`exact_match(ref, pred)`**
- Lowercases and strips quotes from both sides
- Returns `1.0` if equal

**`must_include(ref, pred, tokenize=False)`**
- Checks if every phrase in a list appears as a substring
- `tokenize=True` (used when list has one item): splits prediction into tokens first to avoid false positives

**`fuzzy_match(ref, pred, intent)`**
- Calls GPT-4-turbo via `llm_fuzzy_match()`
- Returns `1.0` if semantically equivalent
- Used when paraphrased answers should count as correct

**`ua_match(ref, pred, intent)`**  
Special case for **unachievable tasks** (`fuzzy_match: "N/A"` in config):
- Agent should answer `N/A` and provide a reason
- Calls `llm_ua_match()` to verify the reason matches the real reason

### Config Shape

```json
"eval": {
  "eval_types": ["string_match"],
  "reference_answers": {
    "exact_match": "single string"
  }
}
```
or
```json
"reference_answers": {
  "must_include": ["phrase1", "phrase2"]
}
```
or
```json
"reference_answers": {
  "fuzzy_match": ["reference answer 1", "reference answer 2"]
}
```
or (unachievable task)
```json
"reference_answers": {
  "fuzzy_match": "N/A"
},
"string_note": "The task cannot be done because X"
```

---

## URLEvaluator

Checks whether `page.url` matches a reference URL after the episode ends.

### Config Shape

```json
"eval": {
  "eval_types": ["url_match"],
  "reference_url": "https://example.com/path?param=val |OR| https://alt.com/path",
  "url_note": "GOLD in PRED"
}
```

### Matching Logic (`"GOLD in PRED"`)

- **base_score**: `1.0` if any reference base path is a substring of the actual URL path
- **query_score**: for each query parameter key in the reference, `1.0` if the actual URL has a matching value
- Final: `base_score × query_score`

Multiple reference URLs separated by ` |OR| ` — any match counts.

---

## HTMLContentEvaluator

Verifies specific content exists on the page by navigating to a URL and running JavaScript selectors.

### Config Shape

```json
"eval": {
  "eval_types": ["program_html"],
  "program_html": [
    {
      "url": "https://example.com/page",
      "locator": "document.querySelector('h1').outerText",
      "required_contents": {
        "exact_match": "Expected Title"
      }
    }
  ]
}
```

### Per-Target Logic

1. **Navigate** to target URL:
   - Literal URL → navigate with 3s wait
   - `"last"` → stay on current page
   - `"func:shopping_get_latest_order_url()"` → call Python helper to compute URL dynamically

2. **Extract content** using locator:
   - Empty → `page.content()` (full HTML)
   - `"document.querySelector(...)"` → `page.evaluate("() => " + locator)`
   - `"func:gitlab_get_project_memeber_role(__page__, 'username')"` → call Python helper with live page

3. **Match** content:
   - `exact_match`: string equality (after HTML entity unescaping)
   - `must_include`: list of required substrings; each entry can have ` |OR| ` alternatives

Score is `1.0` only if all targets match.

---

## EvaluatorComb

```python
class EvaluatorComb:
    def __init__(self, evaluators: list[Evaluator])
    def __call__(self, ...) -> float:
        score = 1.0
        for e in self.evaluators:
            score *= e(...)
        return score
```

---

## evaluator_router()

```python
evaluator_router(config_file) -> EvaluatorComb
```

Reads `config["eval"]["eval_types"]` and instantiates evaluators:

| eval_type | Evaluator |
|---|---|
| `"string_match"` | `StringEvaluator()` |
| `"url_match"` | `URLEvaluator()` |
| `"program_html"` | `HTMLContentEvaluator()` |

Returns `EvaluatorComb` of all matching evaluators.

---

## Helper Functions (`helper_functions.py`)

### Shopping (Magento REST API)

| Function | What it does |
|---|---|
| `shopping_get_auth_token()` | POSTs admin credentials, returns Bearer token |
| `shopping_get_latest_order_url()` | GETs `/rest/V1/orders?sort=created_at DESC`, returns URL of newest order |
| `shopping_get_sku_latest_review_author(sku)` | GETs product reviews, returns last reviewer's nickname |
| `shopping_get_sku_latest_review_rating(sku)` | GETs product reviews, returns last review's rating percentage |

### Reddit

| Function | What it does |
|---|---|
| `reddit_get_post_url(url)` | Normalizes Reddit URL to canonical `/{f}/{subreddit}/{post_id}/` form |

### GitLab

| Function | What it does |
|---|---|
| `gitlab_get_project_memeber_role(page, account_name)` | Runs JavaScript on GitLab members page to find role of a given user |

### LLM-Based Matching

| Function | What it does |
|---|---|
| `llm_fuzzy_match(pred, ref, intent)` | LLM judge: is pred semantically equivalent to ref? → 1.0/0.0 |
| `llm_ua_match(pred, ref, intent)` | LLM judge: does pred's reason for impossibility match ref? → 1.0/0.0 |

Both functions are controlled by the `EVAL_LLM_MODEL` environment variable (default: `gpt-4-1106-preview`). Set `OPENAI_API_BASE` to redirect to a local vLLM endpoint. In `run_experiment.sh`, `EVAL_LLM_MODEL` is set to the same model used for the agent (e.g. `Qwen/Qwen3.5-27B`).

**Prompts** are designed to elicit a single-word verdict:
- `llm_fuzzy_match`: system says `"Reply with exactly one word: correct, incorrect, or partially correct"`. User message ends with `"Reply with exactly one word on its own line: correct, incorrect, or partially correct."` `max_tokens` is capped at 128.
- `llm_ua_match`: same pattern, verdict is `same` or `different`.

**Parsing**: checks for `"partially correct"` / `"incorrect"` / `"correct"` as substrings (in that order of precedence). A `None` response or one that contains none of those keywords returns `0.0` (no crash).

### PseudoPage

```python
class PseudoPage:
    def __init__(self, original_page: Page, url: str)
```
Wraps a Playwright `Page`, overriding `.url` while delegating all other attributes. Used when evaluation needs to treat the page as if it has a different URL.

---

## Integration with run.py

```python
# After the agent loop ends:
evaluator = evaluator_router(config_file)   # reads eval_types
score = evaluator(
    trajectory=trajectory,
    config_file=config_file,
    page=env.page,
    client=env.get_page_client(env.page),
)
# score == 1.0 → PASS, else → FAIL
```
