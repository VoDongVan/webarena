# llms/ — LLM Provider Abstraction

Thin abstraction layer over vLLM, OpenAI, and HuggingFace APIs. Used exclusively by `PromptAgent`. In this deployment all inference goes through a local **vLLM** server (`provider="vllm"`).

---

## File Map

| File | Role |
|---|---|
| `lm_config.py` | `LMConfig` frozen dataclass + `construct_llm_config()` |
| `utils.py` | `call_llm()` dispatcher |
| `tokenizers.py` | `Tokenizer` wrapper (tiktoken / LlamaTokenizer) |
| `providers/openai_utils.py` | OpenAI Completion + Chat API wrappers |
| `providers/hf_utils.py` | HuggingFace `text_generation` wrapper |

---

## LMConfig (`lm_config.py`)

Frozen dataclass — immutable once created.

```python
@dataclass(frozen=True)
class LMConfig:
    provider: str            # "vllm", "openai", or "huggingface"
    model: str               # e.g. "Qwen/Qwen3-8B", "gpt-4"
    model_cls: type | None   # HuggingFace model class (if local)
    tokenizer_cls: type | None
    mode: str | None         # "chat" or "completion"
    gen_config: dict         # all generation parameters
```

### gen_config Keys

**vLLM** (same keys as OpenAI; provider routed identically at the API level):
```python
{
    "temperature": float,
    "top_p": float,
    "context_length": int,
    "max_tokens": int,
    "stop_token": str | None,
    "max_obs_length": int,
    "max_retry": int,
    "api_base": str,         # from VLLM_API_BASE env var (e.g. "http://localhost:8010/v1")
}
```

**OpenAI:**
```python
{
    "temperature": float,
    "top_p": float,
    "context_length": int,
    "max_tokens": int,
    "stop_token": str | None,
    "max_obs_length": int,
    "max_retry": int,
}
```

**HuggingFace:**
```python
{
    "temperature": float,
    "top_p": float,
    "max_new_tokens": int,   # note: different key name from OpenAI/vLLM
    "stop_sequences": list[str] | None,
    "model_endpoint": str,   # HF inference endpoint URL
    "max_obs_length": int,
    "max_retry": int,
}
```

`construct_llm_config(args)` reads CLI args and builds this object. For `provider="vllm"`, it reads `VLLM_API_BASE` from the environment and stores it in `gen_config["api_base"]`.

---

## call_llm() — The Dispatcher (`utils.py`)

```python
def call_llm(lm_config: LMConfig, prompt: str | list) -> str:
```

Routes by `provider` and `mode`:

```
provider="vllm",    mode="chat"       → generate_from_openai_chat_completion(messages=prompt)
provider="vllm",    mode="completion" → generate_from_openai_completion(prompt=prompt)
provider="openai",  mode="chat"       → generate_from_openai_chat_completion(messages=prompt)
provider="openai",  mode="completion" → generate_from_openai_completion(prompt=prompt)
provider="huggingface"                → generate_from_huggingface_completion(prompt=prompt)
```

- `vllm` and `openai` both route to the same `openai_utils.py` functions. The difference is the client endpoint: vLLM reads `gen_config["api_base"]` (set from `VLLM_API_BASE`) while OpenAI uses the default `api.openai.com`.
- Chat mode expects `prompt` to be `list[dict]` (messages array) — asserted at runtime.
- Completion mode and HuggingFace expect `prompt` to be a `str`.

---

## Provider Details

### vLLM / OpenAI (`providers/openai_utils.py`)

vLLM exposes an OpenAI-compatible `/v1/chat/completions` endpoint, so both providers use the same code. The client is a module-level lazy singleton initialized from `VLLM_API_BASE` / `VLLM_API_KEY` env vars (for vLLM) or the standard OpenAI env vars.

**`generate_from_openai_chat_completion(messages, model, temperature, top_p, context_length, max_tokens, ...)`**
- Calls `openai.OpenAI(base_url=..., api_key=...).chat.completions.create()` (openai SDK ≥ 1.0)
- Returns `response.choices[0].message.content`

**`generate_from_openai_completion(prompt, engine, temperature, max_tokens, top_p, stop_token, ...)`**
- Calls `client.completions.create()`
- Returns `response.choices[0].text`

**Error handling:** Both are wrapped with `retry_with_exponential_backoff`:
- Retries up to 3 times on `RateLimitError` / `APIError`
- Exponential backoff with random jitter: `delay *= base * (1 + jitter * random())`

**Async batch variants:** `agenerate_from_openai_chat_completion()` / `agenerate_from_openai_completion()`
- Accept lists of prompts
- Rate-limited via `aiolimiter.AsyncLimiter` (default 300 req/min)
- Progress shown via `tqdm_asyncio`
- Used for offline batch evaluation, not the main agent loop

**Debug stub:** `fake_generate_from_openai_chat_completion()` returns a hardcoded string for testing without API calls.

### HuggingFace (`providers/hf_utils.py`)

**`generate_from_huggingface_completion(prompt, model_endpoint, temperature, top_p, max_new_tokens, stop_sequences)`**
- Creates `text_generation.Client(endpoint, timeout=60)`
- Calls `client.generate(prompt, ...)`
- Returns `generation.generated_text`
- No retry logic

---

## Tokenizer (`tokenizers.py`)

Used by `PromptConstructor` to truncate observations to `max_obs_length` tokens.

```python
class Tokenizer:
    def __init__(self, provider: str, model_name: str)
    def encode(text: str) -> list[int]
    def decode(ids: list[int]) -> str
    def __call__(text: str) -> list[int]  # alias for encode
```

| Provider | Backend |
|---|---|
| `"vllm"` | `tiktoken.encoding_for_model(model_name)` (tiktoken, same as openai) |
| `"openai"` | `tiktoken.encoding_for_model(model_name)` |
| `"huggingface"` | `LlamaTokenizer.from_pretrained(model_name)` (no special tokens) |

### How it's used in PromptConstructor

```python
if max_obs_length:
    obs = tokenizer.decode(tokenizer.encode(obs)[:max_obs_length])
```

This hard-truncates the accessibility tree string to fit within the model's context window.

---

## Integration with PromptAgent

```python
# agent/agent.py — construct_agent()
lm_config = construct_llm_config(args)   # reads VLLM_API_BASE for provider="vllm"
tokenizer = Tokenizer(args.provider, args.model)
constructor = CoTPromptConstructor(instruction_path, lm_config, tokenizer)
agent = PromptAgent(action_set_tag, lm_config, constructor)

# agent/agent.py — PromptAgent.next_action()
prompt = constructor.construct(trajectory, intent, meta_data)
response = call_llm(lm_config, prompt)   # ← this module; routes to vLLM or OpenAI
action_str = constructor.extract_action(response)
```

---

## Adding a New Provider

1. Create `providers/newprovider_utils.py` with a `generate_from_newprovider(...)` function.
2. Add a branch in `call_llm()` in `utils.py`.
3. Add a case in `construct_llm_config()` in `lm_config.py` to populate `gen_config`.
4. Update `Tokenizer.__init__()` in `tokenizers.py` if the provider needs a different tokenizer.
