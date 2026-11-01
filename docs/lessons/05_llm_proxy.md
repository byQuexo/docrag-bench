# Lesson 05: The LLM proxy (`src/docrag/llm/`)

**Goal:** understand the gateway every LLM call goes through: one internal message
format, thin provider adapters, and cross-cutting features (cache, cost, budget,
structured output with one retry) written exactly once.

## What happens step by step

```
AnswerService / judge
      │  proxy.structured("ollama:qwen3:8b", messages, Answer)
      ▼
LLMProxy.chat ── order check ── resolve route ── cache? ── budget? ── provider.chat ── account cost ── cache store
      │                                                              │
      │                                         to_native ─▶ _call (HTTP) ─▶ from_native
      ▼
_parse (unwrap ```json fence) ─▶ Pydantic validate ─▶ on error: 1 retry with the error ─▶ else StructuredOutputError
```

### 1. One internal format (`llm/types.py`)

```python
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletion(BaseModel):
    model: str
    content: str  # final text only - reasoning/thinking is never put here
    usage: Usage = Usage()
    finish_reason: str | None = None
```

A small subset of the OpenAI chat format. Everything in docrag speaks this; only the
adapters know provider formats.

### 2. Provider = template method (`llm/base.py`)

```python
class Provider(ABC):
    async def chat(self, model, messages, response_format=None, temperature=0.0) -> ChatCompletion:
        native = self.to_native(model, messages, response_format, temperature)
        raw = await self._call(native)
        return self.from_native(model, raw)

    @abstractmethod
    def to_native(...) -> dict[str, Any]: ...
    @abstractmethod
    async def _call(self, native: dict[str, Any]) -> Any: ...
    @abstractmethod
    def from_native(self, model: str, raw: Any) -> ChatCompletion: ...
```

`chat()` is fixed; subclasses fill in the three steps. A new provider is one class.

### 3. `OpenAICompatProvider` (`llm/openai_compat.py`)

One class covers OpenAI, Groq, Mistral, DeepSeek, Gemini's OpenAI-compatible
endpoint, and local servers (Ollama, LM Studio, vLLM). Only `base_url` + key differ.

- `to_native`: messages are already OpenAI-shaped (`m.model_dump()`), plus
  `response_format`:

```python
body["response_format"] = {
    "type": "json_schema",
    "json_schema": {
        "name": response_format.__name__,
        "schema": json_schema_for(response_format),
        "strict": True,
    },
}
```
  Strict mode requires `additionalProperties: false`. Our response models use
  `ConfigDict(extra="forbid")`, so Pydantic emits it automatically.
- `_call`: `httpx.AsyncClient.post("chat/completions")`, retrying 408/409/429/5xx and
  connection errors with exponential backoff (1 s, 2 s). Other 4xx → `ProviderError`
  immediately (retrying a 400 never helps).
- `from_native`: takes `choices[0].message.content`, **drops** `reasoning_content`, and strips
  think tags:

```python
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    text = _THINK.sub("", text)
    if "</think>" in text.lower():  # opening tag missing in some templates
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return text.strip()
```

Without this, every answer from Qwen3 / DeepSeek-R1-style local models fails JSON
validation.

### 4. `AnthropicProvider` (`llm/anthropic_provider.py`)

Uses the official `anthropic` SDK (`AsyncAnthropic`). The mapping differences:

| OpenAI format | Anthropic Messages API |
|---|---|
| `{"role": "system"}` message | top-level `system` field |
| `max_tokens` optional | `max_tokens` required |
| `response_format.json_schema` | `output_config.format = {"type": "json_schema", "schema": ...}` |
| `choices[0].message.content` | list of content blocks; we keep `text` blocks, drop `thinking` |
| refusal = normal text | `stop_reason == "refusal"` with HTTP 200 → we raise `ProviderError` |
| `usage.prompt_tokens` | `usage.input_tokens` |

The file is named `anthropic_provider.py`, because a file called `anthropic.py` would shadow the
`anthropic` package on import.

### 5. `LLMProxy` (`llm/proxy.py`)

- **Routing:** `"ollama:qwen3:8b".partition(":")` → provider `ollama`, model
  `qwen3:8b` (the model name may contain colons). Providers are registered in
  `from_settings` only if their API key exists; a missing one gives a clear error.
- **Message order check** (`_check_order`): system messages first, at least one
  user message. Providers disagree on what they accept, so we enforce one portable order
  instead of debugging each provider.
- **Cache:** `stable_hash(route, messages, schema, temperature)`. The messages contain
  the rendered context and the prompt text, so a different retrieval config, prompt
  version or model gives a different key automatically. Cache hits cost 0.
- **Cost + budget:** `cost_eur(route, usage)` from `llm/pricing.py` (USD/1M tokens ×
  0.89381). Before each **uncached** call: if `stats.cost_eur >= budget_eur` →
  `BudgetExceededError`. Local and fake routes cost 0; unknown paid routes log one warning.
  If a provider reports the billed cost itself (`Usage.cost_usd`, e.g. OpenRouter),
  that number wins over the price table.
- **Structured output, exactly one retry:**

```python
completion = await self.chat(route, messages, schema, temperature)
try:
    return _parse(completion.content, schema), completion
except ValidationError as err:
    self.stats.validation_retries += 1
    retry_messages = [
        *messages,
        Message(role="assistant", content=completion.content),
        Message(role="user", content=f"Your reply did not match ... {err} ..."),
    ]
completion = await self.chat(route, retry_messages, schema, temperature)
try:
    return _parse(completion.content, schema), completion
except ValidationError as err:
    self.stats.validation_failures += 1
    raise StructuredOutputError(...) from err
```

The retry shows the model its own bad output plus the exact validation error. After that it's a
hard, counted failure (`validation_failures` in `run.json`). The only "repair" is unwrapping a
```` ```json ```` fence; there are no regex JSON-repair fallbacks.

### 6. `FakeProvider` (`llm/fake.py`)

It parses the context lines `[id] (page …) text` from the prompt and answers by citing the
first chunk with its first 8 words as the quote, so Verify passes. With
`JudgeVerdict` it returns `correct=True`. Tests can pass a custom `responder`
function (e.g. return invalid JSON once to test the retry).

## Why this design

> **Why not LiteLLM?** LiteLLM is the pragmatic choice in a product: 100+ providers
> behind one call. Here the gateway itself is what's being demonstrated: format
> mapping, message ordering, cost accounting, caching and later tracing, all in one place.
> With 2-3 providers that's a few hundred lines that we own and test. If the provider
> list grew, the adapters could be swapped for LiteLLM behind the same `Provider`
> interface.
>
> **Why httpx and not the openai SDK** for the OpenAI-compatible adapter? The conversion is
> the point of the module, and `httpx.MockTransport` makes it trivial to test. For
> Anthropic we use the official SDK (typed responses, built-in retries).
>
> **Why Pydantic + 1 retry instead of JSON repair?** Constrained decoding
> (`response_format`) makes invalid output rare; when it happens, the model fixing its
> own output with the error message is more reliable than regex heuristics, and failures
> are counted, not hidden.

## Python concepts

### ABC and abstract methods

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...
    def describe(self) -> str:               # shared "template" method
        return f"area={self.area():.1f}"

class Square(Shape):
    def __init__(self, s: float) -> None: self.s = s
    def area(self) -> float: return self.s ** 2

Shape()      # TypeError: Can't instantiate abstract class
```

Same idea as a Java abstract class, or a Rust trait with a default method.

### Bounded generics

```python
async def structured[T: BaseModel](self, ..., schema: type[T]) -> tuple[T, ChatCompletion]:
```

`T: BaseModel` ≈ Java `<T extends BaseModel>` / TS `<T extends BaseModel>`.
`type[T]` means "the class T itself", not an instance. So
`structured(..., Answer)` is typed as returning an `Answer`.

### Exceptions: `raise ... from err`

`raise StructuredOutputError(...) from err` chains the original exception
("The above exception was the direct cause of…"), like Java's `new X(msg, cause)`.
`from None` hides it.

### `try / except / else`

In `_call`, the `else:` branch runs only when the `try` didn't raise. It keeps the "success
path" separate from the "connection failed" path.

### httpx async

```python
async with httpx.AsyncClient(base_url="https://api.example.com/") as client:
    resp = await client.post("chat/completions", json=body)
```

It plays the role of `fetch` in TS, `HttpClient.sendAsync` in Java, or `reqwest` in Rust. Reuse one client
(connection pool) and close it (`aclose()`), which the API lifespan does.

## Exercises

1. Write a test using `httpx.MockTransport` that returns
   `{"choices":[{"message":{"content":"<think>hmm</think>{\"a\":1}"}}]}` and check
   `OpenAICompatProvider(...).chat()` returns content `{"a":1}`. **Done when** green.
2. Give `FakeProvider` a responder that returns `"nope"` the first time and valid JSON
   the second time; call `proxy.structured(..., Answer)`. **Done when**
   `stats.validation_retries == 1` and `validation_failures == 0`.
3. Set `LLMProxy(..., budget_eur=0.0)` with a provider whose route is in the price table
   (e.g. register `FakeProvider` under `"anthropic"` in a test). **Done when** the
   second call raises `BudgetExceededError`.
4. Call `proxy.chat` twice with identical messages and a `Cache(":memory:")`. **Done
   when** `stats.calls == 1` and `stats.cache_hits == 1`.
5. Add a `"together"` OpenAI-compatible provider to `from_settings` (base URL + key
   setting). **Done when** `resolve("together:some-model")` works with the env var set.

## Interview questions

**Q: Why did you build your own LLM proxy instead of LiteLLM?**
A: To own the gateway logic I want to demonstrate and test: format conversion, message
ordering, caching, cost, budget, and tracing later, all in one place. For two or three
providers that's small. In production with many providers I'd use LiteLLM behind the same
interface; the adapters are the only part that would change.

**Q: How do you guarantee structured output?**
A: `response_format` with a strict JSON schema from the Pydantic model (OpenAI
format) or `output_config.format` (Anthropic). The reply is validated with Pydantic;
on failure there's exactly one retry that includes the validation error, then a counted hard failure.

**Q: What happens when the model returns invalid JSON twice?**
A: `StructuredOutputError` is raised and `validation_failures` incremented. In the API
that becomes HTTP 502; in the eval, `bounded_gather` records the failure, the question
keeps its retrieval metrics, and the count appears in `run.json`.

**Q: How do you handle reasoning models?**
A: Think tags are stripped before validation and separate reasoning fields are dropped
— reasoning is never shown as answer text. For Anthropic, `thinking` blocks are
skipped and only `text` blocks are used. Reasoning tokens are still billed as output,
so for answers and judging I keep reasoning low.

**Q: How do you control cost?**
A: Usage × price per route, accumulated in `ProxyStats`; a hard budget cap stops new
uncached calls; the cache makes re-runs free; local models cost 0 during development.
The final cost is a column in the results.

**Q: What's in the LLM cache key and why is it safe?**
A: Route, the full messages (prompt + context + question), the schema and the temperature. It
has no API keys. Any change to retrieval or prompt changes the context text and so the
key; stale answers can't be reused.

**Q: Why enforce message order?**
A: Providers differ: Anthropic wants `system` as a separate field, some local chat
templates break on a system message in the middle. One portable rule (system first)
avoids provider-specific bugs.
