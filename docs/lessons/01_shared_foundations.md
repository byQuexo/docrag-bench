# Lesson 01: Shared foundations (`src/docrag/shared/`)

**Goal:** know the building blocks every other module uses: the data contracts,
settings vs experiment config, the cache, the timing decorator and
`bounded_gather`. Along the way you learn Python's four ways to define a "type":
Pydantic model, dataclass, Protocol and ABC.

The plan's rule: *whatever several modules need lives once in `shared/`*. And
`shared/` imports nothing from the other packages, so there are no circular imports.

## What happens step by step

### 1. `schema.py`: the contracts between stages

```python
BlockType = Literal["text", "table", "image"]


class Block(BaseModel):
    model_config = ConfigDict(frozen=True)
    doc_id: str
    type: BlockType
    text: str
    table_md: str | None = None
    caption: str | None = None
    page: int = Field(ge=1)  # 1-based, fixed in the parser
    section_path: list[str]
    order: int
```

`Parser -> list[Block] -> Chunker -> list[Chunk] -> Store -> list[Hit]`. If a stage
changes internally but still produces valid Blocks/Chunks, nothing downstream
changes. `Field(ge=1)` is a runtime constraint: `Block(page=0, ...)` raises a
`ValidationError`. That's how "pages are 1-based" is enforced in code, not only in a
comment.

`Chunk.pages` is a `@property`, i.e. a computed read-only attribute:

```python
@property
def pages(self) -> list[int]:
    return list(range(self.page_start, self.page_end + 1))
```

### 2. Two kinds of configuration

- `settings.py` → `Settings(BaseSettings)`: *environment*. Paths, Qdrant URL, API
  keys. Read from env vars with prefix `DOCRAG_` and from `.env`.
- `config.py` → `ExperimentConfig(BaseModel)`: *what is measured*. Model,
  retrieval mode, thresholds. Read from `configs/*.yaml`, hashed into the report.

```python
class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    embedder: str = "onnx:gte-small"
    retrieval: RetrievalMode = "hybrid_rerank"
    rrf_k: int = 60
    ...

    def config_hash(self) -> str:
        return stable_hash(self.model_dump(exclude={"name"}))[:12]
```

`extra="forbid"` turns a typo like `top_kk: 5` into an error instead of silently
measuring the default. `config_hash` excludes `name`: two configs that measure the
same thing get the same hash.

Secrets are `SecretStr`, and their env names are the usual ones thanks to
`AliasChoices`:

```python
openai_api_key: SecretStr | None = Field(
    default=None,
    validation_alias=AliasChoices("OPENAI_API_KEY", "DOCRAG_OPENAI_API_KEY"),
)
```

`print(settings)` shows `**********`; code must call `.get_secret_value()`
explicitly (done only in `LLMProxy.from_settings`).

`get_settings()` is a lazy singleton:

```python
@lru_cache
def get_settings() -> Settings:
    return Settings()
```

### 3. `cache.py`: the key matters more than the storage

```python
def stable_hash(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

Storage is one SQLite table `kv(ns, key, value)` with namespaces `parse`, `embed`,
`llm`. The key rules (from the RAG-Anything review):

| Include | Never include |
|---|---|
| content hash (PDF bytes, chunk text, full prompt) | file path, mtime |
| component version (`PyMuPdfParser.version`, `OnnxEmbedder.version`) | API keys / secrets |
| options that change the output (prefix on/off, schema, temperature) | streams |

Example from `ingest/pipeline.py`:

```python
def parse_cache_key(data: bytes, parser: Parser, doc_id: str) -> str:
    return f"{parser.name}:{parser.version}:{bytes_hash(data)}:{doc_id}"
```

Change the parser logic → bump `version = "2"` → old entries are simply never looked up
again. There's no manual "clear cache" step and no stale results.

Thread safety: `sqlite3.connect(..., check_same_thread=False)` plus a
`threading.Lock()` around every statement, because FastAPI and `asyncio.to_thread`
use worker threads. WAL mode reduces "database is locked" errors.

### 4. `timing.py`: a decorator with an argument

```python
def timed(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    record(name, time.perf_counter() - start)

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R: ...

        return wrapper

    return decorator
```

`@timed("embed")` above a function means `func = timed("embed")(func)`. Durations go into
a module-level `defaultdict(list)`; `percentiles(name)` gives p50/p95 for `run.json`.
It's the "Timing decorator before OTel" from the plan. Every timed function is
already marked, so adding OpenTelemetry spans later only changed `timing.py`
(and that's what happened, see [lesson 13](13_tracing.md)).

Cache hits are timed separately: a function that answers from the cache calls
`timing.cache_hit()`, and the duration goes to `<name>_cached` (e.g. `llm_cached`).
The first version mixed them, and a re-run reported "llm p50 = 3 ms", which was the
cache lookup, not the LLM. The flag lives in a `ContextVar`, so concurrent async calls
and nested timed calls each see their own.

### 5. `concurrency.py`: `bounded_gather`

```python
async def bounded_gather[T](
    coros: Iterable[Awaitable[T]], limit: int = 4
) -> GatherResult[T]:
    semaphore = asyncio.Semaphore(limit)

    async def run(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    raw = await asyncio.gather(*(run(c) for c in coros), return_exceptions=True)
    ...
```

At most `limit` LLM calls are in flight at once; errors are collected, not raised. The eval
can then report "3 of 500 failed" instead of dying after an hour.

### 6. `text.py`: one notion of "same text"

`normalize()` does NFKC (ligature "ﬁ" → "fi"), rejoins hyphenated line breaks
("infor-\nmation"), unifies quotes/dashes, lowercases and squashes whitespace.
`tokenize()` adds stopword removal. Verify and BM25 both rely on it.

## Why this design

> - **Pydantic at the boundaries, dataclasses inside.** Data that comes from
>   outside (YAML, HTTP, LLM JSON, cached JSON) is validated by Pydantic. Internal
>   helper data (`GatherResult`, `_Pending`, `ProxyStats`) uses dataclasses: lighter,
>   no validation needed.
> - **Settings vs ExperimentConfig:** a results row must never depend on your
>   local paths, and a secret must never end up in `results/`.
> - **Content-addressed cache:** the M5 criterion "second run costs 0 EUR" falls out
>   of the key design.

## Python concepts

### Four ways to define "a type"

| Python | Use for | TS | Java | Rust |
|---|---|---|---|---|
| `class X(BaseModel)` (Pydantic) | validated external data | zod schema + type | record + Bean Validation | struct + serde + validator |
| `@dataclass class X` | plain internal data | `type X = {...}` | record | struct |
| `class X(Protocol)` | interface, structural | `interface` | (no equivalent: Java is nominal) | trait-like, but no `impl` needed |
| `class X(ABC)` | base class with shared code | `abstract class` | `abstract class` | trait with default methods |

```python
from typing import Protocol


class Greeter(Protocol):
    def greet(self) -> str: ...


class English:  # no "implements Greeter"!
    def greet(self) -> str:
        return "hi"


def welcome(g: Greeter) -> None:  # mypy accepts English: it has greet()
    print(g.greet())
```

In this project: `Parser`, `Embedder`, `Reranker` and `ThreadStore` are Protocols
(any object with the right shape works, and fakes need no inheritance). `Provider` is
an ABC because it has a real shared method (`chat()`, lesson 05).

### `Literal` and `frozen`

```python
from typing import Literal

Mode = Literal["dense", "bm25"]  # TS: type Mode = "dense" | "bm25"
```

`ConfigDict(frozen=True)` = immutable + hashable (Java record, Rust struct without
`mut`). `block.text = "x"` raises.

### Decorators

A decorator is just a function that takes a function and returns a function:

```python
def shout(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs).upper()

    return wrapper


@shout
def hello(name: str) -> str:
    return f"hello {name}"


hello("tim")  # "HELLO TIM"
```

TS has decorators too (only on classes and their members); Java annotations are
metadata that a framework has to interpret; in Rust the nearest equivalent is a
proc macro. `*args, **kwargs` means "any positional / any keyword arguments",
like `...args` in TS. `ParamSpec` (`P`) lets mypy keep the original
signature through the wrapper.

### Generics (PEP 695)

```python
class Box[T]:  # class Box<T>
    def __init__(self, v: T) -> None:
        self.v = v


def first[T](xs: list[T]) -> T:  # function <T> first(...)
    return xs[0]
```

### asyncio vs Promises / CompletableFuture / Rust futures

- `async def f()` returns a **coroutine** object. Nothing runs until it's awaited
  or scheduled. Rust futures are lazy in the same way; a JS Promise is not, it starts running immediately.
- `await x` suspends until x is done; other coroutines run meanwhile **on the
  same thread** (single-threaded event loop, like Node).
- `asyncio.gather(a, b)` ≈ `Promise.all`; with `return_exceptions=True` ≈
  `Promise.allSettled`. Java: `CompletableFuture.allOf`. Rust: `join_all`.
- `asyncio.Semaphore(4)` + `async with` = at most 4 inside. Java `Semaphore`,
  Rust `tokio::sync::Semaphore`.
- asyncio helps with **waiting** (HTTP), not **computing**. CPU work blocks the
  loop. Use `asyncio.to_thread` or processes (lessons 02 and 07).

Runnable example:

```python
import asyncio, time


async def call(i: int) -> int:
    await asyncio.sleep(1)  # pretend HTTP request
    return i


async def main() -> None:
    sem = asyncio.Semaphore(2)

    async def bounded(i: int) -> int:
        async with sem:
            return await call(i)

    t = time.perf_counter()
    print(await asyncio.gather(*(bounded(i) for i in range(6))))
    print(f"{time.perf_counter() - t:.1f}s")  # ~3.0s: 6 calls, 2 at a time


asyncio.run(main())
```

## Exercises

1. Create `Block(doc_id="d", type="pic", text="", page=0, section_path=[], order=0)`
   in a Python shell. **Done when** you've read the ValidationError and can name
   both failing fields.
2. Write a test that `ExperimentConfig(name="a").config_hash() ==
   ExperimentConfig(name="b").config_hash()` but differs when `top_k` changes.
   **Done when** it passes.
3. Add `@timed("chunk")` to `chunk_blocks` and print `timing.snapshot()` after
   chunking `fixtures/tiny.pdf`. **Done when** you see `p50_ms`.
4. Run the asyncio example above with `Semaphore(6)` and `Semaphore(1)`. **Done
   when** you can predict the runtime before running.
5. Use `Cache(":memory:")`, store an embedding with `set_bytes` and read it back with
   `np.frombuffer`. **Done when** the arrays are equal.

## Interview questions

**Q: Pydantic vs dataclasses, when do you use which?**
A: Pydantic where data crosses a trust boundary: config YAML, HTTP bodies, LLM JSON,
cached JSON. It validates and gives clear errors. Dataclasses for internal
structs where the data is produced by our own code; they're lighter and faster.

**Q: What goes into a cache key, and what must never?**
A: Content hash, component version and every option that changes the result. Never
paths or mtimes (renaming shouldn't invalidate), never secrets (they would end up on
disk), and never streams. Bumping a component version invalidates exactly that
component's entries.

**Q: Why SQLite as a cache and not Redis?**
A: Single-machine benchmark, zero ops, persists across runs, and it is in the standard
library. For a multi-instance API you'd move to Redis behind the same interface; the
plan lists that as "later".

**Q: What does `bounded_gather` protect against?**
A: Rate limits and overload: 1000 questions would otherwise fire 1000 requests at once.
It also keeps going on errors and counts them, so one bad request doesn't kill a long
benchmark run.

**Q: Why is your timing a decorator instead of OpenTelemetry?**
A: It gives the p50/p95 numbers the results table needs today, with zero
infrastructure. Every measured function is already marked, so adding OTel spans
(milestone M6) changed one file: `@timed` now also opens a span inside a trace,
and the registry still feeds `run.json` (lesson 13).

**Q: How do you keep secrets out of logs and results?**
A: API keys are `SecretStr`, so repr and logs show asterisks. They live only in
`Settings`, which is never written to `results/`; the result files only contain
`ExperimentConfig`.
