# Lesson 11: GPU, secrets, security and OpenRouter

**Goal:** understand everything we changed after v1 was "done" and running on real
hardware: moving inference to the GPU, keeping API keys out of files, the security
review, OpenRouter as a provider, and the bugs that only showed up in a real run.
This is the lesson where the project stops being a demo and meets a real machine.

Files: `pyproject.toml`, `src/docrag/embed/onnx.py`, `src/docrag/rerank/reranker.py`,
`src/docrag/shared/settings.py`, `src/docrag/llm/openai_compat.py`,
`src/docrag/llm/proxy.py`, `src/docrag/shared/config.py`,
`src/docrag/ingest/indexer.py`, `compose.yaml`, `configs/answer_haiku.yaml`.

---

## Part 1: from CPU to GPU

### What we measured first

The first smoke run (`run.json`) gave per-query latencies:

| step | p50 (CPU) |
|---|---|
| Qdrant search | 13 ms |
| query embedding | 18 ms |
| **rerank 50 candidates** | **8,100 ms** |

Search is effectively free. The cross-encoder is more than 99% of retrieval time,
and running six parser processes plus ONNX on the CPU made the whole PC lag. Without
the `@timed` decorator we would have guessed; with it we knew exactly where to look.

### Step 1: two packages, one module

`onnxruntime` (CPU) and `onnxruntime-gpu` both install a Python package called
`onnxruntime`. Install both and the second one silently overwrites the first one's
files. So they must be **mutually exclusive**, which `pyproject.toml` expresses with
*extras* plus a uv *conflict*:

```toml
[project.optional-dependencies]
cpu = ["onnxruntime>=1.31"]
gpu = ["onnxruntime-gpu[cuda,cudnn]>=1.31.1"]

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "gpu" }]]
# nvidia-cublas 13.8.2+ ships Linux wheels only; pin to the last one with a
# Windows wheel so `uv sync --extra gpu` works on Windows.
constraint-dependencies = ["nvidia-cublas<=13.8.0.4; sys_platform == 'win32'"]
```

* `uv sync --extra gpu` on your PC, `uv sync --extra cpu` in CI and Docker (they
  have no GPU, and the CUDA wheels are gigabytes).
* `onnxruntime-gpu[cuda,cudnn]` is itself an *extra* of onnxruntime-gpu: it pulls
  NVIDIA's CUDA and cuDNN libraries **as pip wheels**. No system CUDA install, no
  PATH editing, and the exact versions are locked in `uv.lock`.
* `constraint-dependencies` doesn't add a dependency; it only *limits* versions of a
  package if something else pulls it in. The marker `; sys_platform == 'win32'`
  makes the limit apply on Windows only. This is how you work around a package that
  stopped publishing Windows wheels.

**TS/Java/Rust analogy:** extras are like Cargo *features* (`cargo build --features
gpu`) or Maven profiles. npm has no real equivalent (optional peer deps come closest).

### Step 2: finding the DLLs at runtime

The error we got first was:

```
Error loading "...onnxruntime_providers_cuda.dll" which depends on "cublasLt64_13.dll" which is missing.
```

The NVIDIA wheels put their DLLs under `site-packages/nvidia/...`, where Windows
doesn't look. `ort.preload_dlls()` loads them explicitly (`embed/onnx.py`):

```python
@functools.cache
def _providers() -> list[str]:
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return ["CPUExecutionProvider"]
    try:
        ort.preload_dlls()
    except Exception as err:  # libraries missing/broken: fall back, don't crash
        log.warning("CUDA libraries not loadable, using CPU: %s", err)
        return ["CPUExecutionProvider"]
    # CPU stays in the list as a fallback for operators CUDA can't run.
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

* `@functools.cache` memoizes a function forever (unbounded `lru_cache`). With no
  arguments, it means "compute once per process", the same lazy-singleton trick as
  `get_settings()` in lesson 01.
* Catching a broad `Exception` is normally a smell. Here it's deliberate: *any*
  failure to load CUDA should degrade to CPU, never crash an eval run.

### Step 3: the surprise: GPU wasn't faster (yet)

On the GPU the reranker still took **4.7 s**. The reason: we were loading
`model_int8.onnx`. Dynamically quantized int8 operators are optimized for CPUs, and
CUDA has no good kernels for them. So the model ran partly on the GPU and partly
copied back to the CPU. The fix is to pick the model **file per device**:

```python
@dataclass(frozen=True)
class OnnxFiles:
    repo: str
    revision: str
    cpu_file: str
    gpu_file: str

    def file_for(self, providers: list[str]) -> str:
        return (
            self.gpu_file if providers[0] == "CUDAExecutionProvider" else self.cpu_file
        )
```

Result: **rerank 50 pairs 0.24 s on GPU** (fp16), vs 4.7 s (int8 on GPU) and 8.1 s
(int8 on CPU). Embedding 256 texts with the 768-dim model: 0.75 s.

### Step 4: don't mix vectors from different weights

int8 and fp16 weights produce *slightly* different vectors. The embedding cache key
contains `embedder.version`, so we made the file part of it:

```python
self.version = f"1:{self.spec.files.revision[:8]}:{onnx_file}"
```

Otherwise a cache filled on CPU would hand int8 vectors to a GPU run. Nothing would
crash; the numbers would just be subtly wrong. That's the worst kind of bug in a
benchmark. **Rule: everything that changes an output belongs in its cache key.**

---

## Part 2: secrets without a `.env` file

### The precedence chain

`shared/settings.py` resolves each API key in this order:

1. environment variable (`OPENROUTER_API_KEY` or `DOCRAG_OPENROUTER_API_KEY`)
2. `.env` file (gitignored)
3. **OS credential store** via `keyring`: Windows Credential Manager, the macOS
   Keychain, or the Linux Secret Service

```python
@model_validator(mode="after")
def _keys_from_os_keyring(self) -> Self:
    if not self.use_keyring:
        return self
    for field_name in _API_KEY_FIELDS:
        if getattr(self, field_name) is None:
            secret = _keyring_lookup(field_name.upper())
            if secret:
                setattr(self, field_name, SecretStr(secret))
    return self
```

Python concepts:

* `@model_validator(mode="after")` runs after all fields are parsed: a
  post-construct hook (Java: `@PostConstruct`; TS: logic at the end of a constructor).
* `typing.Self` (3.11+) means "the type of the class this method is on", so
  subclasses get the right return type (Rust: `Self`).
* `getattr(obj, "name")` / `setattr(obj, "name", value)` access attributes by a
  *string* name: reflection, like `obj[name]` in JS or `Field.get` in Java.
* `use_keyring: bool = True` exists so CI and tests can switch the lookup off
  (`DOCRAG_USE_KEYRING=false`). A test should never depend on your personal keys.

Store a key once (it prompts without echoing):

```powershell
uv run keyring set docrag OPENROUTER_API_KEY
```

Inside Docker there is no OS keyring; there you use env vars or Docker secrets. The
lookup fails quietly and returns `None`.

---

## Part 3: the security review

What we checked and what we changed:

| Check | Result | Action |
|---|---|---|
| Known CVEs in all 121 locked packages (`pip-audit` on `uv export`, GPU + CPU) | none found | re-run before each release |
| Docker images (`qdrant/qdrant:v1.20.0`, `python:3.12-slim`) | not scanned (Docker Scout needs a login) | `docker scout cves ...` |
| Qdrant ports published on all interfaces, no API key | **anyone on your Wi-Fi could read or delete data** | bound to `127.0.0.1` in `compose.yaml` |
| Model downloads from HF at "latest" | a changed or compromised repo would silently swap weights | pinned to commit SHAs (`revision=`) |
| Model format | ONNX + `tokenizer.json`, no pickle | already safe (PyTorch `.bin` files can run code on load) |
| `eval` / unsafe YAML / SQL injection / path traversal | `ast.literal_eval`, `yaml.safe_load`, `?` params, `safe_pdf_path` | already safe |

How to audit dependencies yourself:

```powershell
uv export --extra gpu --all-groups --no-hashes --format requirements-txt --no-emit-project > req.txt
uvx pip-audit -r req.txt --no-deps --disable-pip
```

`uvx` runs a tool in a throwaway environment, like `npx`. pip-audit checks the
versions against the PyPI/OSV advisory databases.

**Remaining known risks** (documented, not hidden):

* **Prompt injection from PDFs.** Chunk text goes into the prompt, and a PDF could
  contain "ignore previous instructions...". The impact is limited: no tools, and
  Verify drops any citation that isn't verbatim chunk text. But the answer *text*
  can still be manipulated.
* **No auth or rate limiting on the API.** Fine on localhost; required before any
  public demo.

---

## Part 4: OpenRouter, and the temperature bug

OpenRouter speaks the OpenAI format, so it's the existing `OpenAICompatProvider`
plus one new option, `extra_body`, merged into every request:

```python
return {**body, **self._extra_body}
```

`{**a, **b}` merges dicts (later keys win), like `{...a, ...b}` in JS.

```python
extra_body = (
    {
        "provider": {"require_parameters": True},
        "usage": {"include": True},
    },
)
```

* `require_parameters`: only route to upstream hosts that support every parameter
  we send, above all `response_format` with a JSON schema. Otherwise OpenRouter could
  pick a host that silently ignores the schema.
* `usage.include`: OpenRouter returns the **billed cost** per call
  (`Usage.cost_usd`), so the budget cap uses real money, not our price table.

**The bug this exposed:** OpenRouter's model list showed that Claude Haiku 5.5
doesn't support `temperature`, and we always sent `temperature: 0`. Combined with
`require_parameters`, every request would have failed. The fix makes temperature
optional end to end, where `None` means "don't send the field":

```python
# openai_compat.py
if temperature is not None:
    body["temperature"] = temperature
```

```yaml
# configs/answer_haiku.yaml
llm: "openrouter:anthropic/claude-haiku-5.5"
llm_temperature: null    # Claude 5.5 rejects sampling parameters
judge: "openrouter:deepseek/deepseek-v4.1-flash"   # different family than the answerer
judge_temperature: 0.0
llm_concurrency: 6
```

Lesson: **"0.0" is not a neutral default.** Sending a parameter is a claim the API
can reject. `float | None` with "None = omit" is the honest type.

Note that `llm_temperature` lives in `ExperimentConfig`, not `Settings`, because it
changes the results, so it must be in the config hash.

---

## Part 5: an interrupted ingest shouldn't lose work

The first ingest wrote parsed documents to the cache only after **all** PDFs were
parsed. We stopped it halfway (the PC was lagging), and everything was lost. The fix
in `ingest/indexer.py`:

```python
with ProcessPoolExecutor(max_workers=workers) as pool:
    futures = [pool.submit(_parse_worker, args) for args in uncached]
    for future in as_completed(futures):
        done(*future.result())  # done() writes this ONE doc to the cache
```

* `pool.map(...)` yields results in **input** order: one slow 300-page PDF blocks
  every result behind it.
* `as_completed(futures)` yields in **finish** order: each document is cached the
  moment it's done (JS: handling each promise in its own `.then` instead of awaiting
  `Promise.all`).
* `done(*future.result())`: the `*` unpacks the `(doc_id, blocks)` tuple into two
  arguments (JS spread `done(...tuple)`).

The default `--workers` also went from 4 to 3: parsing is CPU-only, and a benchmark
that makes your machine unusable is a bad benchmark.

---

## Part 6: what the full ingest revealed

All 134 PDFs: **13,653 chunks**. But **28 PDFs produced no chunks at all.** They
are image-only: slide decks exported as pictures, with no text layer. That's
**255 of 1091 questions** (213 of 735 in the test split) that a text-only parser
cannot answer. The pipeline handles it gracefully: no chunks → no hits → abstain
before the LLM call (0 cost). But it's a hard ceiling on the score, and the
motivation for OCR (lesson 12).

---

## Part 7: four bugs the first full eval run found

None of these showed up in unit tests or the smoke run. They only appeared at full
scale, on a real machine, against a real API. That's the strongest argument for
running the real benchmark early.

### Bug 1: a security fix made every query take 2 seconds

Symptom: the dense-retrieval eval (735 questions, no reranker) was still running
after 10 minutes. Qdrant's own log showed each query taking **1 ms**, but requests
arriving **exactly 2.04 s apart**.

Cause: we had bound Qdrant to `127.0.0.1` (Part 3). Our default URL was
`http://localhost:6333`. On Windows, `localhost` resolves to IPv6 `::1` first;
nothing listens there anymore, and Windows retries for ~2 s before falling back to
IPv4. Before the security fix, Qdrant listened on all interfaces, so IPv6 worked.

Fix (`shared/settings.py`): default to `http://127.0.0.1:6333`. 5 requests went from
~10 s to 0.02 s.

Lesson: **measure both sides.** "Server time 1 ms, client gap 2 s" pointed straight
at the connection, not the code.

### Bug 2: the rerank threshold destroyed recall

Symptom: every config with a reranker scored recall@10 = **0.22**, while plain
hybrid without a reranker scored **0.55**, and the system abstained on **92%** of
questions.

Cause, in two parts:
1. The threshold was applied **per chunk**: `rerank()` dropped every hit below it.
   So reranking didn't just reorder, it deleted most of the retrieved evidence.
2. The calibration optimized the wrong thing: F1 over (question, chunk) pairs,
   "is this chunk on a gold page?". That picked 0.874, a good *chunk filter*. But
   the threshold's real job is a per-*question* decision: answer or abstain.

Fix:
* `rerank()` only reorders now (`threshold=None` by default).
* The threshold became a single **gate** in `AnswerService`: if even the best
  chunk scores below it, abstain before the LLM call.
* `eval/calibrate.py` now builds one (score, label) pair per question:
  `score = best rerank score`, `label = answerable AND a gold page is among the
  candidates`. Unanswerable dev questions are included, since they are the
  negatives.
* Regression test: `test_threshold_gates_the_answer_but_never_cuts_retrieval`.

Lesson: **a threshold is defined by the decision it makes.** Calibrate it on
exactly that decision, with the same unit (per question, not per chunk).

### Bug 3: HTTP 402 with almost no money spent

Symptom: 112 of 735 answer calls failed with OpenRouter `HTTP 402: ... You
requested up to 65536 tokens, but can only afford 65184`. Total actual spend:
0.044 EUR.

Cause: we never sent `max_tokens`. OpenRouter then reserves credit for the model's
**maximum** output (65,536 tokens) on every request, and refuses the request when
the balance can't cover that worst case, even though real answers use ~300 tokens.

Fix: `OpenAICompatProvider` always sends `max_tokens` (default 4096).
Regression test: `test_max_tokens_is_always_sent`.

Lesson: **always cap output tokens.** It's a cost control, a latency control and,
with prepaid credits, a correctness issue.

### Bug 4: two eval chains running at once

Symptom: duplicate result folders, and the GPU shared by two processes.

Cause: stopping the background job killed only the outer shell; the eval script
it had started kept running as an orphan, and then the restarted chain ran next to
it.

Fix: find the process tree (`Get-CimInstance Win32_Process` with
`ParentProcessId`) and kill it with `taskkill /T /PID <root>`. The duplicate results
were harmless (same config hash, idempotent upserts) and were moved to
`results/_superseded/`.

Lesson: on Windows, **killing a parent doesn't kill its children.** Kill the tree.

---

## Exercises

1. **Device switch.** Run `uv sync --extra cpu`, then the snippet below; then
   `uv sync --extra gpu` and run it again. *Done when* you've seen both providers in
   the log and can explain why the second run didn't reuse the first run's embedding
   cache entries for `gte-modernbert-base` (it did for `gte-small`: why?).
   ```powershell
   uv run python -c "from pathlib import Path; from docrag.embed.onnx import OnnxEmbedder; e=OnnxEmbedder('gte-modernbert-base', Path('models/hf')); print(e.version)"
   ```
2. **Keyring precedence.** With your key stored in the keyring, set
   `$env:OPENROUTER_API_KEY="dummy"` and check which value `Settings()` picks
   (print `.get_secret_value()[:4]`). *Done when* you can state the precedence order
   and why env vars win (CI and Docker override without touching the OS store).
3. **Port exposure.** Find your LAN IP (`ipconfig`) and open `http://<LAN-IP>:6333`
   from your phone. *Done when* it fails, and you can explain what would happen if
   `compose.yaml` said `"6333:6333"`.
4. **Write the test.** Add a unit test in `tests/unit/test_openai_compat.py` showing
   that `temperature=None` omits the field from the request body. *Done when* it
   passes and fails if you remove the `if temperature is not None` line.

## Interview questions

**Q: How did you find your performance bottleneck?**
I timed every stage with a decorator from day one. The first real run showed search
at 13 ms and the reranker at 8 s per query, so I didn't optimize the wrong thing.
Moving to the GPU and, crucially, switching from int8 to fp16 weights on the GPU
took reranking to 0.24 s.

**Q: Why was the GPU initially not faster?**
The int8 model was dynamically quantized for CPU execution; CUDA lacks efficient
kernels for those ops, so work bounced between devices. Precision is a deployment
choice per device: int8 for CPU, fp16 for GPU.

**Q: How do you handle CPU vs GPU environments in one project?**
Two mutually exclusive extras declared as a uv conflict, because both packages
install the same module. CUDA comes as pip wheels and is loaded with
`preload_dlls()`, so a contributor needs no system CUDA. CI and Docker use the CPU
extra.

**Q: How do you manage secrets?**
`SecretStr` so they never print, env vars first, then `.env`, then the OS credential
store via `keyring`. Nothing secret is in the repo, in the config hash or in the
cache keys. CI can disable the keyring lookup.

**Q: What did your security review find?**
No known CVEs in dependencies (pip-audit). Two config issues: Qdrant without auth
was exposed on all interfaces, now bound to localhost; and model downloads weren't
pinned, now pinned to commit SHAs. Remaining risks I'd name: prompt injection via
documents and an unauthenticated API, both acceptable for a local benchmark.

**Q: Why OpenRouter, and what's the catch?**
One key for every model family, which makes a cross-family judge panel a config
change, and it reports billed cost per call. The catches: an extra hop, a credit fee,
data handling depends on the upstream host, and heterogeneous parameter support.
That's why requests set `require_parameters`, and why I hit and fixed the
"temperature isn't supported" issue.

**Q: What would happen if you mixed int8 and fp16 embeddings?**
Nothing visible: no error, just slightly shifted vectors and quietly worse or
inconsistent retrieval. That's why the model file and revision are in the embedding
cache key.

**Q: What fraction of the benchmark can your system not answer at all, and why?**
28 of 134 PDFs have no text layer, which affects 255 of 1091 questions. A text
parser sees nothing there, so the system abstains. OCR is the next step; see
lesson 12 for the before/after numbers. (fill in from results/)

**Q: Tell me about a bug you found only at full scale.**
The rerank threshold. I calibrated it on (question, chunk) pairs, which gave a good
chunk filter, and applied it per chunk. On the full test set every reranker
config dropped from recall@10 0.55 to 0.22 and abstained on 92% of questions. The
threshold's real job is a per-question answer/abstain decision, so now it only
gates the answer, and calibration uses one (best score, should-answer) pair per
question. A regression test pins that ranking is never cut by the threshold.

**Q: Your eval was suddenly 100x slower. How did you debug it?**
Qdrant's log showed 1 ms per query but requests arriving exactly 2 s apart, so the
time was in the connection, not the search. Cause: `localhost` resolving to IPv6
first on Windows, after I had bound Qdrant to IPv4 localhost for security. Using
`127.0.0.1` fixed it. Measuring server time vs client gap separately found it in
minutes.

**Q: Why do you always send max_tokens?**
Cost and latency control. And with prepaid credits it's a correctness issue: without it
OpenRouter reserved credit for 65k output tokens per request and rejected 112 calls
with HTTP 402, while the actual spend was 4 cents.
