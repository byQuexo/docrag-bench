# Lesson 06: Answers and Verify (`src/docrag/answer/`, `src/docrag/prompts/`)

**Goal:** understand how retrieved chunks become a cited answer and how pure code
checks every citation afterwards. This is the project's differentiator: RAG-Anything
returns plain strings; docrag-bench returns evidence IDs with quotes, verified and
measured. Milestone M3: "answer on tiny.pdf with verified citations".

## What happens step by step (`answer/service.py`)

```python
hits = await asyncio.to_thread(self.retriever.retrieve, question, doc_ids)
context_hits = hits[: self.retriever.config.context_k]

if not context_hits:
    # Nothing passed the rerank threshold: abstain without spending an LLM call.
    result = AnswerResult(question=question, answer=NOT_ANSWERABLE, ..., status="abstained",
                          abstained_before_llm=True)
    return result, hits

prompt = load_prompt(self.retriever.config.prompt_version)
messages = [
    Message(role="system", content=f"{prompt}\n{render_context(context_hits)}"),
    *(history or []),
    Message(role="user", content=question),
]
parsed, _ = await self.proxy.structured(self.llm_route, messages, Answer)
return verify(question, parsed, [h.chunk for h in context_hits]), hits
```

### 1. Retrieve (in a thread)

Embedding and reranking are synchronous CPU work. Called directly inside `async def`,
they'd block the event loop and stall every other request (lesson 07).

### 2. Abstain early

With `hybrid_rerank`, an empty list means "nothing in this document scored as relevant".
Answering "Not answerable" then costs no LLM call and is the correct behavior for
MMLongBench's unanswerable questions.

### 3. Render the context with IDs (`answer/context.py`)

```
[tiny:p2:c1] (page 2, table, section: Northwind Survey Report > 2 Results) Table 1: Revenue by region ... |Europe|41.0|45.5| ...
```

Each chunk starts with its citation handle. Page, type and section give the model
orientation; the body is whitespace-flattened.

### 4. Message order for prompt caching

The static instructions (`prompts/answer_v1.txt`) come first, then the context, then the thread
history, then the question. Providers cache the longest identical **prefix** of a
prompt, so stable content goes first. (Here the context changes per question; the
win grows when the same document is asked many times.)

### 5. The structured output contract (`answer/schema.py`)

```python
class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(description="Chunk ID from the context, e.g. doc3:p2:c1")
    quote: str = Field(description="Verbatim text copied from that chunk")


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(
        description="The answer with inline citations like [doc3:p2:c1]"
    )
    evidence: list[Evidence]
```

The `description`s go into the JSON schema the model sees. The prompt (rules 1-5)
requires short verbatim quotes, no foreign IDs, and exactly "Not answerable" + empty
evidence when the context lacks the answer.

### 6. Verify (`answer/verify.py`): pure code, after the LLM

For every evidence item:

- **(a)** `ev.id` must be one of the chunks we actually sent → else `"id not in context"`
  (an invented source).
- **(b)** the quote must be non-empty and appear in that chunk's text after
  canonicalization → else `"quote not found in chunk"` (paraphrased or made-up quote).

```python
_MARKUP = re.compile(r"<br>|[|*_`#]+")


def _canon(text: str) -> str:
    return _SPACES.sub(" ", _MARKUP.sub(" ", normalize(text))).strip()
```

`normalize` (lesson 01) unifies quotes, dashes, ligatures, hyphenated line breaks and
case. `_MARKUP` removes Markdown table syntax, so the quote `Europe 41.0 45.5` matches
`|Europe|41.0|45.5|`.

Then:

- inline `[id]` markers of rejected citations are removed from the answer text
  (`re.sub` with a function as the replacement),
- the status is set:

| status | meaning |
|---|---|
| `abstained` | answer is "Not answerable" |
| `supported` | ≥1 citation and all citations verified |
| `partially_supported` | some citations verified, some rejected (counted per evidence item, not per ID: the same ID with one real and one invented quote is partial) |
| `unsupported` | an answer, but no citation survived. Returned, clearly labelled |

`AnswerResult` keeps **both** `raw_answer` and the cleaned `answer`, and **all**
citations with `verified` + `reason`. The eval can therefore compute "Verify off" and
"Verify on" from the *same* LLM answers, without an extra call.

## Why this design

> - **Code, not another LLM, verifies:** 0 ms, 0 EUR, deterministic, testable.
> - **Remove, don't silently trust:** a citation the user clicks must lead to text that
>   really is there.
> - **"unsupported" is shown, not hidden:** the UI can show a warning; hiding answers
>   would make the system look better than it is.
> - **What Verify can't catch:** a real quote next to a wrong conclusion passes. That
>   needs an entailment/hallucination model (LettuceDetect, planned as the next
>   Verify stage) or the judge's `faithful` flag.

## Python concepts

### `re.sub` with a function

```python
import re

verified = {"a:p1:c0"}
text = "Revenue rose [a:p1:c0] and fell [a:p9:c9]."
re.sub(r"\s?\[([^\]]+)\]", lambda m: m.group(0) if m.group(1) in verified else "", text)
# 'Revenue rose [a:p1:c0] and fell.'
```

Same as `text.replace(regex, fn)` in JS.

### Unpacking into lists

`[system_msg, *(history or []), user_msg]` works like JS spread `[a, ...h, b]`.
`history or []` uses the "falsy → default" idiom (`None` and `[]` are falsy).

### `importlib.resources`

`resources.files("docrag.prompts") / "answer_v1.txt"` reads a file shipped inside the
package. It works from the repo, a wheel or Docker, unlike paths relative to the current
working directory. Roughly Java's `getResourceAsStream`.

## Exercises

1. Build an `Answer` with one good citation, one with a foreign ID and one with a
   paraphrased quote, and run `verify()` against tiny.pdf's chunks. **Done when** you get
   `partially_supported` and both `reason`s.
2. Find a quote that should match but doesn't (hint: a hyphenated line break inside a
   table cell, or "%" vs " percent"). **Done when** you can argue whether to normalize it or
   accept it as a known limitation.
3. Write `answer_v2.txt` that asks for "one sentence + citations", set `prompt_version:
   answer_v2` in a config and run `docrag ask`. **Done when** you've confirmed the cache
   did *not* reuse v1 answers.
4. Make the FakeProvider cite `x:p1:c0` (not in the context). **Done when** the status is
   `unsupported` and the API response has `rejected_citations: 1`.

## Interview questions

**Q: How do you prevent hallucinated citations?**
A: Every chunk in the context has an ID; the model must return evidence as {id, quote}
via structured output. Verify checks in code that the ID was in the context and the quote is
verbatim in that chunk. Failed citations are removed, and if none remain the answer is marked
unsupported.

**Q: What can Verify not detect?**
A: That the quote actually supports the claim, or that the answer is complete. A
correct quote with a wrong conclusion passes. The next stage would be an NLI model like
LettuceDetect, and the judge's "faithful" score measures it in the eval.

**Q: Why not ask the LLM to verify itself?**
A: Cost, latency and determinism. Substring checking is exact, free and testable. LLM
self-checks are useful for *semantic* support, which is a separate, optional stage.

**Q: What happens if the LLM invents a chunk ID?**
A: Check (a) fails: the citation is kept in `citations` with `verified=False, reason="id not
in context"`, its inline marker is removed from the answer, and the status becomes
partially_supported or unsupported. The API only returns verified sources.

**Q: Why put the context in the system message?**
A: Static instructions + context form a stable prefix that providers can cache, and the
thread history plus the question follow as normal turns. It also keeps "rules and sources" separate
from "conversation".

**Q: How do you handle questions the documents can't answer?**
A: Two layers: if nothing passes the rerank threshold we abstain without calling the LLM;
otherwise the prompt instructs "Not answerable" with empty evidence. The eval scores
unanswerable questions as correct only when the system abstains.

**Q: How do you measure whether Verify helps?**
A: Citation precision (share of citations on a gold evidence page) computed on all raw
citations (Verify off) vs only verified ones (Verify on), from the same answers.
(fill in from results/)
