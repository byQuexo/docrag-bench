# Lesson 00: Setup and tooling

**Goal:** understand how a modern Python project is laid out and built, and what
each tool in `pyproject.toml` does. This is milestone M0 of the plan: "`uv run
pytest` and CI green".

## What happens step by step

### 1. `pyproject.toml` is the single project file

```toml
[project]
name = "docrag-bench"
requires-python = ">=3.12,<3.13"
dependencies = [ "pydantic>=2.14.2", "qdrant-client>=1.20", ... ]

[project.scripts]
docrag = "docrag.cli:main"

[build-system]
requires = ["uv_build>=0.9,<0.14"]
build-backend = "uv_build"

[dependency-groups]
dev = ["mypy>=2.4.1", "pytest>=9.2.0", "ruff>=0.17.3", ...]
```

It plays the role of `package.json` + `tsconfig.json` + `.eslintrc` together, or
`pom.xml`, or `Cargo.toml`:

| pyproject section | TS/Node equivalent | Java | Rust |
|---|---|---|---|
| `[project] dependencies` | `dependencies` | `<dependencies>` | `[dependencies]` |
| `[dependency-groups] dev` | `devDependencies` | `<scope>test</scope>` | `[dev-dependencies]` |
| `[project.scripts]` | `"bin"` | main class in manifest | `[[bin]]` |
| `[tool.ruff]`, `[tool.mypy]` | `.eslintrc`, `tsconfig.json` | checkstyle / javac flags | clippy.toml |
| `uv.lock` | `package-lock.json` | (none by default) | `Cargo.lock` |

`[project.scripts] docrag = "docrag.cli:main"` means: after install, a `docrag`
executable exists in the venv that calls `main()` in `src/docrag/cli.py`. That is
why `uv run docrag eval ...` works.

### 2. uv manages Python, the venv and the lockfile

- `uv sync` creates `.venv/` and installs exactly what `uv.lock` says.
- `uv add httpx` adds a dependency to pyproject *and* the lockfile.
- `uv run <cmd>` runs a command inside the venv without activating it.

**venv vs node_modules:** a Python virtual environment is a folder with its own
`python.exe` plus its own `site-packages/` (installed libraries). Unlike
`node_modules`, it's *one flat environment per project*: Python has no nested
dependency trees, so two libraries needing conflicting versions of a third can't
coexist. That's why the resolver (uv) matters and why the lockfile is committed.

### 3. The `src/` layout

```
src/docrag/__init__.py      <- makes "docrag" a package
src/docrag/shared/__init__.py
src/docrag/shared/ids.py    <- import as: from docrag.shared.ids import make_chunk_id
```

With `src/`, your code is only importable *after it was installed* into the venv
(uv does an editable install). Tests therefore exercise the installed package,
exactly as a user would. Without `src/`, Python would import from the current
directory, which hides packaging bugs (e.g. a missing prompt file).

`src/docrag/py.typed` is an empty marker that tells mypy "this package ships type
hints".

### 4. Imports, modules, packages

- A **module** is one `.py` file. A **package** is a folder with `__init__.py`.
- `from docrag.llm.proxy import LLMProxy` runs `docrag/__init__.py`,
  `docrag/llm/__init__.py`, then `docrag/llm/proxy.py` (once; cached in
  `sys.modules`).
- Importing a module **executes** it. That's why `ingest/parser.py` ends with
  `register("pymupdf", PyMuPdfParser)`: importing the module registers the parser.
- Circular imports fail at import time, so the dependency direction is
  `shared <- ingest/embed/store/rerank/llm <- answer <- api/eval`. `shared` imports
  nothing from the other packages.
- Some imports are inside functions on purpose (lazy imports), e.g. in
  `embed/registry.py`:

```python
case "onnx":
    # Imported lazily: tests using "fake:" never load onnxruntime.
    from docrag.embed.onnx import OnnxEmbedder
```

### 5. `python -m` and `if __name__ == "__main__"`

`python -m docrag.cli eval x.yaml` runs the module `docrag.cli` as a script. Inside
it, `__name__` is `"__main__"`, so the guard at the end of `cli.py` calls
`main()`. When the module is *imported* instead, `__name__` is `"docrag.cli"` and
nothing runs. On Windows, the guard is also required for multiprocessing (lesson 02).

### 6. The quality tools

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

- **ruff** = linter + formatter (replaces flake8, isort, black, pyupgrade), written
  in Rust and very fast. `E`/`F` = errors/pyflakes (unused imports, undefined
  names), `I` = import sorting, `UP` = "use modern syntax" (e.g. `list[int]`
  instead of `List[int]`), `B` = bugbear (likely bugs, e.g. mutable defaults,
  `zip()` without `strict=`).
- **mypy strict**: static type checking. Python ignores type hints at runtime;
  mypy checks them before you run. `strict = true` means every function must be
  annotated and `Any` leaks are errors. `plugins = ["pydantic.mypy"]` teaches mypy
  how Pydantic generates `__init__`.
- **pytest**: test runner. Any `tests/**/test_*.py` with `def test_*()` functions is
  a test; `assert` is the assertion API. `asyncio_mode = "auto"` (pytest-asyncio)
  lets you write `async def test_...`.

## Why this design

> - **uv instead of pip + venv + pip-tools:** one tool, a real lockfile, 10-100x
>   faster installs. Also installs Python itself (`.python-version` pins 3.12).
> - **Python 3.12 only:** PEP 695 generics (`class GatherResult[T]`), better
>   error messages, and every dependency ships 3.12 wheels on Windows. Pinning
>   one minor version keeps CI and local runs identical.
> - **mypy strict from day 1:** retrofitting types later is painful; with
>   Pydantic at the boundaries and mypy inside, most bugs are found before running.

## Python concepts

**Type hints are not enforced at runtime:**

```python
def add(a: int, b: int) -> int:
    return a + b


add("x", "y")  # runs fine and returns "xy"; only mypy complains
```

TS works the same way (types are erased), Java and Rust don't (the compiler
rejects it). Pydantic is how we get *runtime* checks where data enters the system.

**Modern syntax (enforced by ruff `UP`):**

```python
x: list[int] | None = None  # not Optional[List[int]]


def f[T](items: list[T]) -> T: ...  # PEP 695 generics, like <T> in TS/Java
```

## Exercises

1. Run `uv run docrag --help`, then find in `cli.py` where each subcommand is
   registered. **Done when** you can explain what `p.set_defaults(func=...)` does.
2. Add `"SIM"` (flake8-simplify) to ruff's `select` list and run `uv run ruff check
   .`. **Done when** you've read each new warning and decided whether it's useful.
   Revert afterwards.
3. Remove the return type of `make_chunk_id` in `shared/ids.py` and run
   `uv run mypy src`. **Done when** you've seen the strict-mode error and know which
   flag causes it (`disallow_untyped_defs`).
4. Run `uv run python -c "import docrag.ingest.parser as p; print(p._REGISTRY)"`.
   **Done when** you can explain why the registry isn't empty although you never
   called `register()`.

## Interview questions

**Q: Why uv and not Poetry/pip?**
A: uv does venv creation, dependency resolution, the lockfile and Python version
management in one tool, and it's much faster, which matters in CI. Poetry would
also work; plain pip has no lockfile, so builds aren't reproducible.

**Q: Why a src layout?**
A: With `src/` the tests import the *installed* package, so a missing file in the
package (e.g. a prompt template) fails in tests instead of in Docker. It also
prevents accidentally importing from the working directory.

**Q: Python doesn't enforce types. Why bother with mypy strict?**
A: Types document the contracts between the stages (Block, Chunk, Hit) and mypy
finds mismatches without running a 30-minute ingest. At runtime, Pydantic
validates data at the system boundaries: PDFs, YAML configs, LLM output, HTTP requests.

**Q: How does `uv run docrag` find your code?**
A: `[project.scripts]` declares the console script `docrag = "docrag.cli:main"`.
`uv sync` installs the project in editable mode and creates the executable in
`.venv/Scripts/`, which calls `main()`.

**Q: What's in the lockfile and why commit it?**
A: The exact version and hash of every direct and transitive dependency. Committing
it makes CI, Docker and every developer machine install identical packages.
