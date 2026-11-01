"""Runtime settings (where things live, secrets) via pydantic-settings.

Two kinds of configuration exist in this project, on purpose:

* ``Settings`` (this file): *environment* - paths, URLs, API keys. Differs per
  machine, never goes into results, may contain secrets.
* ``ExperimentConfig`` (shared/config.py): *what is measured* - model, retrieval
  variant, thresholds. Lives in YAML under configs/, gets hashed into the report.

Keeping them apart means a result row never depends on your local paths and a
secret can never leak into a results file.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# LEARN: Path(__file__) is the path of THIS .py file. .parents[3] walks up:
# shared -> docrag -> src -> repo root. pathlib.Path is the modern replacement for
# os.path string juggling; `/` joins paths: ROOT / "data".
ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # LEARN: pydantic-settings reads each field from the environment. With the
    # prefix, `qdrant_location` is filled from DOCRAG_QDRANT_LOCATION, and a local
    # `.env` file is read too (it's in .gitignore). Types are validated, so
    # DOCRAG_EMBED_BATCH_SIZE=abc fails loudly at startup instead of deep inside.
    model_config = SettingsConfigDict(
        env_prefix="DOCRAG_", env_file=".env", extra="ignore"
    )

    # ":memory:" (tests), a local folder path (no Docker needed) or an http URL.
    # 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 (::1) first,
    # and with Qdrant bound to 127.0.0.1 only (see compose.yaml) every new
    # connection waits ~2 s before falling back to IPv4. Measured: 2.04 s per query.
    qdrant_location: str = "http://127.0.0.1:6333"
    data_dir: Path = ROOT / "data"
    pdf_dir: Path = ROOT / "data" / "mmlongbench" / "documents"
    qa_glob: str = str(ROOT / "data" / "mmlongbench" / "data" / "*.parquet")
    models_dir: Path = ROOT / "models" / "hf"
    cache_path: Path = ROOT / ".cache" / "docrag.sqlite"
    results_dir: Path = ROOT / "results"
    ui_dir: Path = ROOT / "ui" / "dist"  # built web UI (npm run build); optional

    embed_batch_size: int = 32
    # Ollama / LM Studio / vLLM all speak the OpenAI format.
    ollama_base_url: str = "http://localhost:11434/v1"

    # LEARN: SecretStr hides the value in repr()/logs: print(settings) shows
    # '**********'. You must call .get_secret_value() explicitly to use it.
    # AliasChoices: accept OPENAI_API_KEY (what every tool uses) or the prefixed
    # DOCRAG_OPENAI_API_KEY, so you don't need to duplicate keys in your shell.
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "DOCRAG_OPENAI_API_KEY"),
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "DOCRAG_ANTHROPIC_API_KEY"),
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "DOCRAG_GEMINI_API_KEY"),
    )
    groq_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "DOCRAG_GROQ_API_KEY"),
    )
    mistral_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("MISTRAL_API_KEY", "DOCRAG_MISTRAL_API_KEY"),
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENROUTER_API_KEY", "DOCRAG_OPENROUTER_API_KEY"
        ),
    )
    deepseek_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "DOCRAG_DEEPSEEK_API_KEY"),
    )

    # OTLP/HTTP traces endpoint, e.g. Phoenix: http://127.0.0.1:6006/v1/traces
    # None (default) = tracing off. Only serve/ask/eval read it; ingest never
    # exports traces (see shared/tracing.py for why).
    otel_endpoint: str | None = None

    # Hard stop for paid API usage per process (EUR). See llm/proxy.py.
    budget_eur: float = 5.0
    # Look up missing API keys in the OS credential store (set False in CI/tests).
    use_keyring: bool = True

    # LEARN: a model_validator(mode="after") runs once all fields are set, like a
    # post-construct hook. We use it to fill API keys that the environment didn't
    # provide from the OS credential store.
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


# Order of precedence for API keys: environment variable > .env > OS keyring.
_API_KEY_FIELDS = (
    "openai_api_key",
    "anthropic_api_key",
    "gemini_api_key",
    "groq_api_key",
    "mistral_api_key",
    "openrouter_api_key",
    "deepseek_api_key",
)
KEYRING_SERVICE = "docrag"


def _keyring_lookup(name: str) -> str | None:
    """Read a secret from the OS credential store, if one is available.

    `keyring` talks to Windows Credential Manager, the macOS Keychain or the
    Linux Secret Service. Store a key once with:

        uv run keyring set docrag OPENROUTER_API_KEY

    (it prompts without echoing). Secrets then never touch a file in the repo.
    In Docker/CI there is no keyring: the lookup fails quietly and env vars or
    Docker secrets are used instead.
    """
    try:
        import keyring

        return keyring.get_password(KEYRING_SERVICE, name)
    except Exception as err:  # not installed / no backend / locked: not fatal
        logging.getLogger(__name__).debug("keyring unavailable: %s", err)
        return None


# LEARN: @lru_cache on a zero-arg function = a lazy singleton. The first call
# builds Settings (reads env), every later call returns the same object.
# Tests can call get_settings.cache_clear() to force a re-read.
@lru_cache
def get_settings() -> Settings:
    return Settings()
