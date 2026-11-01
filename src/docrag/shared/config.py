"""ExperimentConfig: one YAML file under configs/ = one row in the results table.

Everything that changes a number in the results lives here (and nothing else),
so the config hash in the report identifies exactly what was measured.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from docrag.shared.cache import stable_hash

RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split: Literal["dev", "test", "all"] = "dev"
    # Limits make a full run feasible on a laptop CPU. None = no limit.
    max_docs: int | None = 20
    max_questions: int | None = None


class ExperimentConfig(BaseModel):
    # LEARN: extra="forbid" turns typos into errors: `top_kk: 5` in the YAML fails
    # validation instead of being silently ignored (and you measuring the default).
    model_config = ConfigDict(extra="forbid")

    name: str
    embedder: str = "onnx:gte-small"  # "<backend>:<model>", see embed/registry.py
    prefix_headings: bool = True  # heading chain in embed_text (ablation axis)
    # Text for pages without a text layer: "none", local "rapidocr", or "vlm"
    # (a vision LLM via the proxy; route in ocr_model). Changes the chunks, so it
    # is part of the Qdrant collection name.
    ocr: Literal["none", "rapidocr", "vlm"] = "none"
    ocr_model: str | None = None  # e.g. "openrouter:anthropic/claude-haiku-5.5"
    ocr_temperature: float | None = None  # None = don't send (Claude 5.5 rejects it)
    retrieval: RetrievalMode = "hybrid_rerank"
    # "doc": search only inside the question's document (MMLongBench is single-doc
    # QA). "corpus": search all documents, a much harder variant.
    scope: Literal["doc", "corpus"] = "doc"
    top_k: int = Field(default=10, ge=1)  # chunks after retrieval (and rerank)
    prefetch_k: int = Field(default=50, ge=1)  # per-branch candidates before RRF
    # Qdrant's default RRF k is 2; the original RRF paper uses 60. We make it explicit.
    rrf_k: int = 60
    reranker: str = "onnx:gte-reranker-modernbert-base"
    # Start value only; calibrate on the dev split (docrag calibrate).
    rerank_threshold: float = 0.2
    context_k: int = Field(default=5, ge=1)  # chunks given to the LLM
    llm: str | None = None  # e.g. "ollama:qwen3:8b"; None = retrieval-only run
    judge: str | None = None
    # 0.0 = as deterministic as the model allows. null = don't send the parameter
    # (required for models that reject sampling parameters, e.g. Claude 5.5).
    llm_temperature: float | None = 0.0
    judge_temperature: float | None = 0.0
    # Parallel LLM calls during eval. ~2 for a local Ollama, 4-8 for hosted APIs.
    llm_concurrency: int = Field(default=2, ge=1)
    prompt_version: str = "answer_v1"
    # judge_v1 saw only the short quotes; judge_v2 sees the full cited passages
    # (v1 under-reported faithfulness: an answer that went beyond a 5-word quote
    # looked unsupported). Changing it re-runs the judge calls (new cache keys).
    judge_prompt_version: str = "judge_v2"
    dataset: DatasetConfig = DatasetConfig()

    def config_hash(self) -> str:
        # `name` is just a label; two configs that measure the same thing get the
        # same hash even if named differently.
        # LEARN: model_dump(exclude=...) -> plain dict, minus the given fields.
        return stable_hash(self.model_dump(exclude={"name"}))[:12]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentConfig":
        # LEARN: @classmethod receives the class (`cls`) instead of an instance:
        # an alternative constructor, like a static factory method in Java.
        # yaml.safe_load (never yaml.load) can't construct arbitrary Python objects.
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
