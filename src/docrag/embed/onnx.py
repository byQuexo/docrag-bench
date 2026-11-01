"""Embedding with ONNX Runtime + HF `tokenizers` (no PyTorch needed).

Why ONNX instead of sentence-transformers/PyTorch?
* Install size: onnxruntime is ~50 MB, torch+CUDA is several GB.
* Same model files run on CPU, CUDA, DirectML or in a browser.
* It's what you'd ship in a production container.
Trade-off: we must do pooling and truncation ourselves (that's this file).
"""

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from docrag.embed.base import Vectors

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnnxFiles:
    """Where a model's files come from, pinned for reproducibility AND security.

    ``revision`` is a git commit of the Hugging Face repo. Without it, a changed
    (or compromised) upstream repo would silently swap the weights we run.
    Two files per model: int8 is fastest on CPU, but CUDA can't accelerate int8
    (dynamically quantized) ops well - on GPU the fp16 export is ~10x faster.
    """

    repo: str
    revision: str
    cpu_file: str
    gpu_file: str

    def file_for(self, providers: list[str]) -> str:
        return (
            self.gpu_file if providers[0] == "CUDAExecutionProvider" else self.cpu_file
        )


@dataclass(frozen=True)
class OnnxModelSpec:
    files: OnnxFiles
    dim: int
    pooling: Literal["cls", "mean"]  # from the model's 1_Pooling/config.json
    max_length: int = 512
    query_prefix: str = ""  # some models (e5, bge) want "query: " prefixes; gte doesn't


# The two models of the "embedding model" axis (768 vs 384 dim).
# int8 = quantized weights: ~4x smaller and ~2-3x faster on CPU, tiny quality loss.
ONNX_MODELS: dict[str, OnnxModelSpec] = {
    "gte-modernbert-base": OnnxModelSpec(
        files=OnnxFiles(
            repo="Alibaba-NLP/gte-modernbert-base",
            revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
            cpu_file="onnx/model_int8.onnx",
            gpu_file="onnx/model_fp16.onnx",
        ),
        dim=768,
        pooling="cls",
    ),
    "gte-small": OnnxModelSpec(
        files=OnnxFiles(
            repo="thenlper/gte-small",
            revision="17e1f347d17fe144873b1201da91788898c639cd",
            # Small enough that fp32 is fast everywhere; no fp16 export exists.
            cpu_file="onnx/model.onnx",
            gpu_file="onnx/model.onnx",
        ),
        dim=384,
        pooling="mean",
    ),
}


def load_session(
    files: OnnxFiles, cache_dir: Path
) -> tuple[ort.InferenceSession, Tokenizer, str]:
    """Download (once) and open an ONNX model + tokenizer. Returns the file used."""
    providers = _providers()
    onnx_file = files.file_for(providers)
    model_path = hf_hub_download(
        files.repo, onnx_file, revision=files.revision, cache_dir=cache_dir
    )
    tok_path = hf_hub_download(
        files.repo, "tokenizer.json", revision=files.revision, cache_dir=cache_dir
    )
    session = ort.InferenceSession(model_path, providers=providers)
    log.info("%s (%s) runs on %s", files.repo, onnx_file, session.get_providers()[0])
    return session, Tokenizer.from_file(tok_path), onnx_file


@functools.cache
def _providers() -> list[str]:
    """CUDA if onnxruntime-gpu + its NVIDIA libraries are usable, else CPU.

    With `uv sync --extra gpu`, CUDA/cuDNN come as pip wheels under
    site-packages/nvidia. preload_dlls() loads them, so no system-wide CUDA
    install (or PATH editing) is needed. Runs once per process (functools.cache).
    """
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return ["CPUExecutionProvider"]
    try:
        ort.preload_dlls()
    except Exception as err:  # libraries missing/broken: fall back, don't crash
        log.warning("CUDA libraries not loadable, using CPU: %s", err)
        return ["CPUExecutionProvider"]
    # CPU stays in the list as a fallback for operators CUDA can't run.
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


class OnnxEmbedder:
    def __init__(self, name: str, cache_dir: Path) -> None:
        if name not in ONNX_MODELS:
            raise ValueError(
                f"Unknown ONNX embedder {name!r}; known: {sorted(ONNX_MODELS)}"
            )
        self.name = name
        self.spec = ONNX_MODELS[name]
        self.dim = self.spec.dim
        self.session, self.tokenizer, onnx_file = load_session(
            self.spec.files, cache_dir
        )
        # The exact file is part of the embedding cache key: int8 and fp16
        # weights produce slightly different vectors, which must not be mixed.
        self.version = f"1:{self.spec.files.revision[:8]}:{onnx_file}"
        self.tokenizer.enable_truncation(max_length=self.spec.max_length)
        self.tokenizer.enable_padding()  # pad to the longest text in the batch
        # BERT wants token_type_ids, ModernBERT doesn't: ask the model what it takes.
        self._input_names = {i.name for i in self.session.get_inputs()}

    def embed(self, texts: list[str]) -> Vectors:
        encodings = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        # Output 0 = last_hidden_state, shape (batch, tokens, hidden).
        hidden = self.session.run(None, feeds)[0]
        if self.spec.pooling == "cls":
            pooled = hidden[:, 0, :]  # the first ([CLS]) token represents the text
        else:
            # Mean over real tokens only: padding must not dilute the average.
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        result: Vectors = pooled.astype(np.float32)
        return result
