"""make_embedder("onnx:gte-small") -> Embedder. The prefix picks the backend."""

from pathlib import Path

from docrag.embed.base import Embedder


def make_embedder(spec: str, models_dir: Path) -> Embedder:
    backend, _, model = spec.partition(":")
    # LEARN: `match` (3.10+) is structural pattern matching, like Rust's match or a
    # Java 21 switch. Here it's used as a readable switch on strings; it can also
    # destructure tuples, dataclasses and dicts.
    match backend:
        case "onnx":
            # Imported lazily: tests using "fake:" never load onnxruntime.
            from docrag.embed.onnx import OnnxEmbedder

            return OnnxEmbedder(model, models_dir)
        case "fake":
            from docrag.embed.fake import FakeEmbedder

            return FakeEmbedder(int(model) if model else 64)
        case _:
            raise ValueError(
                f"Unknown embedder backend in {spec!r} (use onnx: or fake:)"
            )
