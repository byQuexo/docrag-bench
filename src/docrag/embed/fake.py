"""A deterministic, dependency-free embedder for tests and CI.

"Hashing trick" bag of words: every token adds 1 to dimension hash(token) % dim.
Texts sharing words get similar vectors, so retrieval tests are meaningful, but
no model download is needed and results are identical on every machine.
"""

import zlib

import numpy as np

from docrag.embed.base import Vectors
from docrag.shared.text import tokenize


class FakeEmbedder:
    name = "fake"
    version = "1"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> Vectors:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                # crc32 instead of hash(): Python's hash() of str is randomized per
                # process (PYTHONHASHSEED), which would make tests flaky.
                out[row, zlib.crc32(token.encode()) % self.dim] += 1.0
        out[:, 0] += 1e-3  # avoid all-zero vectors for empty/stopword-only text
        return out
