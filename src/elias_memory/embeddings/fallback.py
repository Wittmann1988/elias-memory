from __future__ import annotations
import hashlib
import numpy as np
from .base import Embedder

class HashEmbedder(Embedder):
    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        h = hashlib.sha512(text.encode()).digest()
        needed = self._dim * 4
        data = h
        while len(data) < needed:
            data += hashlib.sha512(data).digest()
        raw = np.frombuffer(data[:needed], dtype=np.float32).copy()
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw /= norm
        return raw
