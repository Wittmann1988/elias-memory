from __future__ import annotations
import hashlib
import numpy as np
from .base import Embedder

class HashEmbedder(Embedder):
    """Deterministic hash-based pseudo-embeddings.

    Not semantically meaningful, but consistent and lightweight.
    Suitable as fallback when no real embedding model is available.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        # Generate enough hash bytes, interpret as uint8, scale to [-1, 1]
        h = hashlib.sha512(text.encode()).digest()
        needed = self._dim
        data = h
        while len(data) < needed:
            data += hashlib.sha512(data).digest()
        raw = np.frombuffer(data[:needed], dtype=np.uint8).astype(np.float32)
        # Scale from [0, 255] to [-1, 1]
        raw = (raw - 127.5) / 127.5
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw /= norm
        return raw
