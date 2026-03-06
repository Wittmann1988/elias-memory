from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class VectorIndex(ABC):
    @abstractmethod
    def add(self, id: str, vec: np.ndarray) -> None: ...
    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]: ...
    @abstractmethod
    def delete(self, id: str) -> None: ...

class NumpyVectorIndex(VectorIndex):
    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._ids: list[str] = []
        self._vecs: list[np.ndarray] = []

    def add(self, id: str, vec: np.ndarray) -> None:
        norm = np.linalg.norm(vec)
        normalized = vec / norm if norm > 0 else vec
        self._ids.append(id)
        self._vecs.append(normalized)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if not self._vecs:
            return []
        norm = np.linalg.norm(query_vec)
        q = query_vec / norm if norm > 0 else query_vec
        matrix = np.stack(self._vecs)
        similarities = matrix.astype(np.float32) @ q.astype(np.float32)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self._ids[i], float(similarities[i])) for i in top_indices]

    def delete(self, id: str) -> None:
        try:
            idx = self._ids.index(id)
            self._ids.pop(idx)
            self._vecs.pop(idx)
        except ValueError:
            pass
