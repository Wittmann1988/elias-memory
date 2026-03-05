from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...
