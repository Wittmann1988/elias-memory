from __future__ import annotations
import httpx
import numpy as np
from .base import Embedder

DEFAULT_URL = "https://integrate.api.nvidia.com/v1/embeddings"
DEFAULT_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"

class NvidiaEmbedder(Embedder):
    def __init__(self, api_key: str, dim: int = 384, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL) -> None:
        if not api_key:
            raise ValueError("API key required for NVIDIA NIM embedder")
        self._api_key = api_key
        self._dim = dim
        self._url = url
        self._model = model

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        response = httpx.post(
            self._url,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            json={"input": [text], "model": self._model, "encoding_format": "float", "input_type": "query"},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        vec = np.array(data["data"][0]["embedding"], dtype=np.float32)
        if len(vec) > self._dim:
            vec = vec[:self._dim]
        elif len(vec) < self._dim:
            vec = np.pad(vec, (0, self._dim - len(vec)))
        return vec
