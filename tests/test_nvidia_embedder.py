import numpy as np
from unittest.mock import patch, MagicMock
from elias_memory.embeddings.nvidia import NvidiaEmbedder

def test_nvidia_embedder_dim():
    emb = NvidiaEmbedder(api_key="fake", dim=384)
    assert emb.dim == 384

def test_nvidia_embedder_no_key_raises():
    import pytest
    with pytest.raises(ValueError, match="API key"):
        NvidiaEmbedder(api_key="", dim=384)

@patch("elias_memory.embeddings.nvidia.httpx")
def test_nvidia_embedder_calls_api(mock_httpx):
    fake_vec = np.random.randn(384).astype(np.float32).tolist()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"embedding": fake_vec}]}
    mock_httpx.post.return_value = mock_response

    emb = NvidiaEmbedder(api_key="test-key", dim=384)
    result = emb.embed("hello")
    assert result.shape == (384,)
    assert result.dtype == np.float32
    mock_httpx.post.assert_called_once()
