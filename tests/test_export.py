import json
from elias_memory.export import export_sft
from elias_memory.types import MemoryRecord

def test_export_sft_creates_jsonl(tmp_path):
    path = tmp_path / "output.jsonl"
    records = [
        MemoryRecord(content="Python is best", type="semantic", importance=0.9),
        MemoryRecord(content="Fixed bug today", type="episodic", importance=0.7),
    ]
    export_sft(records, str(path))
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert "messages" in first
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][1]["role"] == "user"
    assert first["messages"][2]["role"] == "assistant"

def test_export_sft_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    export_sft([], str(path))
    assert path.read_text() == ""
