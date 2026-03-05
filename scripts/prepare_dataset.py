#!/usr/bin/env python3
"""Prepare SFT training dataset from existing traces.

Reads traces from SelfEvolvingFramework, filters by quality,
and outputs HuggingFace Chat-format JSONL.
"""
import json
import sys
from pathlib import Path

TRACES_PATH = Path.home() / "repos/SelfEvolvingFramework/data/traces/traces-2026-03-05.jsonl"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "sft_dataset.jsonl"
MIN_QUALITY = 0.7


def main():
    if not TRACES_PATH.exists():
        print(f"Traces not found at {TRACES_PATH}")
        sys.exit(1)

    traces = []
    with open(TRACES_PATH) as f:
        for line in f:
            trace = json.loads(line)
            if trace.get("quality_score", 0) >= MIN_QUALITY:
                traces.append(trace)

    print(f"Found {len(traces)} traces with quality >= {MIN_QUALITY}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for trace in traces:
            # Convert to HF Chat format
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI memory and task agent. Execute tasks efficiently, manage memory, and learn from interactions.",
                },
            ]
            for msg in trace.get("messages", []):
                messages.append({"role": msg["role"], "content": msg["content"]})

            entry = {
                "messages": messages,
                "metadata": {
                    "trace_id": trace.get("trace_id"),
                    "model": trace.get("model"),
                    "task_type": trace.get("task_type"),
                    "quality_score": trace.get("quality_score"),
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Written {len(traces)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
