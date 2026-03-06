"""Generate diverse SFT training traces for elias-memory agent.

Uses multiple LLM providers for diversity:
- Groq (llama-3.3-70b-versatile, gemma2-9b-it)
- OpenRouter free models
- Ollama Cloud (nemotron)
- Google Gemini (gemini-2.0-flash)

Output: JSONL in HuggingFace Chat format.
"""

import asyncio
import json
import os
import random
import time
from pathlib import Path

import httpx

# --- Provider configs ---

PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"],
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "models": [
            "google/gemma-3-4b-it:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
        ],
    },
    "ollama": {
        "url": "https://ollama.com/v1/chat/completions",
        "key_env": "OLLAMA_API_KEY",
        "models": ["nemotron-3-nano:30b"],
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key_env": "GEMINI_API_KEY",
        "models": ["gemini-2.0-flash"],
    },
}

SYSTEM_PROMPT = """You are an AI memory agent. You manage a persistent memory system with these operations:
- memory.add(content, type="semantic"|"episodic", importance=0.0-1.0, metadata={})
- memory.recall(query, top_k=5) → returns relevant memories ranked by similarity
- memory.reinforce(memory_id) → strengthens a memory, delays decay
- memory.decay_cycle() → applies exponential decay (7-day half-life) to all memories
- memory.export_sft(path) → exports memories as training data

Memory types:
- semantic: facts, concepts, knowledge (e.g. "Python uses indentation for blocks")
- episodic: events, experiences, outcomes (e.g. "User asked about sorting algorithms at 14:30")

You also track:
- Knowledge gaps: topics with low coverage that need learning
- Skill rates: success/failure ratios per skill domain
- Consolidation: converting episodic memories into semantic lessons

Always explain your reasoning for memory operations. Use concrete examples."""

# --- 20 template categories with variations ---

TEMPLATES = [
    # 1. Simple store
    {"cat": "store_semantic", "prompts": [
        "Store the fact that Python 3.12 introduced type parameter syntax.",
        "Remember that Redis uses single-threaded event loop for commands.",
        "Save this knowledge: SQLite WAL mode allows concurrent reads during writes.",
        "Store: The transformer architecture uses self-attention with O(n²) complexity.",
        "Remember that LoRA reduces trainable parameters by 90%+ while maintaining quality.",
        "Save the fact: NVIDIA Jetson AGX Orin has 2048 CUDA cores and 64GB unified memory.",
        "Store this: FastAPI uses Pydantic for request validation automatically.",
        "Remember: Exponential decay with 7-day half-life means importance halves weekly.",
        "Save: ChromaDB supports hybrid search combining vector similarity and metadata filters.",
        "Store the knowledge that DPO training doesn't need a separate reward model.",
    ]},
    # 2. Simple recall
    {"cat": "recall_query", "prompts": [
        "What do I know about Python type hints?",
        "Recall everything about database performance optimization.",
        "What memories do I have about machine learning training?",
        "Find my notes on API design patterns.",
        "What do I remember about memory management strategies?",
        "Search for anything related to Docker container networking.",
        "Recall my knowledge about embedding models and dimensions.",
        "What do I know about exponential decay in memory systems?",
        "Find memories about error handling best practices.",
        "What have I stored about NVIDIA hardware?",
    ]},
    # 3. Store + recall workflow
    {"cat": "store_recall_workflow", "prompts": [
        "First store that 'httpx supports async HTTP/2', then immediately recall what I know about HTTP libraries.",
        "Save 'pytest fixtures support dependency injection' and then search for testing knowledge.",
        "Store 'git rebase -i allows squashing commits' then recall all git-related memories.",
        "Remember 'asyncio.gather runs coroutines concurrently' then find everything about async Python.",
        "Store 'OpenTelemetry supports auto-instrumentation for FastAPI' then recall observability knowledge.",
    ]},
    # 4. Importance scoring
    {"cat": "importance_scoring", "prompts": [
        "I need to store 'the server crashed at 3am due to OOM'. How important is this? Explain your scoring.",
        "Store 'user prefers dark mode'. What importance score and why?",
        "Remember 'critical security vulnerability found in auth module'. Score the importance.",
        "Save 'team meeting moved to Thursday'. How would you score this memory's importance?",
        "Store 'the cosine similarity threshold for good recall is 0.7'. Rate its importance.",
    ]},
    # 5. Reinforce decisions
    {"cat": "reinforce_decision", "prompts": [
        "I just recalled the memory about 'Python GIL limitations' and it was very useful. Should I reinforce it?",
        "A memory about 'yesterday's weather' was recalled. Should it be reinforced or left to decay?",
        "The memory 'API rate limit is 100 req/min' helped prevent an error. Reinforce?",
        "I recalled 'favorite coffee shop name' during a coding session. Worth reinforcing?",
        "The architectural decision about 'using SQLite for embedded storage' proved correct again. Reinforce it?",
    ]},
    # 6. Decay management
    {"cat": "decay_management", "prompts": [
        "It's been 2 weeks since the last decay cycle. What should I expect to happen to memory importance scores?",
        "Some memories have decayed below 0.05 importance. Should I prune them or keep them?",
        "How should I balance aggressive decay (keeping memory lean) vs conservative decay (preserving rare knowledge)?",
        "After a decay cycle, 30% of episodic memories dropped below 0.1. Is this normal?",
        "A memory about a one-time event from 3 months ago has importance 0.02. Keep or prune?",
    ]},
    # 7. Knowledge gap detection
    {"cat": "knowledge_gaps", "prompts": [
        "I have 50 memories about Python but only 2 about Rust. Identify my knowledge gaps.",
        "Analyze my memory distribution and suggest topics I should learn more about.",
        "I notice I have no memories about security best practices. How should I fill this gap?",
        "My skill rates show 90% success in coding but 40% in testing. What knowledge gaps does this reveal?",
        "After consolidation, I see I have mostly semantic memories and few procedural ones. What's missing?",
    ]},
    # 8. Skill tracking
    {"cat": "skill_tracking", "prompts": [
        "I just completed a Python debugging task successfully. Update my skill tracking.",
        "I failed to write a correct SQL query on the first try. Record this in skill tracking.",
        "My git workflow: 8 successful commits, 2 merge conflicts. Calculate and analyze my skill rate.",
        "Track this: API design task completed in 30 minutes with no errors. How does this affect my rates?",
        "I've been struggling with async Python — 3 successes, 5 failures this week. Analyze the trend.",
    ]},
    # 9. Consolidation
    {"cat": "consolidation", "prompts": [
        "I have 20 episodic memories about debugging sessions. Consolidate them into semantic lessons.",
        "Three separate episodes about API timeout errors. What lesson should I extract?",
        "Consolidate: 'deployed v1.0', 'v1.0 had memory leak', 'fixed in v1.1', 'v1.1 stable for 2 weeks'.",
        "I have 5 episodic memories about failed deployments. What patterns and lessons can be extracted?",
        "Time for nightly consolidation. I have 15 unconsolidated episodes. Walk me through the process.",
    ]},
    # 10. Buffer vs persistent
    {"cat": "buffer_vs_persistent", "prompts": [
        "I'm in a debugging session and need to temporarily remember 5 variable values. Buffer or persistent?",
        "User just told me their name is Erik. Should this go in buffer or persistent memory?",
        "I'm processing a list of 100 items. Where should intermediate results be stored?",
        "A one-time API response with configuration data came in. Buffer or persistent?",
        "I discovered a new design pattern during code review. Buffer or persist? Why?",
    ]},
    # 11. Multi-step memory workflows
    {"cat": "multi_step_workflow", "prompts": [
        "Walk me through a complete workflow: receive new information, decide importance, store, and verify it can be recalled.",
        "Demonstrate: store 3 related facts, recall them together, reinforce the most useful, run decay, check results.",
        "Show a complete memory lifecycle: store → recall → reinforce → decay → consolidate → export.",
        "I need to learn a new topic. Plan: identify gaps, store new facts, verify recall, track skill improvement.",
        "Perform a memory health check: count by type, find lowest importance, identify orphaned memories, suggest cleanup.",
    ]},
    # 12. Cross-referencing
    {"cat": "cross_reference", "prompts": [
        "I have a memory about 'FastAPI uses Starlette' and another about 'Starlette supports WebSocket'. Connect them.",
        "Find memories that relate to both 'Python' and 'performance'. How do they cross-reference?",
        "I stored facts about 'Docker' and 'deployment' separately. Should I link these memories?",
        "A new memory about 'SQLite-vec' relates to existing memories about 'vector search' and 'embeddings'. Map the connections.",
        "Show how memories about 'testing', 'CI/CD', and 'deployment' form a knowledge graph.",
    ]},
    # 13. Error handling
    {"cat": "error_handling", "prompts": [
        "I tried to recall a memory but got 0 results. What should I do?",
        "A store operation failed because importance was -0.5. How should the system handle this?",
        "The decay cycle encountered a corrupted memory record. What's the recovery strategy?",
        "Recall returned 10 results but none seem relevant (all similarity < 0.3). How to handle?",
        "I tried to reinforce a memory_id that doesn't exist. What's the correct error response?",
    ]},
    # 14. Metadata strategies
    {"cat": "metadata_strategy", "prompts": [
        "What metadata should I attach when storing a code snippet about sorting algorithms?",
        "I'm storing a memory about a team decision. What metadata fields are most useful?",
        "Design a metadata schema for episodic memories from debugging sessions.",
        "What metadata helps with knowledge gap detection? Give concrete field examples.",
        "How should I tag memories for efficient cross-domain retrieval?",
    ]},
    # 15. Export and training
    {"cat": "export_training", "prompts": [
        "I have 500 memories. How should I prepare them for SFT training export?",
        "What quality threshold should I use when exporting memories as training data?",
        "Explain how memory recall traces can be converted to SFT instruction-response pairs.",
        "I want to create DPO training pairs from my memory operations. What's the strategy?",
        "After export, how do I validate that training data quality is sufficient?",
    ]},
    # 16. Embedding strategy
    {"cat": "embedding_strategy", "prompts": [
        "Should I use 384-dim or 768-dim embeddings for my memory system? What are the trade-offs?",
        "My hash-based fallback embedder gives worse recall than NVIDIA NIM. When is fallback acceptable?",
        "How does embedding dimension affect memory recall quality vs storage cost?",
        "I'm switching from hash embeddings to NVIDIA NIM 384-dim. How should I handle existing memories?",
        "Compare cosine similarity vs dot product for memory retrieval. Which is better for my use case?",
    ]},
    # 17. Scaling decisions
    {"cat": "scaling", "prompts": [
        "My memory database has grown to 10,000 entries. Should I change my vector search strategy?",
        "When should I switch from numpy brute-force to sqlite-vec or ChromaDB?",
        "Memory recall is getting slow (>100ms for 5000 entries). How to optimize?",
        "I need to shard memories across multiple databases. What's the best partitioning strategy?",
        "How do I handle memory migration when upgrading the embedding model?",
    ]},
    # 18. Context-aware operations
    {"cat": "context_aware", "prompts": [
        "I'm working on a Python project. Recall should prioritize Python-related memories. How to implement?",
        "It's 3am and I'm debugging. How should context (time, activity) affect memory operations?",
        "The user switched from coding to writing docs. How should memory recall adapt?",
        "I have memories from 3 different projects. How to scope recall to the current project?",
        "Context: production incident. Which memories should be auto-surfaced with highest priority?",
    ]},
    # 19. Memory architecture decisions
    {"cat": "architecture", "prompts": [
        "Should episodic and semantic memories use the same vector space or separate indices?",
        "Design a memory consolidation pipeline that runs nightly without blocking real-time operations.",
        "How should I implement memory versioning — when a fact gets updated, keep history or overwrite?",
        "Compare WAL mode vs journal mode for a memory database that does 80% reads, 20% writes.",
        "Should the decay function be time-based only, or also consider access patterns?",
    ]},
    # 20. Real-world scenarios
    {"cat": "real_world", "prompts": [
        "I'm building an AI coding assistant. Design its memory system for storing code patterns, user preferences, and project context.",
        "An autonomous agent needs to remember tool usage outcomes across sessions. Design the memory schema.",
        "Design a memory system for a research agent that reads papers and must track claims, citations, and contradictions.",
        "How should a customer support AI manage memory about customer history, product knowledge, and resolution patterns?",
        "Design memory architecture for an AI tutor that tracks student progress, knowledge gaps, and teaching strategies.",
    ]},
]


async def call_llm(provider: str, model: str, user_prompt: str, temperature: float = 0.7) -> str | None:
    """Call an LLM via OpenAI-compatible API."""
    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["key_env"], "")
    if not api_key:
        return None

    headers = {"Content-Type": "application/json"}
    url = cfg["url"]
    if provider == "gemini":
        # Gemini uses API key as query param
        url = f"{cfg['url']}?key={api_key}"
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 1024,
        "temperature": temperature,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code != 200:
                print(f"  [{provider}/{model}] HTTP {resp.status_code}: {resp.text[:100]}")
                return None
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if len(content) < 50:
                return None
            return content
    except Exception as e:
        print(f"  [{provider}/{model}] Error: {e}")
        return None


async def generate_traces(output_path: str = "data/memory-traces-v2.jsonl", target: int = 200):
    """Generate diverse memory operation training traces."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build (provider, model) pairs
    model_pool = []
    for prov, cfg in PROVIDERS.items():
        if os.environ.get(cfg["key_env"]):
            for m in cfg["models"]:
                model_pool.append((prov, m))

    if not model_pool:
        print("ERROR: No API keys found. Set GROQ_API_KEY, OPENROUTER_API_KEY, OLLAMA_API_KEY, or GEMINI_API_KEY.")
        return

    print(f"Models available: {len(model_pool)}")
    for p, m in model_pool:
        print(f"  {p}/{m}")

    # Flatten all prompts
    all_prompts = []
    for tmpl in TEMPLATES:
        for prompt in tmpl["prompts"]:
            all_prompts.append({"category": tmpl["cat"], "prompt": prompt})

    random.shuffle(all_prompts)
    print(f"\nTotal prompts: {len(all_prompts)}, target traces: {target}")

    traces = []
    errors = 0
    sem = asyncio.Semaphore(3)  # max 3 concurrent API calls (rate limit friendly)

    async def gen_one(item: dict) -> dict | None:
        nonlocal errors
        async with sem:
            prov, model = random.choice(model_pool)
            temp = round(random.uniform(0.6, 0.9), 2)
            response = await call_llm(prov, model, item["prompt"], temp)
            if response is None:
                errors += 1
                return None
            return {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": response},
                ],
                "category": item["category"],
                "model": f"{prov}/{model}",
                "temperature": temp,
            }

    # Process in batches of 20
    for i in range(0, min(len(all_prompts), target), 20):
        batch = all_prompts[i:i+20]
        results = await asyncio.gather(*[gen_one(item) for item in batch])
        for r in results:
            if r is not None:
                traces.append(r)

        print(f"  Progress: {len(traces)} traces, {errors} errors, batch {i//20 + 1}")

        # Rate limit pause between batches
        await asyncio.sleep(3)

        # Save intermediate
        if len(traces) % 50 == 0 and traces:
            with open(out, "w") as f:
                for t in traces:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Final save
    with open(out, "w") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(traces)} traces saved to {out}")
    print(f"Errors: {errors}")
    print(f"Categories: {len(set(t['category'] for t in traces))}")
    print(f"Models used: {len(set(t['model'] for t in traces))}")
    return out


if __name__ == "__main__":
    asyncio.run(generate_traces())
