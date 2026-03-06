# Detail-Memory (ausgelagert aus MEMORY.md)

## SelfEvolvingFramework (~/repos/SelfEvolvingFramework/)
- GitHub: Wittmann1988/SelfEvolvingFramework, 6 Commits auf main
- **Implementiert**: Memory Store, Trace Logger, Benchmark Suite (Python)
- CLI: `sef` Alias (python3 cli.py) — memory/traces/bench Befehle
- Memory Store: SQLite + numpy Vektoren, NVIDIA Embeddings (2048-dim), 30 Memories
- Trace Logger: 30 Research-Traces geloggt, 24 SFT-eligible
- Benchmark: Baseline 80/100, 17 Tests in 6 Bereichen
- Research: 5 Runs komplett, SYNTHESIS.md pro Run
- Script: scripts/research-run.sh v2 (18+ Modelle, 7 Provider)
- Naechste Phase: Erstes SFT Training auf HF Jobs mit gesammelten Traces

## Installed Termux Packages
- android-tools, nmap, termux-api, termux-shizuku-tools, termux-gui-package, sshpass

## Multi-Modell Pipeline (~/repos/EliasMemory/impl-pipeline.sh)
- 18 Modelle + 1 Orchestrator (Claude Opus) = 19 AI-Systeme
- Parallele Ausfuehrung via bash background jobs (&)
- Providers: Ollama Cloud, OpenAI, xAI, Google, Groq, OpenRouter, NVIDIA NIM
- Vollstaendiges Inventar: ~/repos/EliasMemory/modell-inventar.md
- Impl-Runs: impl-run1/, impl-run2/, impl-run3/ (Code-Fokus, nicht Konzepte)

## Redundanz-Analyse (2026-03-05)
- Memory-System x3: AgentForge/memory/, MemoryFramework/, SelfEvolvingFramework/
- CLAUDE.md x2: ~/CLAUDE.md + ClaudeEnhancements/config/
- Modell-Config x3: EliasMemory, AgentForge, ollama-sidekick
- Task-Tracking x2: Tasks-Repo + AgentForge/projects.json
- AgenticFramework + DeepResearch: beides Konzept-Docs -> zusammenfuehren
- Standalone (kein Overlap): SystemManager, AsusWRT-Merlin, HackAI, HackerFramework
