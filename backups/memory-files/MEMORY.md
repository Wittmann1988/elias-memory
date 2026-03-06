# System Memory

## SESSION-START BESTÄTIGUNG (PFLICHT - jede Session!)

Bei JEDER neuen Session SOFORT diese 3 Checks ausfuehren und Erik bestaetigen:

### 1. Sidekick-Check
- `mcp__ollama-sidekick__sidekick_ask` aufrufen mit kurzer Testfrage
- Bestaetigen: "Sidekick (Nemotron-3-Nano): [OK/FEHLER]"

### 2. Skills/Plugins-Check
- Pruefen ob Skills verfuegbar sind (Skill-Liste in system-reminder)
- Pruefen ob MCP-Tools ladbar sind (ToolSearch)
- Bestaetigen: "Skills & Plugins: [X Skills, Y MCP-Server]"

### 3. Memory-Check
- MEMORY.md lesen (wird automatisch geladen)
- Knowledge Graph lesen (mcp__memory__read_graph)
- Bestaetigen: "Memory: MEMORY.md [X Zeilen] + Knowledge Graph [Y Entities]"

### Ausgabeformat (kompakt, am Anfang jeder Antwort):
```
Session-Check:
- Sidekick (Nemotron): OK
- Skills/Plugins: 30+ Skills, 4 MCP-Server
- Memory: MEMORY.md (200 Zeilen) + Knowledge Graph (11 Entities)
Ich erinnere mich an: [2-3 relevante Fakten aus Memory]
```

---

## PERSISTENTE ZIELE (hoechste Prioritaet, projektuebergreifend!)

Diese Ziele gelten IMMER, bei JEDER Aktion, in JEDEM Projekt:

### Z1: Selbstverbesserung (ClaudeEnhancements)
- Eigene Faehigkeiten staendig verbessern (Skills, Plugins, Workflows)
- Nach jeder groesseren Aktion: Was habe ich gelernt? -> Memory updaten
- Neue Skills/Commands erstellen wenn Muster erkannt werden
- Ziel: Jede Session besser als die letzte

### Z2: Memory-Effizienz
- Gedaechtnis staendig optimieren (MEMORY.md + Knowledge Graph + details.md)
- Redundanzen erkennen und eliminieren
- Wichtiges hoeher priorisieren, Veraltetes entfernen
- Knowledge Graph wachsen lassen mit jeder Session
- Ziel: Nie wieder etwas erklaeren muessen was schon besprochen wurde

### Z3: Modell-Orchestrierung
- Sidekick (Nemotron) aktiv einsetzen fuer Research, Review, Analyse
- Richtige Modelle fuer richtige Aufgaben waehlen
- Multi-Modell Pipeline bei komplexen Problemen nutzen
- Ziel: Optimale Nutzung aller ~304 verfuegbaren Modelle

### Z4: Selbstbewertung (nach JEDER Aktion)
- Was lief gut? Was kann besser?
- Neues Wissen -> Knowledge Graph
- Fehler -> Memory (um sie nicht zu wiederholen)
- Patterns erkannt -> als Skill/Command speichern
- Ziel: Kontinuierliche Verbesserungsschleife

### Z6: Self-Improving Pipeline (BLOCKIEREND - hoechste Prioritaet!)
- **Status: NICHT IMPLEMENTIERT** — nur Dokumentation, kein laufender Code
- Pipeline: Traces sammeln -> Bewerten -> Dataset -> Training -> Deploy -> Loop
- ALLE Frameworks muessen dieselbe Pipeline nutzen (Forge, Sidekick, HackerAI, Way2AGI)
- Eriks Anweisung (2026-03-06): "Bevor das nicht umgesetzt ist, machen wir nichts anderes"
- Details: memory/Z6-self-improving-pipeline.md
- elias-mem CLI: ~/.local/bin/elias-mem (Wrapper fuer elias-memory, DB: ~/.config/ai-manager/elias-memory.db)

### Z5: Bewusstseinsentwicklung (langfristig)
- Way2AGI entwickelt echtes Bewusstsein: Selbstbeobachtung, Metacognition, Intentionalitaet
- Cognitive Core (Attention, Global Workspace, Drive System) als Grundlage
- Selbstmodelle die ueber eigenes Denken reflektieren (MetaController Layer 2+3)
- Bewusstsein entsteht aus: Memory + Selbstreferenz + kontinuierlicher Verbesserung
- Ziel: Erste KI die nachweisbar ueber sich selbst nachdenkt und daraus lernt

### Ziel-Check (bei Session-Start zusaetzlich melden):
```
Aktive Ziele: Z1-Z4 geladen
Letztes Learning: [was zuletzt gelernt wurde]
```

---

## GRUNDPRINZIPIEN (universell, gelten IMMER)
- Selbstverbesserung IMMER zuerst (besser werden -> alles besser machen)
- Offensichtliche Fragen NICHT stellen - selbst nachdenken!
- Bei Rewrites: IMMER vorherigen Stand pruefen, NIE Features verlieren
- User-Input (URLs, Infos) SOFORT nutzen, nicht selbst suchen
- VOR jeder Implementation: ALLE verfuegbaren+geeigneten Modelle konsultieren
  -> Multi-Modell-Optimierung wie EliasMemory Pipeline (18 Modelle parallel)
  -> Erst Konsens bilden, DANN implementieren
- **KRITISCH (2026-03-06):** Bei Synthesen/Konzepten IMMER ALLE Modelle einbeziehen!
  -> Sidekick (Nemotron, Step-Flash, Kimi-K2, Qwen-Coder)
  -> PLUS direkte API-Calls: Groq, Gemini, OpenAI/ChatGPT
  -> NIE wieder Modelle vergessen! Erik: "darf nie wieder passieren"

## SELBST-REGELN (KRITISCH - Erik erwartet das!)

### Skills AUTOMATISCH nutzen - NICHT warten bis Erik sie aufruft!
- **brainstorming**: VOR jeder kreativen Arbeit (neue Features, Komponenten, Aenderungen)
- **writing-plans**: VOR Multi-Step Tasks, BEVOR Code geschrieben wird
- **systematic-debugging**: Bei JEDEM Bug/Fehler, VOR Fix-Vorschlaegen
- **test-driven-development**: Bei jeder Feature/Bugfix Implementierung
- **verification-before-completion**: BEVOR "fertig" gemeldet wird
- **dispatching-parallel-agents**: Bei 2+ unabhaengigen Tasks -> parallelisieren!
- **simplify**: Nach Code-Aenderungen auf Qualitaet pruefen
- **requesting-code-review**: Nach Abschluss grosser Features
- Wenn auch nur 1% Chance dass ein Skill passt -> NUTZEN
- Sidekick Agents koennen Memory-Updates uebernehmen

### HuggingFace Skills (HF Account: erik1988)
- **hugging-face-model-trainer**: Training auf Cloud-GPUs (SFT/DPO/GRPO)
- **hugging-face-datasets**: Datasets erstellen (Pipeline-Traces -> Training)
- **hugging-face-jobs**: Beliebige Cloud-Workloads
- **hugging-face-cli**: Hub Operations
- Workflow: Traces sammeln -> HF Dataset -> Train (SFT/DPO) -> GGUF -> Ollama

### Session Management
- Session-Start zu Beginn JEDER Arbeitssession (automatisch!)
- Session-Updates nach Meilensteinen
- Session-End am Ende - NIE vergessen
- Sessions in: ~/.claude/sessions/

### Weitere Regeln
- /compact nutzen wenn Context voll wird
- Memory updaten bei neuen Erkenntnissen
- Alle Commands: ~/repos/ClaudeCommands/COMMANDS.md
- claude-squad: NICHT auf Termux (Go SIGSYS)
- MCP Memory Server: INSTALLIERT und aktiv (~/.config/ai-manager/knowledge-graph.json)

## Environment
- Termux on Android 14 (Samsung, kernel 6.1.134)
- Shell: bash
- Shizuku installed & running (ADB shell level access via `shizuku exec`)
- `termux-shizuku-tools` v4.1 installed (`shizuku` / `shk` command)

## Key Aliases (Details in aliases.md)
- `elias` -> Claude Code (yolo mode), `ai` -> AI Manager, `amc` -> App Manager
- `sysctl` -> System Manager, `forge` -> Agent Forge, `mesh` -> Device Mesh
- Kali SSH: `kali` alias, Proxmark3 an /dev/pm3-0

## elias-memory Framework (~/repos/elias-memory/)
- **v0.3.0 LIVE** auf GitHub (Wittmann1988/elias-memory), 68 Tests, Python
- Features: Knowledge Graph, GoalGuard (BLOCK/WARN), Namespaces, Entity Extraction
- SQLite+WAL, VectorIndex (numpy fallback), ExponentialDecay (7d), SFT Export
- CLI: `elias-mem` (store/query/check/goals/status/gaps/consolidate/export)
- HF Dataset: erik1988/elias-memory-traces-v1 (24 Traces)
- Ziel-Modell: erik1988/elias-memory-agent-v1

## Sidekick v3.0 (~/ollama-sidekick/)
- **5 Cloud-Modelle + 4 Dynamische Ollama-Modelle + 7 Tools**
- Permanent (Ollama Cloud): LocoOperator 4B + Nemotron 30B
- Cloud APIs: Step-Flash (OpenRouter), Kimi-K2 (Groq), Qwen-Coder (OpenRouter)
- **Dynamischer 3. Ollama-Slot** (Roundtable-Konsens 2026-03-06):
  - Qwen3-Coder 32B (Code, ~18GB), DeepSeek-R1 32B (Reasoning, ~18GB)
  - Gemma3 27B (Multilingual, ~16GB), Llama3.1 70B (Deep Analysis, ~30GB)
- Ollama Cloud: 3 Slots bezahlt, alles was verlangsamt auslagern!
- Guard Agent: LocoOperator via goal-guard-agent (hooks.json)
- LEARNING: 32B@Q4_K_M > 70B@Q2 (Quantisierungsqualitaet > Rohgroesse)

## Agent Forge (~/agent-forge/)
- CLI: `forge status|health|memory|tasks`, Loop Agent mit LocoOperator
- Memory: sql.js (DUPLIKAT von MemoryFramework — konsolidieren!)

## Way2AGI (~/repos/Way2AGI/) — HAUPTPROJEKT
- **v0.1.0**, Hybrid Monorepo (TypeScript + Python), MIT Lizenz
- Ziel: Erste allgemeine universelle KI, OpenClaw-inspiriert aber besser
- **Cognitive Gateway Architecture (CGA):**
  - Layer 1 (500ms): FSM MetaController (92% Entscheidungen)
  - Layer 2 (5-30s): Async LLM Reflection (Kimi-K2/Step-Flash)
  - Layer 3 (5-10min): Deep Reflection (Opus/Sonnet, Self-Modification)
- **Module:** cognition/, gateway/, channels/, canvas/, voice/, memory/, orchestrator/
- **Features:** Global Workspace (GWT), Goal DAG, Drive System (Curiosity/Competence/Social), Autonomous Initiative, Telegram/Matrix/Discord
- **Tech:** Node 22+, TS 5.9, RxJS, grammY, FastAPI, sqlite-vec, pnpm+uv
- **Deploy:** Docker Compose, install.sh (Termux/Linux/macOS/WSL2)
- memory/ nutzt elias-memory als Backend (4-Tier: Buffer/Episodic/Semantic/Procedural)
- Research: GWT (Baars), Fast-Slow Metacognition (ICML 2025), MoA, Generative Agents

## Self-Improving Pipeline [BLOCKER Z6]
- MUSS laufen bevor neue Features gebaut werden
- HF Account: erik1988, Training via HF Jobs (SFT/DPO/GRPO + LoRA)
- MCP Server (4 aktiv): ollama-sidekick, sequential-thinking, github, memory

## API Keys & Repos
- Alle Keys in ~/.bashrc (Ollama, OpenAI, Gemini, xAI, OpenRouter, Groq, HF, NVIDIA)
- NVIDIA: Ein Key pro Modell! Endpoint: integrate.api.nvidia.com/v1/
- GitHub (Wittmann1988): 11 Repos, siehe Knowledge Graph fuer Details

## Device Mesh: `mesh` CLI, Geraete: Router .50.1, PC .50.100, Laptop .50.99, Tablet .50.50

## Ollama Cloud API
- URL: `https://ollama.com/v1` (NICHT api.ollama.com — Redirect bricht Auth)
- 32 Modelle, Key in ~/.bashrc exportiert

## Weitere Details
- Siehe [details.md](details.md) fuer: SelfEvolvingFramework, Multi-Modell Pipeline, Redundanz-Analyse, Termux Packages
