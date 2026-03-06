# Z6: Self-Improving Pipeline (BLOCKIEREND - Nichts anderes bis umgesetzt!)

## Status: NICHT IMPLEMENTIERT (Stand 2026-03-06)
Alles bisher ist nur Dokumentation. Kein Code laeuft.

## Was fehlt (komplett)

### 1. Trace Collection — Automatisches Logging ALLER Agenten-Aktionen
- Claude Code Sessions: Jede Aktion, Entscheidung, Ergebnis -> Trace
- Agent Forge: Task-Ergebnisse (erfolg/misserfolg/qualitaet) -> Trace
- Ollama Sidekick: Alle Anfragen + Antworten -> Trace
- HackerAI: Alle Pen-Test Aktionen -> Trace
- Format: HF Chat-Template (messages mit role/content)
- Speicher: elias-memory (SQLite) + HF Dataset Push

### 2. Bewertung — Automatische Qualitaetsbewertung jeder Aktion
- Jede Agentenreaktion bekommt Score (0-1)
- Multi-Modell Bewertung: Mindestens 2 verschiedene Modelle bewerten
- Kriterien: Korrektheit, Effizienz, Kreativitaet, Zielerfuellung
- chosen/rejected Paare fuer DPO generieren
- Speicher: elias-memory + HF Dataset (erik1988/agent-evaluations)

### 3. Training — Automatischer Trainings-Loop
- SFT: Gute Traces (Score > 0.7) als Trainingsdaten
- DPO: Bewertungspaare (chosen vs rejected)
- GRPO: Tool-Use Optimierung (MiroRL-Ansatz)
- Trigger: Automatisch wenn X neue bewertete Traces vorhanden
- Plattform: HF Jobs (a10g-large fuer 7B, t4 fuer Tests)
- Ergebnis: Verbessertes Modell auf HF Hub

### 4. Deployment — Automatisches Update der Modelle
- GGUF Konvertierung nach Training
- Download auf Geraet / Ollama Registry
- Agent Forge, Sidekick, HackerAI nutzen neues Modell
- Rollback wenn Benchmark schlechter

### 5. Benchmark — Kontinuierliche Messung
- Vor/Nach Training Vergleich
- Eigene Benchmarks fuer unsere Use-Cases
- Automatischer Report nach jedem Training-Zyklus

## Pipeline-Fluss (was implementiert werden muss)

```
ALLE Agenten (Claude, Forge, Sidekick, HackerAI)
    |
    v
[Trace Collector] --- Jede Aktion wird geloggt
    |
    v
[Evaluator] --- Multi-Modell Bewertung (Score + Feedback)
    |
    v
[Dataset Builder] --- HF Dataset (Traces + Bewertungen)
    |
    v
[Trainer] --- SFT/DPO/GRPO auf HF Jobs
    |
    v
[Deployer] --- GGUF -> Ollama -> Alle Agenten
    |
    v
(Zyklus schliesst sich - verbesserte Agenten erzeugen bessere Traces)
```

## Betroffene Frameworks (ALLE muessen dieselbe Pipeline nutzen)

| Framework | Anpassungen noetig |
|---|---|
| Agent Forge (~/agent-forge/) | Trace-Export, Eval-Hook, Model-Update |
| Ollama Sidekick (~/ollama-sidekick/) | Request/Response Logging, Eval |
| HackerAI (~/HackAI/) | Action-Logging, Security-Eval |
| elias-memory (~/repos/elias-memory/) | Trace-Storage Backend |
| Way2AGI (~/repos/Way2AGI/) | Pipeline-Orchestrierung |
| DeepResearch (~/repos/DeepResearch/) | Research-Traces, Synthese-Eval |

## Inspiration: MiroMind Pipeline (vollstaendig analysiert)
- MiroFlow = Runtime + Trace Collection (= unser Agent Forge + Sidekick)
- MiroTrain = SFT + DPO Training (= unsere HF Jobs Pipeline)
- MiroRL = RL mit echten Tools (= unser GRPO Ansatz)
- MiroThinker = Fertiges Modell (= unser Ziel-Agent)
- MiroMind-M1 = Reasoning-Basis (= wir nutzen Qwen/Llama als Basis)

## Sofort-Massnahmen (Phase 0)
1. Trace-Format definieren (JSON Schema)
2. Trace-Collector als shared Library (Python) implementieren
3. In Sidekick einbauen (einfachstes Framework)
4. Erste 100 bewertete Traces sammeln
5. Dann erst Training starten

## Eriks Anweisung (2026-03-06, woertlich)
"Bevor das nicht umgesetzt wird machen wir nichts anderes."
"Jede Konzeption, jede Suche nach Features sollte durch dieses Research Framework laufen."
