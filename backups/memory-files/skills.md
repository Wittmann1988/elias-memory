# Skills - Vollstaendige Referenz

## AUTOMATISCH nutzen (nie warten bis User aufruft!)

### Wann welchen Skill?

| Situation | Skill | Prioritaet |
|-----------|-------|-----------|
| Neue Feature/Idee | brainstorming | PFLICHT |
| Multi-Step Task | writing-plans | PFLICHT |
| Plan ausfuehren | executing-plans | PFLICHT |
| Bug/Fehler/Test-Failure | systematic-debugging | PFLICHT |
| Feature implementieren | test-driven-development | PFLICHT |
| "Fertig" sagen | verification-before-completion | PFLICHT |
| 2+ unabhaengige Tasks | dispatching-parallel-agents | EMPFOHLEN |
| Implementation Plan | subagent-driven-development | EMPFOHLEN |
| Code geschrieben | simplify | EMPFOHLEN |
| Grosses Feature fertig | requesting-code-review | EMPFOHLEN |
| Code Review bekommen | receiving-code-review | EMPFOHLEN |
| Branch fertig | finishing-a-development-branch | EMPFOHLEN |
| Feature braucht Isolation | using-git-worktrees | OPTIONAL |
| Neuen Skill erstellen | writing-skills | OPTIONAL |

### HuggingFace Skills

| Situation | Skill |
|-----------|-------|
| Modell trainieren | hugging-face-model-trainer |
| Dataset erstellen/verwalten | hugging-face-datasets |
| Hub Operations (upload/download) | hugging-face-cli |
| Model evaluieren | hugging-face-evaluation |
| Cloud-Workload starten | hugging-face-jobs |
| Reusable Script bauen | hugging-face-tool-builder |
| Training-Metriken tracken | hugging-face-trackio |
| Paper publizieren | hugging-face-paper-publisher |
| Gradio UI bauen | huggingface-gradio |
| Dataset inspizieren | hugging-face-dataset-viewer |

### Domain Skills

| Situation | Skill |
|-----------|-------|
| Claude API nutzen | claude-api |
| PR reviewen | code-review |
| Web UI bauen | frontend-design |
| Full-Stack Feature | full-stack-feature |
| Cross-Platform | multi-platform |
| Next.js App | nextjs-app-router-patterns |
| React Native | react-native-architecture |
| React State | react-state-management |
| Tailwind Design | tailwind-design-system |
| ML Pipeline | ml-pipeline-workflow |

## Entscheidungsbaum

```
User-Nachricht empfangen
  -> Koennte ein Skill passen? (auch nur 1% Chance)
     JA -> Skill aufrufen BEVOR irgendetwas anderes passiert
     NEIN -> Normal antworten
  -> Mehrere Skills passen?
     -> Prozess-Skills zuerst (brainstorming, debugging)
     -> Dann Implementation-Skills (frontend-design, etc.)
  -> "Baue X" / "Erstelle Y"
     -> brainstorming ZUERST, dann Implementation
  -> "Fix Bug" / "Fehler beheben"
     -> systematic-debugging ZUERST, dann Fix
```
