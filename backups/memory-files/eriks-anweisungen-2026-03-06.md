# Eriks Anweisungen — 2026-03-06 (woertlich!)

## Kontext
Erik ist frustriert weil:
- Das Gedaechtnis schlecht ist (wir wissen nicht was wir gestern gemacht haben)
- Die Pipeline nur Markdown ist, kein laufender Code
- Kein Framework sich selbst verbessert
- MicroThinker Code nicht angepasst wurde

## Anweisungen (in Reihenfolge)

### 1. Pipeline MUSS funktionieren
"Die Pipeline funktioniert doch im Way2AGI Repository — ihr habt doch gestern Modelle trainiert"
-> TEILWEISE WAHR: generate_traces.py + train_sft.py existieren und liefen EINMAL.
-> ABER: Kein automatischer Loop, keine Bewertung, keine Session-Traces.

### 2. Code Review durch alle Modelle
"Alle verfuegbaren Modelle nehmen und unseren Code ueberpruefen lassen"
-> Einmal Code Review, nicht dreimal
-> Bugs die gefunden werden MUESSEN gefixt werden

### 3. Drei Durchlaeufe fuer Konzeptualisierung
"Drei Durchlaeufe fuer Konzeptualisierung der restlichen Umsetzung und teilweise Umsetzung"
-> Durchlauf 1: Konzept + teilweise Umsetzung
-> Durchlauf 2: Verbesserung basierend auf Durchlauf 1
-> Durchlauf 3: Finale Verbesserung
-> Benchmark wie sich das verbessert

### 4. Alles durch Pipeline
"Hauptsache alles laeuft durch Pipelines, wird bewertet und es wird aus allem gelernt"
-> JEDE Aktion muss durch die Pipeline
-> JEDE Aktion wird bewertet
-> Aus ALLEM wird gelernt

### 5. Gedaechtnis fixen
"Euer Gedaechtnis ist total beschissen, ihr wisst nicht was ihr wisst"
-> MEMORY.md ist unzureichend
-> Knowledge Graph ist zu duenn
-> elias-memory wird jetzt GENUTZT nicht nur entwickelt
-> ALLES aufschreiben was passiert

### 6. Alles dokumentieren und pushen
"Du schreibst dir jetzt alles auf was ich sage und speicherst das in der Textdatei und pusht die"
-> Diese Datei hier ist das Ergebnis
-> Muss committed und gepusht werden
-> Jede Session muss ALLES ins Gedaechtnis schreiben

## Was gestern tatsaechlich gemacht wurde (2026-03-05)
- elias-memory v0.1.0 fertiggestellt (38 Tests)
- 65 Memory-Traces generiert ueber 5 Modelle
- SFT Training v2 auf HF Jobs: Loss 2.2->1.5, Accuracy 54%->68%
- Modell auf HF Hub gepusht: erik1988/elias-memory-agent-v1
- Way2AGI Memory Server implementiert (FastAPI)
- Sidekick v2.1 mit 5 Modellen konfiguriert
- Knowledge Graph Entity Z6-Pipeline erstellt

## Was JETZT passieren muss (Prioritaet)
1. Gedaechtnis sofort verbessern (elias-mem CLI nutzen, alles speichern)
2. Code Review durch alle Modelle (1 Durchlauf)
3. Bugs fixen die gefunden werden
4. 3x Konzept-Durchlauf fuer Pipeline-Implementierung
5. Pipeline tatsaechlich implementieren (nicht nur dokumentieren!)
6. Benchmark der Verbesserung pro Durchlauf
