# Infrastructure Details

## Jetson Orin (Permanenter Server)
- **RAM:** 64GB unified (CPU+GPU shared)
- **Storage:** 4TB Festplatte (Netzwerk-angebunden)
- **Rolle:** Permanenter Inferenz- und Memory-Server
- **Dienste:**
  - NVIDIA Embeddings (echte semantische Vektoren, nicht Hash)
  - FAISS/Knowledge Graph Index
  - Ollama fuer lokale Modelle
  - elias-memory Desktop-Instanz (Server-Modus)
  - Roundtable lokale Agenten (Qwen3-8B, 4B etc.)
- **Netzwerk:** Im gleichen LAN wie Tablet (192.168.50.x)
- **Status:** Wird dauerhaft eingeschaltet (ab 2026-03-06)

## Desktop PC (Windows + WSL2 Kali)
- **GPU:** RTX 5090 (32GB VRAM)
- **SSH:** `kali` alias, 192.168.50.99
- **Rolle:** Training (SFT/DPO/GRPO), schwere Inferenz, Proxmark3
- **Status:** Nicht immer an

## Tablet (Samsung Galaxy Note, Termux)
- **Rolle:** Primary Interface, Leichtgewichtiger Client
- **Memory:** Lokaler SQLite-Cache, delegiert an Jetson fuer Embeddings/FAISS
- **Status:** Immer an (Hauptgeraet)

## Architektur-Entscheidung (2026-03-06)
- Tablet = duenner Client, Jetson = dicker Server
- Memory/Embeddings/KG laufen auf Jetson, nicht lokal auf Tablet
- 4TB Platte fuer: Modelle, Traces, Datasets, Knowledge Base, Backups
