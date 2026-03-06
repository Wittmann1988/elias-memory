# Key Aliases & Scripts (Detail)

## AI & Claude
- `elias` -> `CLAUDE_CODE_TMPDIR=$PREFIX/tmp claude --dangerously-skip-permissions`
- `ai` -> AI Settings Manager (`~/.local/bin/ai`)
  - `ai menu` -> Interaktiv, `ai status`, `ai provider <id>`, `ai model <id>`
  - Config: `~/.config/ai-manager/settings.json`
  - Default: Anthropic Claude Sonnet 4.6 via Abo

## System Management
- `amc` -> App Manager Control (`~/app-manager-control.sh`)
- `sysctl` -> System Manager (`~/.local/bin/sysctl`)
  - status, ram, freeze, unfreeze, screen-off, screen-on, monitor, clean
- `manager` -> App Manager Server via Shizuku (Port 60001)
- `am-save-token` -> App Manager Server Token speichern
- `~/start-shizuku.sh` -> Manual Shizuku start

## Development
- `forge` -> Agent Forge CLI (`~/agent-forge`)
- `mesh` -> Device Mesh Network (`~/repos/AsusWRT-Merlin/mesh-control/mesh.sh`)
  - Config: `~/.config/mesh-network/devices.conf`
  - Befehle: status, wake, ssh, exec, scp, tunnel, setup-keys

## System Info
- 127+ Bloatware eingefroren, ~5GB RAM frei, Shizuku aktiv
- shizuku exec Bug: `grep -v` killt Exit-Code -> `return 0` anhaengen
- AppManager: `amc <cmd>`, Shizuku fuer pm/am/appops/dumpsys
