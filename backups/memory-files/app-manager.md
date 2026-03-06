# App Manager - Feature Reference

## Package
`io.github.muntashirakon.AppManager.debug`

## Key Capabilities
1. **Package Management** - Install/uninstall/freeze/unfreeze apps, extract APKs
2. **Component Blocking** - Block activities, services, receivers, providers (IFW/PM methods)
3. **Tracker Detection** - Scan & block ad/analytics trackers, VirusTotal/Pithus integration
4. **App Ops** - View/modify all app operation modes
5. **Permission Management** - Grant/revoke runtime & development permissions
6. **Backup/Restore** - Full backups with encryption (OpenPGP, RSA, ECC, AES)
7. **Process Monitor** - Running apps, force-stop, kill by UID
8. **Battery/Network** - Optimization, net policy per-app
9. **Built-in Tools** - File manager, code editor, terminal, log viewer, audio player
10. **Profiles** - Reusable configuration sets for batch operations
11. **1-Click Ops** - Block trackers, clear cache, backup all, etc.
12. **23 Batch Operations** - backup, freeze, clear data, block trackers, set app ops, etc.

## Automation from Termux
```bash
# Trigger profile
am start -n io.github.muntashirakon.AppManager.debug/io.github.muntashirakon.AppManager.crypto.auth.AuthFeatureDemultiplexer \
  --es auth "AUTH_KEY" --es feature "profile" --es prof "PROFILE_ID" --es state "on"

# Open specific pages
amc details <package>
amc running
amc profiles
```

## Control Script
`~/app-manager-control.sh` (aliased as `amc`)
- Navigation: open, details, settings, profiles, running, usage, history, labs, files, log
- Actions: install, scan, edit, explore, manifest
- Automation: profile <auth> <id> <on|off>
- System: list-apps, app-info, app-ops, set-op, perms, grant, revoke, force-stop, clear-data, disable, enable, uninstall

## Access Modes
- No-Root: basic viewing
- ADB/Wireless Debugging: permissions, app ops, force-stop
- Root: full access (component blocking, SharedPrefs, SSAID, backups with data)
- Shizuku: not yet supported natively, but we use Shizuku via Termux for equivalent ops

## Limitations
- Only "profile" feature available via automation API
- No REST/socket/CLI API
- Auth key must be set up manually in GUI
- FmProvider not exported
