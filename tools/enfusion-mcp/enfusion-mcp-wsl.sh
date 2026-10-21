#!/usr/bin/env bash
# enfusion-mcp-wsl.sh - start steffenbk/enfusion-mcp-BK (built on E:) as a stdio MCP server from WSL.
#
# DriveDiT setup, 2026-10-20 (DD-3). Versioned in the repo under tools/enfusion-mcp/ (runbook: README.md there).
# Evidence: /mnt/e/drivedit_realign/enfusion/mcp-setup.md
#
#   enfusion-mcp-wsl.sh              exec the MCP server (Claude Code calls it like this)
#   enfusion-mcp-wsl.sh --print-env  print the resolved environment to stderr and exit (touches nothing)
#   enfusion-mcp-wsl.sh --check      also open+close one TCP connection to the NET API port (opens a
#                                    socket to Workbench; it sends no request)
#
# stdout belongs to JSON-RPC: every diagnostic in this script goes to stderr.
# Any variable below can be overridden by exporting it beforehand (or via "env" in the MCP config).
set -euo pipefail

ROOT="${EMCP_ROOT:-/mnt/e/drivedit_realign/mcp}"   # work root of build-bk.sh (build, data, HOME/TMP redirects)
DIST="${EMCP_DIST:-$ROOT/build/enfusion-mcp-BK/dist/index.js}"

log() { printf '[enfusion-mcp-wsl] %s\n' "$*" >&2; }
case "$ROOT/" in   # the WSL ext4.vhdx (~, /tmp) and /mnt/c are both on C:
  /mnt/c/*|"$HOME"/*|/tmp/*) log "refusing EMCP_ROOT=$ROOT (on C: or the WSL disk)"; exit 2;;
esac

# --- node binary (resolved before HOME is redirected; nvm lives under the real HOME) -------------
if [[ -z "${NODE_BIN:-}" ]]; then
  NODE_BIN="$(command -v node 2>/dev/null || true)"
  if [[ -z "$NODE_BIN" ]]; then
    NODE_BIN="$(ls -1d "$HOME"/.nvm/versions/node/v*/bin/node 2>/dev/null | sort -V | tail -n1 || true)"
  fi
fi
[[ -x "${NODE_BIN:-}" ]] || { log "node not found (set NODE_BIN)"; exit 1; }

# --- Windows host as seen from this WSL distro -----------------------------------------------------
# NAT mode (this machine: 'wslinfo --networking-mode' -> nat; Windows 10 19045 cannot use mirrored):
# the Windows host is the default gateway. Mirrored mode (Win11 22H2+ only): 127.0.0.1.
if [[ -z "${ENFUSION_WORKBENCH_HOST:-}" ]]; then
  netmode="$(wslinfo --networking-mode 2>/dev/null || echo nat)"
  if [[ "$netmode" == "mirrored" ]]; then
    ENFUSION_WORKBENCH_HOST=127.0.0.1
  else
    ENFUSION_WORKBENCH_HOST="$(ip -4 route show default 2>/dev/null | awk '$1=="default"{print $3; exit}')"
  fi
fi
[[ -n "${ENFUSION_WORKBENCH_HOST:-}" ]] || { log "cannot determine the Windows host IP (ip route show default)"; exit 1; }
export ENFUSION_WORKBENCH_HOST
export ENFUSION_WORKBENCH_PORT="${ENFUSION_WORKBENCH_PORT:-5775}"

# --- paths: WSL form for everything the server opens itself ----------------------------------------
# Generators (project write, script_create, prefab create, mod create, game_duplicate, ...) write here.
# After the re-home (FIX_PLAN step 3) the active project is E:\DriveDiT\addons\DataCapture: keep this off C:/OneDrive.
export ENFUSION_PROJECT_PATH="${ENFUSION_PROJECT_PATH:-/mnt/e/DriveDiT/addons}"
export ENFUSION_GAME_PATH="${ENFUSION_GAME_PATH:-/mnt/d/SteamLibrary/steamapps/common/Arma Reforger}"
export ENFUSION_WORKBENCH_PATH="${ENFUSION_WORKBENCH_PATH:-/mnt/c/Program Files (x86)/Steam/steamapps/common/Arma Reforger Tools}"
# Data dir must contain api/ wiki/ patterns/ recipes/ (kb/ is always read from the package).
# Only export it when it is really populated: BK's loader silently falls back to an EMPTY index
# when api/*.json is missing (src/index/loader.ts loadJson(..., [])).
if [[ -z "${ENFUSION_MCP_DATA_DIR:-}" && -s "$ROOT/data/api/arma-classes.json" ]]; then
  export ENFUSION_MCP_DATA_DIR="$ROOT/data"
fi
# If anything ever injects handlers, it must target the addon that is actually open, not the
# alphabetically-first addon (ArmaDrive) that findFallbackGproj() would otherwise pick.
export ENFUSION_DEFAULT_MOD="${ENFUSION_DEFAULT_MOD:-DataCapture}"
# Guards added by tools/bk-wsl-guard.patch (no effect on an unpatched build):
#  - no implicit handler copy into C: on "Undefined API func"
#  - no auto-launch of Workbench from WSL (BK passes /mnt/... paths to the Windows exe)
#  - /mnt/<x>/... string params are sent to Workbench as <X>:/... (e.g. RegisterResourceFile absFilePath)
export ENFUSION_MCP_NO_AUTO_INSTALL="${ENFUSION_MCP_NO_AUTO_INSTALL:-1}"
export ENFUSION_MCP_NO_AUTO_LAUNCH="${ENFUSION_MCP_NO_AUTO_LAUNCH:-1}"
export ENFUSION_MCP_WSL_PATHS="${ENFUSION_MCP_WSL_PATHS:-1}"

# --- keep incidental HOME/TMP writes off C: (the WSL ext4.vhdx lives on C:) ------------------------
mkdir -p "$ROOT/home" "$ROOT/tmp"
export HOME="$ROOT/home"            # also: BK reads $HOME/.enfusion-mcp/config.json (absent)
export TMPDIR="$ROOT/tmp"
export XDG_CACHE_HOME="$ROOT/xdg/cache" XDG_CONFIG_HOME="$ROOT/xdg/config"
export XDG_STATE_HOME="$ROOT/xdg/state" XDG_DATA_HOME="$ROOT/xdg/data"

case "${1:-}" in
  --print-env|--check)
    for v in NODE_BIN DIST ENFUSION_WORKBENCH_HOST ENFUSION_WORKBENCH_PORT ENFUSION_PROJECT_PATH \
             ENFUSION_GAME_PATH ENFUSION_WORKBENCH_PATH ENFUSION_MCP_DATA_DIR ENFUSION_DEFAULT_MOD \
             ENFUSION_MCP_NO_AUTO_INSTALL ENFUSION_MCP_NO_AUTO_LAUNCH ENFUSION_MCP_WSL_PATHS HOME TMPDIR; do
      printf '  %-30s %s\n' "$v" "${!v:-<unset>}" >&2
    done
    [[ -f "$DIST" ]] && log "dist present: $DIST" || log "dist MISSING: $DIST (run build-bk.sh)"
    for p in "$ENFUSION_PROJECT_PATH" "$ENFUSION_GAME_PATH/addons" "$ENFUSION_WORKBENCH_PATH/Workbench/ArmaReforgerWorkbenchSteamDiag.exe"; do
      [[ -e "$p" ]] && log "ok      $p" || log "MISSING $p"
    done
    if [[ "${1}" == "--check" ]]; then
      if timeout 3 bash -c "exec 3<>/dev/tcp/${ENFUSION_WORKBENCH_HOST}/${ENFUSION_WORKBENCH_PORT}" 2>/dev/null; then
        log "TCP ${ENFUSION_WORKBENCH_HOST}:${ENFUSION_WORKBENCH_PORT} reachable (handlers not tested; use wb_diagnose)"
      else
        log "TCP ${ENFUSION_WORKBENCH_HOST}:${ENFUSION_WORKBENCH_PORT} NOT reachable (Workbench closed, NET API off, or Windows firewall)"
      fi
    fi
    exit 0
    ;;
esac

[[ -f "$DIST" ]] || { log "missing $DIST - run build-bk.sh (tools/enfusion-mcp/README.md) first"; exit 1; }
exec "$NODE_BIN" "$DIST" "$@"
