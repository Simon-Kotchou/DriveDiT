#!/usr/bin/env bash
# enfusion-mcp-wsl.sh - start steffenbk/enfusion-mcp-BK (built on the data drive) as a stdio MCP server from WSL.
#
# DriveDiT setup, 2026-10-20 (DD-3). Versioned in the repo under tools/enfusion-mcp/ (runbook: README.md there).
# Evidence: <REALIGN_ROOT>/enfusion/mcp-setup.md. <REALIGN_ROOT> and the defaults below come from the local config
# (DD-22; tools/local_config.py); without it the wrapper refuses (exit 2). Write paths pass its write boundary.
#
#   enfusion-mcp-wsl.sh              exec the MCP server (Claude Code calls it like this)
#   enfusion-mcp-wsl.sh --print-env  print the resolved environment to stderr and exit (creates nothing)
#   enfusion-mcp-wsl.sh --print-root print the work root (EMCP_ROOT, else <REALIGN_ROOT>/mcp) and exit
#   enfusion-mcp-wsl.sh --check      also open+close one TCP connection to the NET API port (opens a
#                                    socket to Workbench; it sends no request)
#
# DD-14a (CC-05): the server runs behind mcp_gate.py, which picks the pin (EMCP_PIN, else the ACTIVE pointer),
# serves only that pin's API index ($EMCP_ROOT/pins/<pin id>/data, stamped by api_index.py) and its project root
# (pinlib.DEPLOY_ROOTS), and runs the endpoint handshake (endpoint_handshake.py) before every wb_* call. A second
# config for another pin sets EMCP_PIN in the MCP entry's env. Any mismatch refuses before the server starts.
#
# stdout belongs to JSON-RPC: every diagnostic in this script goes to stderr.
# Paths, host and port can be overridden by exporting them beforehand (or via "env" in the MCP config).
# The three guards cannot: they are always 1 here, and the wrapper refuses to start a build without them.
# Exit codes: 0 ok; 1 missing node/dist, unguarded build or no Windows host IP; 2 refused path, bad setting or no
# local config.
set -euo pipefail

NAME=enfusion-mcp-wsl
log() { printf '[enfusion-mcp-wsl] %s\n' "$*" >&2; }
die() { local code="$1"; shift; log "$*"; exit "$code"; }
HERE="$(dirname "$(realpath -- "${BASH_SOURCE[0]}")")"
# DD-22: this machine's paths come from the untracked local config (tools/local_config.py, drivedit.local.env).
# An installed MCP root holds a copy of local_config.py next to this script; tell it the file with
# DRIVEDIT_LOCAL_CONFIG (the MCP entry's "env"). In the repo the loader is ../local_config.py.
LOCAL_CONFIG_PY=""
for _c in "$HERE/local_config.py" "$HERE/../local_config.py"; do
  if [[ -f "$_c" ]]; then LOCAL_CONFIG_PY="$_c"; break; fi
done
local_config() {   # local_config KEY [--win | --drive]: one value; the loader names the file and key on a refusal
  [[ -n "$LOCAL_CONFIG_PY" ]] || { echo "$NAME: local_config.py is missing next to this script and in ../" >&2; return 2; }
  /usr/bin/python3 -I -B "$LOCAL_CONFIG_PY" get "$@"
}
local_config DATA_ROOT >/dev/null || die 2 "no local config (see the message above); nothing started"

# Every path this wrapper (or the server behind it) writes under goes through the write boundary.
write_ok() {   # write_ok PATH: print the resolved path when the write boundary allows writing there, else fail
  # (tools/local_config.py check-write: RAM or the verified data drive, never a volume whose root holds
  # Windows/System32, whatever the config says). A relative path is refused: it would resolve against the cwd.
  [[ -n "$LOCAL_CONFIG_PY" ]] || { echo "$NAME: local_config.py is missing next to this script and in ../" >&2; return 2; }
  [[ "${1:-}" == /* ]] || { echo "$NAME: '${1:-}' is not an absolute path" >&2; return 1; }
  /usr/bin/python3 -I -B "$LOCAL_CONFIG_PY" check-write "$1"
}

ROOT="${EMCP_ROOT:-}"                              # work root of build-bk.sh (build, data, HOME/TMP redirects)
if [[ -z "$ROOT" ]]; then
  ROOT="$(local_config REALIGN_ROOT)/mcp" || die 2 "no REALIGN_ROOT in the local config (see the message above)"
fi
# --print-root: print the work root as resolved here (before its check) and exit (tools/migrate_local_config.py).
if [[ "${1:-}" == "--print-root" ]]; then printf '%s\n' "$ROOT"; exit 0; fi
ROOT_OK="$(write_ok "$ROOT")" || die 2 "refusing EMCP_ROOT=$ROOT: the write boundary refuses it (see the message above)"
ROOT="$ROOT_OK"
export EMCP_ROOT="$ROOT"                           # mcp_gate.py serves the index of the same root
DIST="${EMCP_DIST:-$ROOT/build/enfusion-mcp-BK/dist/index.js}"

# --- node binary (resolved before HOME is redirected; nvm lives under the real HOME) -------------
if [[ -z "${NODE_BIN:-}" ]]; then
  NODE_BIN="$(command -v node 2>/dev/null || true)"
  if [[ -z "$NODE_BIN" ]]; then
    NODE_BIN="$(ls -1d "$HOME"/.nvm/versions/node/v*/bin/node 2>/dev/null | sort -V | tail -n1 || true)"
  fi
fi
[[ -x "${NODE_BIN:-}" ]] || die 1 "node not found (set NODE_BIN)"

# --- Windows host as seen from this WSL distro -----------------------------------------------------
# NAT mode ('wslinfo --networking-mode' -> nat): the Windows host is the default gateway (mcp_gate.py verifies it
# against the Windows vEthernet (WSL) adapter). Mirrored mode (recent Windows 11 only): 127.0.0.1.
is_ipv4() {
  local o re='^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$'
  [[ "${1:-}" =~ $re ]] || return 1
  for o in "${BASH_REMATCH[@]:1}"; do (( 10#$o <= 255 )) || return 1; done
}
# IPv4 default gateway: the 'via' address of 'ip -4 route show default'; /proc/net/route when 'ip' is
# missing or fails. Prints nothing (never fails) when there is no gateway, e.g. a device-only route.
default_gateway() {
  local out="" gw=""
  if out="$(ip -4 route show default 2>/dev/null)"; then
    gw="$(awk '{ for (i = 1; i < NF; i++) if ($i == "via") { print $(i + 1); exit } }' <<<"$out")"
  fi
  if [[ -z "$gw" && -r /proc/net/route ]]; then
    gw="$(awk 'NR > 1 && $2 == "00000000" && $8 == "00000000" && $3 != "00000000" { print $3; exit }' /proc/net/route)"
    [[ "$gw" =~ ^[0-9A-Fa-f]{8}$ ]] && gw="$(printf '%d.%d.%d.%d' "0x${gw:6:2}" "0x${gw:4:2}" "0x${gw:2:2}" "0x${gw:0:2}")"
  fi
  printf '%s' "$gw"
}
if [[ -z "${ENFUSION_WORKBENCH_HOST:-}" ]]; then
  netmode="$(wslinfo --networking-mode 2>/dev/null | tr -d '\r' || true)"
  if [[ "$netmode" == "mirrored" ]]; then
    ENFUSION_WORKBENCH_HOST=127.0.0.1
  else
    ENFUSION_WORKBENCH_HOST="$(default_gateway)"
    is_ipv4 "$ENFUSION_WORKBENCH_HOST" || die 1 "cannot determine the Windows host IP: no IPv4 default gateway" \
      "in 'ip -4 route show default' or /proc/net/route (networking mode '${netmode:-unknown}')." \
      "Set ENFUSION_WORKBENCH_HOST explicitly (the README explains how to find it)."
  fi
fi
[[ "$ENFUSION_WORKBENCH_HOST" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]] || die 2 "invalid ENFUSION_WORKBENCH_HOST '$ENFUSION_WORKBENCH_HOST'"
ENFUSION_WORKBENCH_PORT="${ENFUSION_WORKBENCH_PORT:-5775}"
[[ "$ENFUSION_WORKBENCH_PORT" =~ ^[0-9]{1,5}$ ]] && (( 10#$ENFUSION_WORKBENCH_PORT >= 1 && 10#$ENFUSION_WORKBENCH_PORT <= 65535 )) \
  || die 2 "invalid ENFUSION_WORKBENCH_PORT '$ENFUSION_WORKBENCH_PORT'"
export ENFUSION_WORKBENCH_HOST ENFUSION_WORKBENCH_PORT

# --- paths: WSL form for everything the server opens itself ----------------------------------------
# Generators (project write, script_create, prefab create, mod create, game_duplicate, ...) write into the project
# path, and so would any handler copy: it must stay off the system drive (never the OneDrive addons dir).
# mcp_gate.py sets it to the pin's project root (<DATA_ROOT>/addons for a stable pin); a value set here must be that root.
if [[ -n "${ENFUSION_PROJECT_PATH:-}" ]]; then
  write_ok "$ENFUSION_PROJECT_PATH" >/dev/null || die 2 "refusing ENFUSION_PROJECT_PATH=$ENFUSION_PROJECT_PATH: the server writes there, and the write boundary refuses it (see the message above)"
fi
# ENFUSION_GAME_PATH and ENFUSION_WORKBENCH_PATH (read only; launch and mod build are guarded) are the pin's
# installs, and ENFUSION_MCP_DATA_DIR is the pin's index: mcp_gate.py sets all three. BK's loader silently falls back
# to an EMPTY index when api/*.json is missing (src/index/loader.ts loadJson(..., [])), so the gate checks every
# api file against the index stamp first.
GATE="${EMCP_GATE:-$HERE/mcp_gate.py}"
PY=/usr/bin/python3
# If anything ever injects handlers, it must target the addon that is actually open, not the
# alphabetically-first addon that findFallbackGproj() would otherwise pick.
export ENFUSION_DEFAULT_MOD="${ENFUSION_DEFAULT_MOD:-DataCapture}"

# --- guards (patches/bk-wsl-guard.patch + patches/20-bk-mod-build-guard.patch) ---------------------
#  - ENFUSION_MCP_NO_AUTO_INSTALL: no implicit handler copy on "Undefined API func"
#  - ENFUSION_MCP_NO_AUTO_LAUNCH:  no auto-launch / wb_launch / mod build (BK passes /mnt/... paths to the Windows exe)
#  - ENFUSION_MCP_WSL_PATHS:       /mnt/<x>/... string params are sent to Workbench as <X>:/...
# The patched code tests for exactly "1", so any other value (0, true, yes, ...) would silently turn a
# guard off. From WSL they are always on.
for g in ENFUSION_MCP_NO_AUTO_INSTALL ENFUSION_MCP_NO_AUTO_LAUNCH ENFUSION_MCP_WSL_PATHS; do
  if [[ -n "${!g:-}" && "${!g}" != "1" ]]; then log "ignoring $g=${!g}: this guard is always 1 from WSL"; fi
  export "$g=1"
done
# A build without the guard code would ignore the variables above: refuse to start one.
DIST_DIR="$(dirname "$DIST")"
guarded_build() {
  [[ -f "$DIST_DIR/workbench/client.js" && -f "$DIST_DIR/tools/mod.js" ]] || return 1
  grep -q 'ENFUSION_MCP_NO_AUTO_INSTALL' "$DIST_DIR/workbench/client.js" \
    && grep -q 'ENFUSION_MCP_NO_AUTO_LAUNCH' "$DIST_DIR/workbench/client.js" \
    && grep -q 'ENFUSION_MCP_WSL_PATHS' "$DIST_DIR/workbench/client.js" \
    && grep -q 'ENFUSION_MCP_NO_AUTO_LAUNCH' "$DIST_DIR/tools/mod.js"
}

# --- keep incidental writes off the system drive (the WSL ext4.vhdx lives on it) -------------------
export HOME="$ROOT/home"            # also: BK reads $HOME/.enfusion-mcp/config.json (absent)
export TMPDIR="$ROOT/tmp"           # also the default dir of node's compile cache
export XDG_CACHE_HOME="$ROOT/xdg/cache" XDG_CONFIG_HOME="$ROOT/xdg/config"
export XDG_STATE_HOME="$ROOT/xdg/state" XDG_DATA_HOME="$ROOT/xdg/data"
# Inherited node settings could write elsewhere (diagnostic reports, coverage, caches).
unset NODE_OPTIONS NODE_V8_COVERAGE NODE_COMPILE_CACHE
# The gate's test hooks (fixture route, resolver and ipconfig text) never reach a real session.
unset EMCP_TEST_ROUTE EMCP_TEST_RESOLV EMCP_TEST_IPCONFIG
# The server's cwd: BK resolves relative tool arguments (projectPath, wb_cleanup modDir, mod outputPath)
# against it, so it must not stay the caller's cwd.
WORK_CWD="$ROOT/cwd"

case "${1:-}" in
  --print-env|--check)
    for v in NODE_BIN DIST GATE EMCP_PIN ENFUSION_WORKBENCH_HOST ENFUSION_WORKBENCH_PORT ENFUSION_PROJECT_PATH \
             ENFUSION_GAME_PATH ENFUSION_WORKBENCH_PATH ENFUSION_MCP_DATA_DIR ENFUSION_DEFAULT_MOD \
             ENFUSION_MCP_NO_AUTO_INSTALL ENFUSION_MCP_NO_AUTO_LAUNCH ENFUSION_MCP_WSL_PATHS HOME TMPDIR WORK_CWD; do
      printf '  %-30s %s\n' "$v" "${!v:-<unset>}" >&2
    done
    if [[ -f "$DIST" ]]; then log "dist present: $DIST"; else log "dist MISSING: $DIST (run build-bk.sh)"; fi
    if guarded_build; then log "guards compiled into dist: yes"; else log "guards compiled into dist: NO (the server will refuse to start)"; fi
    if [[ -f "$GATE" ]] && "$PY" -I -B "$GATE" --print-config; then log "pin gate: ok"; else log "pin gate: REFUSED or missing ($GATE); the server will not start"; fi
    if [[ "$1" == "--check" ]]; then
      if timeout 3 bash -c 'exec 3<>"/dev/tcp/$1/$2"' _ "$ENFUSION_WORKBENCH_HOST" "$ENFUSION_WORKBENCH_PORT" 2>/dev/null; then
        log "TCP ${ENFUSION_WORKBENCH_HOST}:${ENFUSION_WORKBENCH_PORT} reachable (handlers not tested; use wb_connect)"
      else
        log "TCP ${ENFUSION_WORKBENCH_HOST}:${ENFUSION_WORKBENCH_PORT} NOT reachable (Workbench closed, NET API off, or Windows firewall)"
      fi
    fi
    exit 0
    ;;
esac

[[ -f "$DIST" ]] || die 1 "missing $DIST - run build-bk.sh (tools/enfusion-mcp/README.md) first"
guarded_build || die 1 "refusing to start $DIST: the WSL guard code is not compiled in (rebuild with build-bk.sh)"
[[ -f "$GATE" && -x "$PY" ]] || die 1 "missing $GATE or $PY: the server never runs without the pin gate (DD-14a)"
mkdir -p "$HOME" "$TMPDIR" "$WORK_CWD"
cd "$WORK_CWD"
exec "$PY" -I -B "$GATE" -- "$NODE_BIN" "$DIST" "$@"
