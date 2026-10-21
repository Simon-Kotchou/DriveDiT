#!/usr/bin/env bash
# build-bk.sh - build steffenbk/enfusion-mcp-BK on E: with every npm/node/git side effect on E:.
# Versioned in the DriveDiT repo under tools/enfusion-mcp/ (DD-3); the live copy sits in $EMCP_ROOT.
# Runbook: tools/enfusion-mcp/README.md. Evidence: /mnt/e/drivedit_realign/enfusion/mcp-setup.md
#
#   bash build-bk.sh            # build at the pinned commit + guard patches + prune
#   bash build-bk.sh --data18   # additionally compose an E: data dir that swaps in fma-rc's Reforger 1.8 API index
#   bash build-bk.sh --smoke    # MCP initialize + tools/list over stdio (no Workbench I/O: unreachable host, guards on)
#
# Env overrides: EMCP_ROOT (work root: clones, build, caches; default /mnt/e/drivedit_realign/mcp),
#                EMCP_PATCH_DIR (guard patches; default: patches/ next to this script).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${EMCP_ROOT:-/mnt/e/drivedit_realign/mcp}"
# The WSL ext4.vhdx (~, /tmp) and /mnt/c are both on C:, which is nearly full: never build there.
case "$ROOT/" in
  /mnt/c/*|"$HOME"/*|/tmp/*) echo "refusing EMCP_ROOT=$ROOT (on C: or the WSL disk); use an E: path" >&2; exit 2;;
esac
export EMCP_ROOT="$ROOT"                           # the wrapper reads the same root
SRC="$ROOT/src/enfusion-mcp-BK"                    # pristine clone (untrusted content, never built in place)
BUILD="$ROOT/build/enfusion-mcp-BK"                # build copy
PIN=0acfa884228477043c6b4c2b8c7f0c270d648398       # BK main @ 2026-08-05 (package.json 0.12.0)
FMARC="$ROOT/src/others/fma-rc_enfusion-mcp"       # fma-rc fork @ 4cd14db (2026-10-03): 1.8 API index
PATCH_DIR="${EMCP_PATCH_DIR:-$HERE/patches}"
PATCHES=("$PATCH_DIR/bk-wsl-guard.patch" "$PATCH_DIR/20-bk-mod-build-guard.patch")
WRAPPER="$HERE/enfusion-mcp-wsl.sh"

# node from nvm, resolved via the REAL home before HOME is redirected
NODE_DIR="$(dirname "$(command -v node || ls -1d "$HOME"/.nvm/versions/node/v*/bin/node | sort -V | tail -n1)")"
export PATH="$NODE_DIR:$PATH"

# ---- every cache/log/temp/config write goes to E: --------------------------------------------------
export HOME="$ROOT/home"                           # ~/.npmrc, ~/.cache, ~/.config, ~/.gitconfig lookups
export XDG_CACHE_HOME="$ROOT/xdg/cache" XDG_CONFIG_HOME="$ROOT/xdg/config"
export XDG_STATE_HOME="$ROOT/xdg/state" XDG_DATA_HOME="$ROOT/xdg/data"
export TMPDIR="$ROOT/tmp"
export NODE_COMPILE_CACHE="$ROOT/tmp/node-compile-cache"  # npm 11 / tsc 5.9 call module.enableCompileCache()
export npm_config_cache="$ROOT/npm-cache"          # default would be ~/.npm (C:)
export npm_config_logs_dir="$ROOT/npm-logs"        # default would be <cache>/_logs
export npm_config_userconfig="$ROOT/home/.npmrc"
export npm_config_update_notifier=false npm_config_fund=false npm_config_audit=false
export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1          # playwright is a devDependency (scraper only)
export PLAYWRIGHT_BROWSERS_PATH="$ROOT/pw-browsers"
export GIT_CONFIG_NOSYSTEM=1
mkdir -p "$HOME" "$TMPDIR" "$npm_config_cache" "$npm_config_logs_dir" "$ROOT/build" \
         "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_STATE_HOME" "$XDG_DATA_HOME"

smoke() {
  # initialize + tools/list over stdio; the server never contacts Workbench for these. Belt and braces:
  # all three guards on, and the NET API host/port point at a closed WSL loopback port, not at Workbench.
  { printf '%s\n' \
      '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' \
      '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
      '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
    sleep 8; } \
  | ENFUSION_MCP_NO_AUTO_INSTALL=1 ENFUSION_MCP_NO_AUTO_LAUNCH=1 ENFUSION_MCP_WSL_PATHS=1 \
    ENFUSION_WORKBENCH_HOST=127.0.0.1 ENFUSION_WORKBENCH_PORT=1 \
    timeout 60 "$WRAPPER" 2>"$ROOT/smoke.stderr" > "$ROOT/smoke.stdout" || true
  python3 -I -B -c '
import json,sys
for line in open(sys.argv[1], encoding="utf-8"):
    line=line.strip()
    if not line: continue
    m=json.loads(line)
    if m.get("id")==1: print("serverInfo:", m["result"].get("serverInfo"))
    if m.get("id")==2:
        names=sorted(t["name"] for t in m["result"]["tools"]); print(len(names),"tools:", " ".join(names))
' "$ROOT/smoke.stdout"
  grep -h "Loaded index" "$ROOT/smoke.stderr" || true
  echo "stderr: $ROOT/smoke.stderr"
}

if [[ "${1:-}" == "--smoke" ]]; then smoke; exit 0; fi

# ---- 1. build copy at the pinned commit --------------------------------------------------------------
if [[ ! -d "$BUILD/.git" ]]; then
  git clone --quiet --no-hardlinks "$SRC" "$BUILD"
fi
git -C "$BUILD" -c advice.detachedHead=false checkout --quiet "$PIN"

# ---- 2. local guard patches (env-gated; no behaviour change unless the env vars are 1) -------------
for PATCH in "${PATCHES[@]}"; do
  if git -C "$BUILD" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "already applied: $PATCH"
  else
    git -C "$BUILD" apply --check "$PATCH"
    git -C "$BUILD" apply "$PATCH"
  fi
done

# ---- 3. install exactly the lockfile, no lifecycle scripts (skips 'prepare', esbuild postinstall) ---
cd "$BUILD"
npm ci --ignore-scripts --no-audit --no-fund --loglevel=warn
# ---- 4. compile (what 'npm run build' does, minus the clean step) ------------------------------------
rm -rf dist
node node_modules/typescript/bin/tsc -p tsconfig.build.json
test -f dist/index.js
# ---- 5. drop devDependencies (typescript, vitest, playwright, tsx) ------------------------------------
npm prune --omit=dev --ignore-scripts --no-audit --no-fund --loglevel=warn
du -sh "$BUILD/node_modules" "$BUILD/dist" || true

# ---- 6 (optional). E: data dir with the Reforger 1.8 API index from fma-rc ---------------------------
if [[ "${1:-}" == "--data18" ]]; then
  DATA="$ROOT/data"
  mkdir -p "$DATA/api"
  cp -r "$BUILD/data/patterns" "$BUILD/data/recipes" "$BUILD/data/wiki" "$DATA/"
  cp "$FMARC/data/api/arma-classes.json" "$FMARC/data/api/enfusion-classes.json" \
     "$FMARC/data/api/groups.json" "$FMARC/data/api/hierarchy.json" "$DATA/api/"
  git -C "$FMARC" rev-parse HEAD > "$DATA/api/SOURCE_fma-rc_commit.txt"
  echo "composed $DATA (wrapper exports ENFUSION_MCP_DATA_DIR automatically when api/ is populated)"
fi

"$WRAPPER" --print-env
echo "handlers: $BUILD/mod/Scripts/WorkbenchGame/EnfusionMCP ($(ls -1 "$BUILD"/mod/Scripts/WorkbenchGame/EnfusionMCP/EMCP_WB_*.c | wc -l) EMCP_WB_*.c)"
echo "next: bash $0 --smoke"
