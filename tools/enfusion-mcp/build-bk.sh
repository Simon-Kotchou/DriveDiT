#!/usr/bin/env bash
# build-bk.sh - build steffenbk/enfusion-mcp-BK on E: with every npm/node/git side effect on E:.
# Versioned in the DriveDiT repo under tools/enfusion-mcp/ (DD-3); the live copy sits in $EMCP_ROOT.
# Runbook: tools/enfusion-mcp/README.md. Evidence: /mnt/e/drivedit_realign/enfusion/mcp-setup.md
#
#   bash build-bk.sh            # build at the pinned commit + guard patches + prune
#   bash build-bk.sh --data18   # additionally compose an E: data dir that swaps in fma-rc's Reforger 1.8 API index
#   bash build-bk.sh --smoke    # MCP initialize + tools/list over stdio (no Workbench I/O: unreachable host, guards on);
#                               # exits 1 unless serverInfo and the expected tool count come back
#
# Env overrides: EMCP_ROOT (work root: clones, build, caches; default /mnt/e/drivedit_realign/mcp; must be an
#                absolute path on a drive other than C:), EMCP_PATCH_DIR (guard patches; default: patches/ next
#                to this script).
# Re-running is safe: the clone, checkout, patches (already-applied check), npm ci, tsc, prune and the data
# copy all converge on the same tree.
set -euo pipefail

die() { echo "build-bk: $*" >&2; exit "${EXIT_CODE:-1}"; }
# True only for an absolute path that resolves (symlinks and '..' included) onto a Windows drive other than C:.
# /mnt/c and the whole WSL ext4.vhdx (~, /tmp, /var, ...) are on C:, which is nearly full; relative paths
# would resolve against the caller's cwd.
off_c_drive() {
  local r="" re='^/mnt/[abdefghijklmnopqrstuvwxyz](/|$)'
  [[ "${1:-}" == /* ]] || return 1
  r="$(realpath -m -- "$1" 2>/dev/null)" || return 1
  [[ "$r" =~ $re ]]
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${EMCP_ROOT:-/mnt/e/drivedit_realign/mcp}"
off_c_drive "$ROOT" || EXIT_CODE=2 die "refusing EMCP_ROOT=$ROOT: use an absolute path on a drive other than C: (/mnt/c and the WSL disk are on C:)"
ROOT="$(realpath -m -- "$ROOT")"
export EMCP_ROOT="$ROOT"                           # the wrapper reads the same root
SRC="$ROOT/src/enfusion-mcp-BK"                    # pristine clone (untrusted content, never built in place)
BUILD="$ROOT/build/enfusion-mcp-BK"                # build copy
PIN=0acfa884228477043c6b4c2b8c7f0c270d648398       # BK main @ 2026-08-05 (package.json 0.12.0)
EXPECT_TOOLS=54                                    # tools/list size at PIN (smoke fails on anything else)
FMARC="$ROOT/src/others/fma-rc_enfusion-mcp"       # fma-rc fork: source of the Reforger 1.8 API index
FMARC_PIN=4cd14dbbfb8a3b2832b0ec2480a5dcc8fe1b2c49 # fma-rc @ 2026-10-03 (887 + 8,085 classes, Tools buildid 24870687)
PATCH_DIR="${EMCP_PATCH_DIR:-$HERE/patches}"
PATCHES=("$PATCH_DIR/bk-wsl-guard.patch" "$PATCH_DIR/20-bk-mod-build-guard.patch")
WRAPPER="$HERE/enfusion-mcp-wsl.sh"

# node from nvm, resolved via the REAL home before HOME is redirected
NODE_BIN="$(command -v node 2>/dev/null || true)"
[[ -n "$NODE_BIN" ]] || NODE_BIN="$(ls -1d "$HOME"/.nvm/versions/node/v*/bin/node 2>/dev/null | sort -V | tail -n1 || true)"
[[ -x "$NODE_BIN" ]] || die "node not found (install it with nvm under the real HOME, or put it on PATH)"
NODE_DIR="$(dirname "$NODE_BIN")"
[[ -x "$NODE_DIR/npm" ]] || die "npm not found next to $NODE_BIN"
export PATH="$NODE_DIR:$PATH"

# ---- every cache/log/temp/config write goes to E: --------------------------------------------------
export HOME="$ROOT/home"                           # ~/.npmrc, ~/.cache, ~/.config, ~/.gitconfig lookups
export XDG_CACHE_HOME="$ROOT/xdg/cache" XDG_CONFIG_HOME="$ROOT/xdg/config"
export XDG_STATE_HOME="$ROOT/xdg/state" XDG_DATA_HOME="$ROOT/xdg/data"
export TMPDIR="$ROOT/tmp"
export NODE_COMPILE_CACHE="$ROOT/tmp/node-compile-cache"  # npm 11 / tsc 5.9 call module.enableCompileCache()
unset NODE_OPTIONS NODE_V8_COVERAGE                # inherited node flags could write reports/coverage elsewhere
export npm_config_cache="$ROOT/npm-cache"          # default would be ~/.npm (C:)
export npm_config_logs_dir="$ROOT/npm-logs"        # default would be <cache>/_logs
export npm_config_userconfig="$ROOT/home/.npmrc"
export npm_config_update_notifier=false npm_config_fund=false npm_config_audit=false
export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1          # playwright is a devDependency (scraper only)
export PLAYWRIGHT_BROWSERS_PATH="$ROOT/pw-browsers"
export GIT_CONFIG_NOSYSTEM=1
mkdir -p "$HOME" "$TMPDIR" "$npm_config_cache" "$npm_config_logs_dir" "$ROOT/build" \
         "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_STATE_HOME" "$XDG_DATA_HOME"
cd "$ROOT"                                         # never run anything from the caller's (possibly C:) cwd

smoke() {
  # initialize + tools/list over stdio; the server never contacts Workbench for these. Belt and braces:
  # the wrapper forces all three guards on, and the NET API host/port point at a closed WSL loopback
  # port, not at Workbench.
  local rc=0
  { printf '%s\n' \
      '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' \
      '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
      '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
    sleep 8; } \
  | ENFUSION_WORKBENCH_HOST=127.0.0.1 ENFUSION_WORKBENCH_PORT=1 \
    timeout 60 "$WRAPPER" 2>"$ROOT/smoke.stderr" > "$ROOT/smoke.stdout" || true
  python3 -I -B -c '
import json, sys
info, names = None, None
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    m = json.loads(line)
    if m.get("id") == 1:
        info = m.get("result", {}).get("serverInfo")
    if m.get("id") == 2:
        names = sorted(t["name"] for t in m.get("result", {}).get("tools", []))
print("serverInfo:", info)
print(len(names or []), "tools:", " ".join(names or []))
ok = bool(info) and info.get("name") == "enfusion-mcp" and names is not None and len(names) == int(sys.argv[2])
if not ok:
    print("SMOKE FAILED: expected serverInfo enfusion-mcp and %s tools" % sys.argv[2], file=sys.stderr)
sys.exit(0 if ok else 1)
' "$ROOT/smoke.stdout" "$EXPECT_TOOLS" || rc=1
  grep -h "Loaded index" "$ROOT/smoke.stderr" || true
  if [[ $rc -ne 0 ]]; then sed -n '1,20p' "$ROOT/smoke.stderr" >&2; fi
  echo "stderr: $ROOT/smoke.stderr"
  return "$rc"
}

if [[ "${1:-}" == "--smoke" ]]; then smoke; exit $?; fi

# ---- 1. build copy at the pinned commit --------------------------------------------------------------
[[ -d "$SRC/.git" ]] || die "missing pristine clone $SRC (README: Build, prerequisite)"
if [[ ! -d "$BUILD/.git" ]]; then
  git clone --quiet --no-hardlinks "$SRC" "$BUILD"
fi
git -C "$BUILD" -c advice.detachedHead=false checkout --quiet "$PIN"

# ---- 2. local guard patches (env-gated; no behaviour change unless the env vars are 1) -------------
for PATCH in "${PATCHES[@]}"; do
  [[ -f "$PATCH" ]] || die "missing guard patch $PATCH"
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
"$NODE_BIN" node_modules/typescript/bin/tsc -p tsconfig.build.json
test -f dist/index.js
grep -q 'ENFUSION_MCP_NO_AUTO_INSTALL' dist/workbench/client.js && grep -q 'ENFUSION_MCP_NO_AUTO_LAUNCH' dist/tools/mod.js \
  || die "guard code missing from dist after tsc"
# ---- 5. drop devDependencies (typescript, vitest, playwright, tsx) ------------------------------------
npm prune --omit=dev --ignore-scripts --no-audit --no-fund --loglevel=warn
du -sh "$BUILD/node_modules" "$BUILD/dist" || true
cd "$ROOT"

# ---- 6 (optional). E: data dir with the Reforger 1.8 API index from fma-rc ---------------------------
# Read from the pinned commit's objects (not the working tree), so a moved checkout cannot change the index.
if [[ "${1:-}" == "--data18" ]]; then
  DATA="$ROOT/data"
  git -C "$FMARC" cat-file -e "$FMARC_PIN^{commit}" 2>/dev/null || die "fma-rc commit $FMARC_PIN not in $FMARC (README: Build, prerequisite)"
  mkdir -p "$DATA/api"
  cp -r "$BUILD/data/patterns" "$BUILD/data/recipes" "$BUILD/data/wiki" "$DATA/"
  for f in arma-classes.json enfusion-classes.json groups.json hierarchy.json; do
    git -C "$FMARC" show "$FMARC_PIN:data/api/$f" > "$DATA/api/$f.part"
    mv -f "$DATA/api/$f.part" "$DATA/api/$f"
  done
  echo "$FMARC_PIN" > "$DATA/api/SOURCE_fma-rc_commit.txt"
  echo "composed $DATA (wrapper exports ENFUSION_MCP_DATA_DIR automatically when api/ is populated)"
fi

"$WRAPPER" --print-env
echo "handlers: $BUILD/mod/Scripts/WorkbenchGame/EnfusionMCP ($(ls -1 "$BUILD"/mod/Scripts/WorkbenchGame/EnfusionMCP/EMCP_WB_*.c | wc -l) EMCP_WB_*.c)"
echo "next: bash $HERE/build-bk.sh --smoke"
