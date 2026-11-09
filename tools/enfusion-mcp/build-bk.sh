#!/usr/bin/env bash
# build-bk.sh - build steffenbk/enfusion-mcp-BK on the data drive with every npm/node/git side effect there.
# Versioned in the DriveDiT repo under tools/enfusion-mcp/ (DD-3); the live copy sits in $EMCP_ROOT.
# Runbook: tools/enfusion-mcp/README.md. Evidence: <REALIGN_ROOT>/enfusion/mcp-setup.md (REALIGN_ROOT comes from the
# local config, DD-22; tools/local_config.py).
#
#   bash build-bk.sh            # build at the pinned commit + guard patches + prune
#   bash build-bk.sh --data18   # additionally compose a data dir that swaps in fma-rc's Reforger 1.8 API index
#   bash build-bk.sh --smoke    # MCP initialize + tools/list over stdio (no Workbench I/O: unreachable host, guards on);
#                               # exits 1 unless serverInfo and the expected tool count come back
#
# Env overrides: EMCP_ROOT (work root: clones, build, caches; default <REALIGN_ROOT>/mcp; must pass the write
#                boundary of tools/local_config.py: the verified data drive, never the system drive), EMCP_PATCH_DIR
#                (guard patches; default: patches/ next to this script), EMCP_PREVIOUS_PATCHES (the patch files an
#                older build copy holds, ':'-separated; only when no record of them exists, see step 1).
# Re-running is safe: the clone, the patches (applied once; a build copy is reset only when its changes are proven to
# be a previous patch set), npm ci, tsc, prune and the data copy all converge on the same tree.
set -euo pipefail

NAME=build-bk
die() { echo "build-bk: $*" >&2; exit "${EXIT_CODE:-1}"; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
local_config DATA_ROOT >/dev/null || EXIT_CODE=2 die "no local config (see the message above)"
# The work root (and so every cache, log and temp file below) goes through the write boundary.
write_ok() {   # write_ok PATH: print the resolved path when the write boundary allows writing there, else fail
  # (tools/local_config.py check-write: RAM or the verified data drive, never a volume whose root holds
  # Windows/System32, whatever the config says). A relative path is refused: it would resolve against the cwd.
  [[ -n "$LOCAL_CONFIG_PY" ]] || { echo "$NAME: local_config.py is missing next to this script and in ../" >&2; return 2; }
  [[ "${1:-}" == /* ]] || { echo "$NAME: '${1:-}' is not an absolute path" >&2; return 1; }
  /usr/bin/python3 -I -B "$LOCAL_CONFIG_PY" check-write "$1"
}

ROOT="${EMCP_ROOT:-}"
if [[ -z "$ROOT" ]]; then
  ROOT="$(local_config REALIGN_ROOT)/mcp" || EXIT_CODE=2 die "no REALIGN_ROOT in the local config (see the message above)"
fi
ROOT_OK="$(write_ok "$ROOT")" || EXIT_CODE=2 die "refusing EMCP_ROOT=$ROOT: the write boundary refuses it (see the message above)"
ROOT="$ROOT_OK"
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

# ---- every cache/log/temp/config write goes under ROOT, on the data drive ---------------------------
export HOME="$ROOT/home"                           # ~/.npmrc, ~/.cache, ~/.config, ~/.gitconfig lookups
export XDG_CACHE_HOME="$ROOT/xdg/cache" XDG_CONFIG_HOME="$ROOT/xdg/config"
export XDG_STATE_HOME="$ROOT/xdg/state" XDG_DATA_HOME="$ROOT/xdg/data"
export TMPDIR="$ROOT/tmp"
export NODE_COMPILE_CACHE="$ROOT/tmp/node-compile-cache"  # npm 11 / tsc 5.9 call module.enableCompileCache()
unset NODE_OPTIONS NODE_V8_COVERAGE                # inherited node flags could write reports/coverage elsewhere
export npm_config_cache="$ROOT/npm-cache"          # default would be ~/.npm (the WSL disk)
export npm_config_logs_dir="$ROOT/npm-logs"        # default would be <cache>/_logs
export npm_config_userconfig="$ROOT/home/.npmrc"
export npm_config_update_notifier=false npm_config_fund=false npm_config_audit=false
export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1          # playwright is a devDependency (scraper only)
export PLAYWRIGHT_BROWSERS_PATH="$ROOT/pw-browsers"
export GIT_CONFIG_NOSYSTEM=1
mkdir -p "$HOME" "$TMPDIR" "$npm_config_cache" "$npm_config_logs_dir" "$ROOT/build" \
         "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_STATE_HOME" "$XDG_DATA_HOME"
cd "$ROOT"                                         # never run anything from the caller's cwd

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
  git -C "$BUILD" -c advice.detachedHead=false checkout --quiet "$PIN"
fi
# The build copy is reset only when its changes are provably ours. Its tracked files (on disk) must be exactly the
# pinned tree, the pinned tree plus the current guard patches, or the pinned tree plus the PREVIOUS patch set: the
# patches the last build applied, as recorded in $APPLIED_LIST (or, for a build copy made before that record
# existed, the patch files EMCP_PREVIOUS_PATCHES names, ':'-separated). Only in that last case is the copy reset to
# the pin (checkout -f; untracked node_modules/ and dist/ are kept), after `git apply -R --check` of the previous
# patches and a tree comparison prove the difference is exactly theirs. Anything else (another commit, staged
# edits, any other change) refuses and touches nothing. Trees are compared through a scratch index under $TMPDIR.
APPLIED_DIR="$ROOT/build/applied-patches"
APPLIED_LIST="$ROOT/build/applied-patches.list"
SCRATCH_INDEX="$TMPDIR/build-bk-scratch-index"
for PATCH in "${PATCHES[@]}"; do [[ -f "$PATCH" ]] || die "missing guard patch $PATCH"; done
tree_of() {   # tree_of [PATCH...]: the tree of PIN with the patches applied, in the scratch index
  GIT_INDEX_FILE="$SCRATCH_INDEX" git -C "$BUILD" read-tree "$PIN" \
    && { [[ $# -eq 0 ]] || GIT_INDEX_FILE="$SCRATCH_INDEX" git -C "$BUILD" apply --cached "$@"; } \
    && GIT_INDEX_FILE="$SCRATCH_INDEX" git -C "$BUILD" write-tree
}
work_tree() {   # the tree of the build copy's tracked files as they are on disk, in the scratch index
  GIT_INDEX_FILE="$SCRATCH_INDEX" git -C "$BUILD" read-tree HEAD \
    && GIT_INDEX_FILE="$SCRATCH_INDEX" git -C "$BUILD" add -u \
    && GIT_INDEX_FILE="$SCRATCH_INDEX" git -C "$BUILD" write-tree
}
HEAD_NOW="$(git -C "$BUILD" rev-parse HEAD)"
[[ "$HEAD_NOW" == "$PIN" ]] || die "build copy $BUILD is at $HEAD_NOW, not the pinned $PIN; it holds work that is not this script's. Inspect it, or move it away to rebuild from $SRC"
git -C "$BUILD" diff --cached --quiet || die "build copy $BUILD has staged changes; this script never stages. Inspect them; nothing was reset"
T_WORK="$(work_tree)" || die "cannot read the tracked files of $BUILD"
T_PIN="$(tree_of)" || die "cannot read the pinned tree $PIN"
T_NEW="$(tree_of "${PATCHES[@]}")" || die "the guard patches do not apply to the pinned tree $PIN"
if [[ "$T_WORK" == "$T_NEW" ]]; then
  echo "already applied: ${PATCHES[*]}"
else
  if [[ "$T_WORK" != "$T_PIN" ]]; then
    PREV=()
    if [[ -f "$APPLIED_LIST" ]]; then
      while IFS= read -r name; do [[ -n "$name" ]] && PREV+=("$APPLIED_DIR/$name"); done <"$APPLIED_LIST"
    elif [[ -n "${EMCP_PREVIOUS_PATCHES:-}" ]]; then
      IFS=: read -r -a PREV <<<"$EMCP_PREVIOUS_PATCHES"
    fi
    [[ ${#PREV[@]} -gt 0 ]] || die "build copy $BUILD has changes that are not the guard patches, and no previous patch set is recorded ($APPLIED_LIST) or named (EMCP_PREVIOUS_PATCHES); refusing to reset it. Inspect: git -C '$BUILD' status"
    PREV_REV=()
    for ((i = ${#PREV[@]} - 1; i >= 0; i--)); do
      [[ -f "${PREV[$i]}" ]] || die "missing previous patch ${PREV[$i]}"
      PREV_REV+=("${PREV[$i]}")
    done
    git -C "$BUILD" apply --reverse --check "${PREV_REV[@]}" \
      || die "build copy $BUILD has changes that are not exactly the previous patch set (${PREV[*]}); refusing to reset it"
    T_PREV="$(tree_of "${PREV[@]}")" || die "the previous patch set does not apply to the pinned tree $PIN"
    [[ "$T_WORK" == "$T_PREV" ]] \
      || die "build copy $BUILD has changes beyond the previous patch set (${PREV[*]}); refusing to reset it"
    echo "resetting $BUILD: its tracked changes are exactly the previous patch set"
    git -C "$BUILD" -c advice.detachedHead=false checkout --quiet -f "$PIN"
  fi
  # ---- 2. local guard patches (env-gated; no behaviour change unless the env vars are 1) -----------
  for PATCH in "${PATCHES[@]}"; do
    git -C "$BUILD" apply --check "$PATCH"
    git -C "$BUILD" apply "$PATCH"
  done
fi
# Record the patch set the build copy now holds, for the next run's proof (copies, then the list naming them).
mkdir -p "$APPLIED_DIR"
: >"$APPLIED_LIST.part"
for i in "${!PATCHES[@]}"; do
  name="$(printf '%02d' "$i")-$(basename -- "${PATCHES[$i]}")"
  cp -- "${PATCHES[$i]}" "$APPLIED_DIR/$name"
  printf '%s\n' "$name" >>"$APPLIED_LIST.part"
done
mv -f -- "$APPLIED_LIST.part" "$APPLIED_LIST"

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

# ---- 6 (optional). data dir with the Reforger 1.8 API index from fma-rc ------------------------------
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
