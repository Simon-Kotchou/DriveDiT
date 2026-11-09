#!/usr/bin/env bash
# launch-workbench.sh - start Arma Reforger Workbench from WSL with WINDOWS paths on the command line.
# BK's own wb_launch passes Linux (/mnt/...) paths to the Windows exe, which Workbench cannot open.
# Preferred launcher once DD-0 is deployed: <DATA_ROOT>\tools\launch\wb_R1.bat. This script is the WSL
# fallback with the same behaviour: it refuses if Workbench is already running (or that cannot be checked),
# runs the DD-0 build-pin preflight, then starts the exe with the same pinned arguments.
#
#   launch-workbench.sh [GPROJ_WSL_PATH] [-- extra Workbench args...]
#   launch-workbench.sh -- extra Workbench args...          (default GPROJ)
#   default GPROJ: <DATA_ROOT>/addons/DataCapture/addon.gproj (re-homed capture addon; handlers live inside it)
#   Added unless the extra args already carry them: -profile "<DATA_ROOT>\wb_profile" (WB_PROFILE_WIN overrides),
#   -forceUpdate, -maxFPS 60 (the wb_R1.bat arguments).
# DD-22: <DATA_ROOT>, the installs (GAME_INSTALL, TOOLS_INSTALL) and the system drive (SYSTEM_DRIVE, for tasklist.exe
# only) come from the local config (tools/local_config.py), the file wb_R1.bat reads too; without it nothing starts.
#
# Refuses (exit 1): a GPROJ, -profile or pin dir the write boundary refuses (tools/local_config.py check-write: only
# the verified data drive or RAM; never the system drive or the WSL disk. Workbench writes .meta files and
# resourceDatabase.rdb into the opened project, and logs/captures into the profile); a running Workbench;
# tasklist.exe not runnable; no active pin; a failed pin preflight. The preflight is
# <DATA_ROOT>/tools/pin/preflight.py --pin-active --pin-dir <DATA_ROOT>/pin --exe wb (WB_PREFLIGHT and
# WB_PIN_DIR override): it checks the pin that <DATA_ROOT>/pin/ACTIVE names, exactly like wb_R1.bat, and writes
# its launch record under <pin dir>/launches. This script names no game version. 1.8.0.13 is the interim pin; the
# plan is to move to 1.9 by activating a new pin with DD-0's pin_activate.py (see DD-0's tools/pin/README.md).
# WB_PIN (an explicit pin file) is no longer accepted, because it would bypass that checked switch.
# The exe this script starts (WB_DIR) is the Tools install in DD-0's pinlib.LAUNCH_PATHS, and the preflight's
# tools.launch_path check refuses any pin not made from that install, so the pin that vouches for a launch is always
# one made from the exe started here. Both read TOOLS_INSTALL of the local config, as wb_R1.bat does.
# Like wb_R1.bat, it starts Workbench only if the preflight's stdout is exactly one launch id
# (YYYYMMDDTHHMMSSZ-wb-xxxx) and <pin dir>/launches/launch_<id>.json exists. The preflight runs on
# /usr/bin/python3, the interpreter the .bat launchers use, not whatever python3 PATH holds.
# There is no launch without the preflight and its launch record (the WB_SKIP_PREFLIGHT bypass is gone, DD-22).
#
# Windows-side equivalent (PowerShell), same arguments, no preflight (values from the local config):
#   Start-Process -WorkingDirectory "<TOOLS_INSTALL>\Workbench" `
#     "<TOOLS_INSTALL>\Workbench\ArmaReforgerWorkbenchSteamDiag.exe" `
#     -ArgumentList '-profile','"<DATA_ROOT>\wb_profile"','-gproj','"<DATA_ROOT>\addons\DataCapture\addon.gproj"','-forceUpdate','-maxFPS','60'
set -euo pipefail

NAME=launch-workbench
die() { echo "launch-workbench: $*" >&2; exit 1; }
# DD-22: this machine's paths come from the untracked local config (tools/local_config.py, drivedit.local.env).
# An installed MCP root holds a copy of local_config.py next to this script; tell it the file with
# DRIVEDIT_LOCAL_CONFIG. In the repo the loader is ../local_config.py.
HERE="$(dirname "$(realpath -- "${BASH_SOURCE[0]}")")"
LOCAL_CONFIG_PY=""
for _c in "$HERE/local_config.py" "$HERE/../local_config.py"; do
  if [[ -f "$_c" ]]; then LOCAL_CONFIG_PY="$_c"; break; fi
done
local_config() {   # local_config KEY [--win | --drive]: one value; the loader names the file and key on a refusal
  [[ -n "$LOCAL_CONFIG_PY" ]] || { echo "launch-workbench: local_config.py is missing next to this script and in ../" >&2; return 2; }
  /usr/bin/python3 -I -B "$LOCAL_CONFIG_PY" get "$@"
}
DATA_ROOT="$(local_config DATA_ROOT)" || die "no local config (see the message above); Workbench not started"
DATA_ROOT_WIN="$(local_config DATA_ROOT --win)" || die "no local config (see the message above)"
DATA_DRIVE="$(local_config DATA_ROOT --drive)" || die "no local config (see the message above)"
TOOLS_INSTALL="$(local_config TOOLS_INSTALL)" || die "no TOOLS_INSTALL in the local config (see above)"
GAME_INSTALL="$(local_config GAME_INSTALL)" || die "no GAME_INSTALL in the local config (see above)"
SYSTEM_DRIVE="$(local_config SYSTEM_DRIVE)" || die "no SYSTEM_DRIVE in the local config (see above)"   # tasklist.exe only
# Every path Workbench or the preflight writes under goes through the write boundary.
write_ok() {   # write_ok PATH: print the resolved path when the write boundary allows writing there, else fail
  # (tools/local_config.py check-write: RAM or the verified data drive, never a volume whose root holds
  # Windows/System32, whatever the config says). A relative path is refused: it would resolve against the cwd.
  [[ -n "$LOCAL_CONFIG_PY" ]] || { echo "$NAME: local_config.py is missing next to this script and in ../" >&2; return 2; }
  [[ "${1:-}" == /* ]] || { echo "$NAME: '${1:-}' is not an absolute path" >&2; return 1; }
  /usr/bin/python3 -I -B "$LOCAL_CONFIG_PY" check-write "$1"
}
derived_dir() {   # derived_dir DIR ROOT: DIR, to be created or reused below the checked ROOT, is safe to write
  # into: it lies below ROOT, no component from DIR up to ROOT is a symlink, and every existing one is a directory on
  # ROOT's device (stat -c %d), so a redirected or foreign-mounted home, temp, cache or build dir is refused before
  # anything is written there. A DIR that does not exist yet passes when its parent does (DD-22).
  local d="${1:-}" root="${2%/}" root_dev="" dev="" p=""
  [[ -n "$root" && "$d" == "$root"/* && "$d/" != */../* && "$d/" != */./* ]] \
    || { echo "$NAME: refusing $d: it is not below ${2:-} (or holds . or ..)" >&2; return 1; }
  root_dev="$(stat -c %d -- "$root" 2>/dev/null)" || { echo "$NAME: refusing $d: cannot stat $root" >&2; return 1; }
  p="$d"
  while [[ "$p" != "$root" ]]; do
    [[ "$p" != / && "$p" != . ]] || { echo "$NAME: refusing $d: it is not below $root" >&2; return 1; }
    if [[ -L "$p" ]]; then echo "$NAME: refusing $d: $p is a symlink" >&2; return 1; fi
    if [[ -e "$p" ]]; then
      [[ -d "$p" ]] || { echo "$NAME: refusing $d: $p is not a directory" >&2; return 1; }
      dev="$(stat -c %d -- "$p" 2>/dev/null)" || { echo "$NAME: refusing $d: cannot stat $p" >&2; return 1; }
      [[ "$dev" == "$root_dev" ]] || { echo "$NAME: refusing $d: $p is on device $dev, not on $root's ($root_dev)" >&2; return 1; }
    fi
    p="$(dirname -- "$p")"
  done
}
win_write_ok() {   # a Windows drive path X:\... whose WSL form /mnt/x/... passes write_ok
  [[ "${1:1:2}" == ':\' && "${1:0:1}" == [[:alpha:]] ]] || return 1
  local drive="${1:0:1}" rest="${1:3}"
  write_ok "/mnt/${drive,,}/${rest//\\//}" >/dev/null
}

WB_DIR="$TOOLS_INSTALL/Workbench"
WB_EXE="$WB_DIR/ArmaReforgerWorkbenchSteamDiag.exe"
WB_IMAGE="ArmaReforgerWorkbenchSteamDiag.exe"
TASKLIST="$SYSTEM_DRIVE/Windows/System32/tasklist.exe"
PY=/usr/bin/python3
PREFLIGHT="${WB_PREFLIGHT:-$DATA_ROOT/tools/pin/preflight.py}"
PIN_DIR="${WB_PIN_DIR:-$DATA_ROOT/pin}"
[[ -z "${WB_PIN:-}" ]] || die "WB_PIN is no longer supported: the launch checks the active pin ($PIN_DIR/ACTIVE). Switch pins with pin_activate.py; unset WB_PIN."
PROFILE_WIN="${WB_PROFILE_WIN:-$DATA_ROOT_WIN\\wb_profile}"
# A direct exe launch does not get the game install's addons dir the way a Steam launch does, so
# ArmaReforger.gproj is "not found" and Workbench exits (measured 2026-10-21). Same value as wb_R1.bat
# GAME_ADDONS: the pinned game install (DD-0 pinlib.LAUNCH_PATHS "game") plus /addons.
GAME_ADDONS="$GAME_INSTALL/addons"

GPROJ="$DATA_ROOT/addons/DataCapture/addon.gproj"
if [[ $# -gt 0 && "$1" != "--" ]]; then GPROJ="$1"; shift; fi
[[ "${1:-}" == "--" ]] && shift
EXTRA=("$@")

[[ -f "$GPROJ" ]] || die "no such gproj: $GPROJ"
write_ok "$GPROJ" >/dev/null || die "refusing gproj $GPROJ: Workbench writes into the opened project, and the write boundary refuses it (see above)"
[[ -f "$WB_EXE" ]] || die "Workbench exe not found: $WB_EXE"
[[ -f "$GAME_ADDONS/data/ArmaReforger.gproj" ]] || die "game addons not found: $GAME_ADDONS/data/ArmaReforger.gproj"

# -profile: ours unless the extra args carry one; either way it must be a Windows path off the system drive.
has_arg() { local a; for a in "${EXTRA[@]}"; do [[ "$a" == "$1" ]] && return 0; done; return 1; }
PROFILE="$PROFILE_WIN"
for i in "${!EXTRA[@]}"; do [[ "${EXTRA[$i]}" == "-profile" ]] && PROFILE="${EXTRA[$((i + 1))]:-}"; done
win_write_ok "$PROFILE" || die "refusing -profile '$PROFILE': Workbench writes logs and captures there; it must be a Windows drive path the write boundary allows"
ARGS=()
has_arg -profile || ARGS+=(-profile "$PROFILE_WIN")
ARGS+=(-gproj "$(wslpath -w "$GPROJ")")
has_arg -addonsDir || ARGS+=(-addonsDir "$(wslpath -w "$GAME_ADDONS")")
has_arg -forceUpdate || ARGS+=(-forceUpdate)
has_arg -maxFPS || ARGS+=(-maxFPS 60)
ARGS+=("${EXTRA[@]}")

# Never a second instance. Fail closed: if tasklist cannot run, we cannot prove Workbench is closed.
# CSV output: the default table format truncates the image name to 25 characters
# ("ArmaReforgerWorkbenchStea"), so matching the full name there never succeeds.
[[ -x "$TASKLIST" ]] || die "cannot run $TASKLIST (WSL interop off?), so cannot check for a running Workbench"
TL="$("$TASKLIST" /FI "IMAGENAME eq $WB_IMAGE" /FO CSV /NH 2>/dev/null | tr -d '\r')" \
  || die "tasklist.exe failed, so cannot check for a running Workbench"
if grep -qiF "\"$WB_IMAGE\"" <<<"$TL"; then
  die "Workbench is already running - close it first (this script never kills it)."
fi

# Build-pin preflight (DD-0 / capture program TKT-0): refuses on any drift from the ACTIVE pin. Every launch runs it
# and needs its launch record; there is no way around it.
[[ -f "$PREFLIGHT" ]] || die "pin preflight not deployed ($PREFLIGHT). Deploy DD-0 or use wb_R1.bat; Workbench not started."
write_ok "$PIN_DIR" >/dev/null \
  || die "refusing pin dir $PIN_DIR: the preflight writes its launch record under it, and the write boundary refuses it"
derived_dir "$PIN_DIR/launches" "$PIN_DIR" || die "refusing $PIN_DIR/launches, where the preflight writes its launch record"
[[ -e "$PIN_DIR/ACTIVE" || -L "$PIN_DIR/ACTIVE" ]] \
  || die "no active pin ($PIN_DIR/ACTIVE is missing). Activate one with: $PY -I -B $(dirname "$PREFLIGHT")/pin_activate.py <pin id>"
[[ -x "$PY" ]] || die "cannot run $PY, so the pin cannot be checked; Workbench not started."
LID="$(cd "$DATA_DRIVE" && DRIVEDIT_LOCAL_CONFIG="$(/usr/bin/python3 -I -B "$LOCAL_CONFIG_PY" where)" \
       "$PY" -I -B "$PREFLIGHT" --pin-active --pin-dir "$PIN_DIR" --exe wb)" \
  || die "PIN CHECK FAILED; Workbench not started (preflight messages above)."
[[ -n "$LID" ]] || die "PIN CHECK FAILED: the preflight printed no launch id; Workbench not started."
[[ "$LID" =~ ^[0-9]{8}T[0-9]{6}Z-wb-[0-9a-f]{4}$ ]] \
  || die "PIN CHECK FAILED: the preflight output is not exactly one launch id; Workbench not started."
REC="$PIN_DIR/launches/launch_$LID.json"
[[ -f "$REC" && ! -L "$REC" ]] \
  || die "PIN CHECK FAILED: no launch record $REC; a launch id alone is not proof the check passed. Workbench not started."
echo "pin OK, launch id $LID" >&2

# CWD = Workbench dir, exactly like the Steam launch (console.log: 'Current working directory:
# <TOOLS_INSTALL>\Workbench\'); that dir's
# steam_appid.txt is 1874910 (Tools). BK's launcher uses the game dir instead (steam_appid 1874880).
cd "$WB_DIR"
# Direct interop exec (WSL quotes each argv element for the Windows command line, so paths with
# spaces survive; 'cmd /c start' would re-parse them). setsid+nohup detach it from this shell;
# stdio goes to /dev/null so nothing is written on the system drive or in WSL (no nohup.out).
echo "launching: $WB_EXE ${ARGS[*]}" >&2
setsid nohup "$WB_EXE" "${ARGS[@]}" </dev/null >/dev/null 2>&1 &
echo "started (pid $!). Check $PROFILE\\logs for 'Module: WorkbenchGame; loaded', then wb_connect." >&2
