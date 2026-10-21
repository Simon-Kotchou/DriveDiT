#!/usr/bin/env bash
# launch-workbench.sh - start Arma Reforger Workbench from WSL with WINDOWS paths on the command line.
# BK's own wb_launch passes Linux (/mnt/...) paths to the Windows exe, which Workbench cannot open.
# Preferred launcher once DD-0 is deployed: E:\DriveDiT\tools\launch\wb_R1.bat. This script is the WSL
# fallback with the same behaviour: it refuses if Workbench is already running (or that cannot be checked),
# runs the DD-0 build-pin preflight, then starts the exe with the same pinned arguments.
#
#   launch-workbench.sh [GPROJ_WSL_PATH] [-- extra Workbench args...]
#   launch-workbench.sh -- extra Workbench args...          (default GPROJ)
#   default GPROJ: /mnt/e/DriveDiT/addons/DataCapture/addon.gproj (re-homed capture addon; handlers live inside it)
#   Added unless the extra args already carry them: -profile "E:\DriveDiT\wb_profile" (WB_PROFILE_WIN overrides),
#   -forceUpdate, -maxFPS 60 (the wb_R1.bat arguments).
#
# Refuses (exit 1): a GPROJ or -profile on C: or on the WSL disk (Workbench writes .meta files and
# resourceDatabase.rdb into the opened project, and logs/captures into the profile); a running Workbench;
# tasklist.exe not runnable; no active pin; a failed pin preflight. The preflight is
# /mnt/e/DriveDiT/tools/pin/preflight.py --pin-active --pin-dir /mnt/e/DriveDiT/pin --exe wb (WB_PREFLIGHT and
# WB_PIN_DIR override): it checks the pin that /mnt/e/DriveDiT/pin/ACTIVE names, exactly like wb_R1.bat, and writes
# its launch record under <pin dir>/launches. This script names no game version. 1.8.0.13 is the interim pin; the
# plan is to move to 1.9 by activating a new pin with DD-0's pin_activate.py (see DD-0's tools/pin/README.md).
# WB_PIN (an explicit pin file) is no longer accepted, because it would bypass that checked switch.
# The exe this script starts (WB_DIR) is the Tools install in DD-0's pinlib.LAUNCH_PATHS, and the preflight's
# tools.launch_path check refuses any pin not made from that install, so the pin that vouches for a launch is always
# one made from the exe started here. Change WB_DIR only together with LAUNCH_PATHS and wb_R1.bat.
# Like wb_R1.bat, it starts Workbench only if the preflight's stdout is exactly one launch id
# (YYYYMMDDTHHMMSSZ-wb-xxxx) and <pin dir>/launches/launch_<id>.json exists. The preflight runs on
# /usr/bin/python3, the interpreter the .bat launchers use, not whatever python3 PATH holds.
# Until DD-0 is deployed, WB_SKIP_PREFLIGHT=1 launches without the pin check (it says so loudly).
#
# Windows-side equivalent (PowerShell), same arguments, no preflight:
#   Start-Process -WorkingDirectory "C:\Program Files (x86)\Steam\steamapps\common\Arma Reforger Tools\Workbench" `
#     "C:\Program Files (x86)\Steam\steamapps\common\Arma Reforger Tools\Workbench\ArmaReforgerWorkbenchSteamDiag.exe" `
#     -ArgumentList '-profile','"E:\DriveDiT\wb_profile"','-gproj','"E:\DriveDiT\addons\DataCapture\addon.gproj"','-forceUpdate','-maxFPS','60'
set -euo pipefail

die() { echo "launch-workbench: $*" >&2; exit 1; }
off_c_drive() {   # absolute WSL path that resolves onto a Windows drive other than C:
  local r="" re='^/mnt/[abdefghijklmnopqrstuvwxyz](/|$)'
  [[ "${1:-}" == /* ]] || return 1
  r="$(realpath -m -- "$1" 2>/dev/null)" || return 1
  [[ "$r" =~ $re ]]
}
win_off_c_drive() {   # Windows drive path X:\... with X != C
  local d="${1:0:1}"
  [[ "${1:1:2}" == ':\' && "$d" == [[:alpha:]] && "${d^^}" != "C" ]]
}

WB_DIR="/mnt/c/Program Files (x86)/Steam/steamapps/common/Arma Reforger Tools/Workbench"
WB_EXE="$WB_DIR/ArmaReforgerWorkbenchSteamDiag.exe"
WB_IMAGE="ArmaReforgerWorkbenchSteamDiag.exe"
TASKLIST=/mnt/c/Windows/System32/tasklist.exe
PY=/usr/bin/python3
PREFLIGHT="${WB_PREFLIGHT:-/mnt/e/DriveDiT/tools/pin/preflight.py}"
PIN_DIR="${WB_PIN_DIR:-/mnt/e/DriveDiT/pin}"
[[ -z "${WB_PIN:-}" ]] || die "WB_PIN is no longer supported: the launch checks the active pin ($PIN_DIR/ACTIVE). Switch pins with pin_activate.py; unset WB_PIN."
PROFILE_WIN="${WB_PROFILE_WIN:-E:\\DriveDiT\\wb_profile}"

GPROJ=/mnt/e/DriveDiT/addons/DataCapture/addon.gproj
if [[ $# -gt 0 && "$1" != "--" ]]; then GPROJ="$1"; shift; fi
[[ "${1:-}" == "--" ]] && shift
EXTRA=("$@")

[[ -f "$GPROJ" ]] || die "no such gproj: $GPROJ"
off_c_drive "$GPROJ" || die "refusing gproj $GPROJ: Workbench writes into the opened project; it must be an absolute path off C: (E:)"
[[ -f "$WB_EXE" ]] || die "Workbench exe not found: $WB_EXE"

# -profile: ours unless the extra args carry one; either way it must be a Windows path off C:.
has_arg() { local a; for a in "${EXTRA[@]}"; do [[ "$a" == "$1" ]] && return 0; done; return 1; }
PROFILE="$PROFILE_WIN"
for i in "${!EXTRA[@]}"; do [[ "${EXTRA[$i]}" == "-profile" ]] && PROFILE="${EXTRA[$((i + 1))]:-}"; done
win_off_c_drive "$PROFILE" || die "refusing -profile '$PROFILE': Workbench writes logs and captures there; use a Windows path off C: (E:\\...)"
ARGS=()
has_arg -profile || ARGS+=(-profile "$PROFILE_WIN")
ARGS+=(-gproj "$(wslpath -w "$GPROJ")")
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

# Build-pin preflight (DD-0 / capture program TKT-0): refuses on any drift from the ACTIVE pin.
if [[ "${WB_SKIP_PREFLIGHT:-0}" == "1" ]]; then
  echo "launch-workbench: WARNING: WB_SKIP_PREFLIGHT=1 - the build pin is NOT checked for this launch" >&2
else
  [[ -f "$PREFLIGHT" ]] \
    || die "pin preflight not deployed ($PREFLIGHT). Deploy DD-0, use wb_R1.bat, or set WB_SKIP_PREFLIGHT=1 knowingly."
  off_c_drive "$PIN_DIR" \
    || die "refusing pin dir $PIN_DIR: the preflight writes its launch record under it; use an absolute path off C: (E:)"
  [[ -e "$PIN_DIR/ACTIVE" || -L "$PIN_DIR/ACTIVE" ]] \
    || die "no active pin ($PIN_DIR/ACTIVE is missing). Activate one with: $PY -I -B $(dirname "$PREFLIGHT")/pin_activate.py <pin id>"
  [[ -x "$PY" ]] || die "cannot run $PY, so the pin cannot be checked; Workbench not started."
  LID="$(cd /mnt/e && "$PY" -I -B "$PREFLIGHT" --pin-active --pin-dir "$PIN_DIR" --exe wb)" \
    || die "PIN CHECK FAILED; Workbench not started (preflight messages above)."
  [[ -n "$LID" ]] || die "PIN CHECK FAILED: the preflight printed no launch id; Workbench not started."
  [[ "$LID" =~ ^[0-9]{8}T[0-9]{6}Z-wb-[0-9a-f]{4}$ ]] \
    || die "PIN CHECK FAILED: the preflight output is not exactly one launch id; Workbench not started."
  REC="$PIN_DIR/launches/launch_$LID.json"
  [[ -f "$REC" && ! -L "$REC" ]] \
    || die "PIN CHECK FAILED: no launch record $REC; a launch id alone is not proof the check passed. Workbench not started."
  echo "pin OK, launch id $LID" >&2
fi

# CWD = Workbench dir, exactly like the Steam launch (console.log: 'Current working directory:
# C:\Program Files (x86)\Steam\steamapps\common\Arma Reforger Tools\Workbench\'); that dir's
# steam_appid.txt is 1874910 (Tools). BK's launcher uses the game dir instead (steam_appid 1874880).
cd "$WB_DIR"
# Direct interop exec (WSL quotes each argv element for the Windows command line, so paths with
# spaces survive; 'cmd /c start' would re-parse them). setsid+nohup detach it from this shell;
# stdio goes to /dev/null so nothing is written on C: or in WSL (no nohup.out).
echo "launching: $WB_EXE ${ARGS[*]}" >&2
setsid nohup "$WB_EXE" "${ARGS[@]}" </dev/null >/dev/null 2>&1 &
echo "started (pid $!). Check E:\\DriveDiT\\wb_profile\\logs for 'Module: WorkbenchGame; loaded', then wb_connect." >&2
