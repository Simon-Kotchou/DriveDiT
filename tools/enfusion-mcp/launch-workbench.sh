#!/usr/bin/env bash
# launch-workbench.sh - start Arma Reforger Workbench from WSL with WINDOWS paths on the command line.
# BK's own wb_launch passes Linux (/mnt/...) paths to the Windows exe, which Workbench cannot open.
# NOT run yet. Do not run while another Workbench instance is open.
# Preferred launcher once DD-0 is deployed: E:\DriveDiT\tools\launch\wb_R1.bat (it runs the build-pin preflight;
# this script does not). Use this one only as the WSL fallback.
#
#   launch-workbench.sh [GPROJ_WSL_PATH] [-- extra Workbench args...]
#   default GPROJ: /mnt/e/DriveDiT/addons/DataCapture/addon.gproj (re-homed capture addon; handlers live inside it)
#   -profile "E:\DriveDiT\wb_profile" is added unless the extra args already contain -profile (WB_PROFILE_WIN overrides).
#   e.g. DataTest: launch-workbench.sh "/mnt/c/Users/Windows/OneDrive/Documents/My Games/ArmaReforgerWorkbench/addons/DataTest/addon.gproj"
#
# Windows-side equivalent (PowerShell), same effect without WSL interop:
#   Start-Process -WorkingDirectory "C:\Program Files (x86)\Steam\steamapps\common\Arma Reforger Tools\Workbench" `
#     "C:\Program Files (x86)\Steam\steamapps\common\Arma Reforger Tools\Workbench\ArmaReforgerWorkbenchSteamDiag.exe" `
#     -ArgumentList '-profile','"E:\DriveDiT\wb_profile"','-gproj','"E:\DriveDiT\addons\DataCapture\addon.gproj"'
set -euo pipefail

WB_DIR="/mnt/c/Program Files (x86)/Steam/steamapps/common/Arma Reforger Tools/Workbench"
WB_EXE="$WB_DIR/ArmaReforgerWorkbenchSteamDiag.exe"
GPROJ="${1:-/mnt/e/DriveDiT/addons/DataCapture/addon.gproj}"
PROFILE_WIN="${WB_PROFILE_WIN:-E:\\DriveDiT\\wb_profile}"
shift || true
[[ "${1:-}" == "--" ]] && shift
[[ -f "$GPROJ" ]] || { echo "no such gproj: $GPROJ" >&2; exit 1; }

if /mnt/c/Windows/System32/tasklist.exe /FI "IMAGENAME eq ArmaReforgerWorkbenchSteamDiag.exe" 2>/dev/null \
     | tr -d '\r' | grep -qi '^ArmaReforgerWorkbenchSteamDiag.exe'; then
  echo "Workbench is already running - close it first (this script never kills it)." >&2
  exit 1
fi

GPROJ_WIN="$(wslpath -w "$GPROJ")"
# CWD = Workbench dir, exactly like the Steam launch (console.log: 'Current working directory:
# C:\Program Files (x86)\Steam\steamapps\common\Arma Reforger Tools\Workbench\'); that dir's
# steam_appid.txt is 1874910 (Tools). BK's launcher uses the game dir instead (steam_appid 1874880).
cd "$WB_DIR"
# Direct interop exec (WSL quotes each argv element for the Windows command line, so paths with
# spaces such as ".../My Games/..." survive; 'cmd /c start' would re-parse them). setsid+nohup
# detach it from this shell; stdio goes to /dev/null so nothing is written on C: or in WSL.
EXTRA=("$@")
if ! printf '%s\n' "${EXTRA[@]}" | grep -qx -- '-profile'; then
  EXTRA=(-profile "$PROFILE_WIN" "${EXTRA[@]}")
fi
echo "launching: $WB_EXE -gproj \"$GPROJ_WIN\" ${EXTRA[*]}" >&2
setsid nohup "$WB_EXE" -gproj "$GPROJ_WIN" "${EXTRA[@]}" </dev/null >/dev/null 2>&1 &
echo "started (pid $!). Wait for 'Module: WorkbenchGame; loaded' in the new console.log, then wb_connect." >&2
