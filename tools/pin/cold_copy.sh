#!/usr/bin/env bash
# cold_copy.sh - copy the pinned Arma Reforger and Arma Reforger Tools installs to E: and write SHA256SUMS.
# Ticket DD-0 (capture program r2 TKT-0, P0 step 3; decision D2b). The copy lets a pinned build be restored
# if Steam updates anyway. It is about 27.9 GB: game about 27.2 GB from D:, Tools about 0.7 GB from C:.
#
#   cold_copy.sh [--run] [--pin PIN_JSON] [--dest DEST_DIR]
#     default PIN_JSON  /mnt/e/DriveDiT/pin/pin_1.8.0.13.json
#     default DEST_DIR  /mnt/e/DriveDiT/pin/<pin_id>    (results: DEST/game, DEST/tools, DEST/SHA256SUMS, DEST/meta)
#
# DRY RUN BY DEFAULT: without --run it only reads, checks the live installs against the pin, and prints the plan.
# The sources are only read. DEST must resolve under /mnt/e; anything on C: is refused.
#
# With --run:
#   1. preflight --check on the live installs. AutoUpdateBehavior is waived here, because the copy is the
#      fallback if it stays "0". Every other check must pass, so a half-updated install is never copied.
#   2. rsync -rt each install into DEST/game and DEST/tools. A marker file, DEST/.cold_copy_incomplete, lets an
#      interrupted run be resumed by running the script again. A DEST that already contains a symlink is refused
#      (rsync -rt never makes one, and one could lead a write off E:).
#   3. preflight --check on the live installs again, to show nothing changed during the copy.
#   4. preflight --check on the copy (--game-root/--tools-root) with 1 s mtime resolution: WSL drvfs truncates
#      the mtimes it sets to whole seconds (measured on E:). Then each copy must hold exactly the regular files
#      of its source: none missing, none extra (such as an rsync temp file left by a killed run).
#   5. SHA256SUMS for every file in the copy, paths relative to DEST (sha256sum -c format).
#   6. The pin JSON and both appmanifests are copied to DEST/meta.
# The script never deletes anything except its own marker file.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFLIGHT="$HERE/preflight.py"
PY=(python3 -I -B)
PIN="/mnt/e/DriveDiT/pin/pin_1.8.0.13.json"
DEST=""
RUN=0
MARGIN=$((2 * 1024 * 1024 * 1024))   # 2 GiB free-space margin

die() { echo "cold_copy: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN=1; shift ;;
    --pin) PIN="${2:?--pin needs a path}"; shift 2 ;;
    --dest) DEST="${2:?--dest needs a path}"; shift 2 ;;
    -h|--help) sed -n '2,/^[^#]/{/^#/p}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -f "$PIN" ]] || die "pin not found: $PIN (run make_pin.py first)"
[[ -f "$PREFLIGHT" ]] || die "preflight.py not found next to this script: $PREFLIGHT"

# Pin fields: pin_id, and the live acf and install paths of both apps.
mapfile -t F < <("${PY[@]}" -c '
import json, sys
p = json.load(open(sys.argv[1]))
print(p["pin_id"])
for app in ("game", "tools"):
    print(p[app]["install"]["wsl"]); print(p[app]["acf"]["wsl"])
' "$PIN")
[[ ${#F[@]} -eq 5 ]] || die "cannot read pin fields from $PIN"
PIN_ID="${F[0]}"; GAME_SRC="${F[1]}"; GAME_ACF="${F[2]}"; TOOLS_SRC="${F[3]}"; TOOLS_ACF="${F[4]}"
[[ "$PIN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "unexpected pin_id: $PIN_ID"
[[ -n "$DEST" ]] || DEST="$(dirname "$PIN")/$PIN_ID"

# Write guard: resolve, then allow only /mnt/e; refuse C: explicitly.
DEST_REAL="$(realpath -m -- "$DEST")"
case "${DEST_REAL,,}/" in
  /mnt/c/*) die "refusing to write on C: ($DEST -> $DEST_REAL)" ;;
esac
case "$DEST_REAL/" in
  /mnt/e/?*) ;;
  *) die "refusing to write outside /mnt/e: $DEST -> $DEST_REAL" ;;
esac
DEST="$DEST_REAL"
for src in "$GAME_SRC" "$TOOLS_SRC"; do
  [[ -d "$src" ]] || die "install dir missing: $src"
  case "$(realpath -m -- "$src")/" in
    "$DEST"/*) die "source $src is inside the destination" ;;
  esac
done

MARKER="$DEST/.cold_copy_incomplete"
STATE="new"
if [[ -e "$DEST/SHA256SUMS" && ! -e "$MARKER" ]]; then
  die "$DEST already holds a finished cold copy (SHA256SUMS present). Verify it with: cd '$DEST' && sha256sum -c --quiet SHA256SUMS"
elif [[ -e "$MARKER" ]]; then
  STATE="resume (marker $MARKER present)"
elif [[ -d "$DEST" && -n "$(ls -A -- "$DEST" 2>/dev/null)" ]]; then
  die "$DEST exists and is not empty, and has no resume marker; refusing to mix copies"
fi
# rsync -rt never creates symlinks. One inside DEST (for example DEST/game -> /mnt/c/...) was put there by
# something else and would lead rsync's writes off E:, so refuse it before anything is written.
if [[ -d "$DEST" ]]; then
  LINK="$(find "$DEST" -type l -print -quit)"
  [[ -z "$LINK" ]] || die "$DEST contains a symlink ($LINK); a copy holds only regular files and directories"
fi

# Exit unless the copy holds exactly the regular files of its source (paths only; content is checked by
# preflight size/mtime and SHA256SUMS). Catches files missing from the copy and extra files in it, such as
# an rsync temp file left behind by a killed run.
same_file_set() {   # $1 label, $2 source dir, $3 copy dir
  local diffs
  diffs="$(LC_ALL=C comm -3 <(cd -- "$2" && find . -type f | LC_ALL=C sort) \
                            <(cd -- "$3" && find . -type f | LC_ALL=C sort))"
  [[ -z "$diffs" ]] || die "$1: file lists differ (marker left in place). Not indented: only in the source;
indented: only in the copy. First 20:
$(printf '%s\n' "$diffs" | head -n 20)"
}

PARENT="$DEST"; while [[ ! -d "$PARENT" ]]; do PARENT="$(dirname "$PARENT")"; done
NEED=$(( $(du -sb -- "$GAME_SRC" | cut -f1) + $(du -sb -- "$TOOLS_SRC" | cut -f1) ))
HAVE=$(df -B1 --output=avail -- "$PARENT" | tail -n 1 | tr -d ' ')

echo "pin        $PIN (pin_id $PIN_ID)"
echo "game  src  $GAME_SRC"
echo "tools src  $TOOLS_SRC"
echo "dest       $DEST  [$STATE]"
awk -v n="$NEED" -v h="$HAVE" -v p="$PARENT" 'BEGIN { printf "size       %.2f GB to copy; %.2f GB free on %s\n", n / 1e9, h / 1e9, p }'
(( HAVE > NEED + MARGIN )) || [[ "$STATE" != "new" ]] || die "not enough free space (need copy size + 2 GiB)"

WAIVE=(--allow-fail game.auto_update_behavior --allow-fail tools.auto_update_behavior)
echo
echo "== live installs vs pin (AutoUpdateBehavior waived for the copy)"
if ! "${PY[@]}" "$PREFLIGHT" --pin "$PIN" --check "${WAIVE[@]}"; then
  die "the live installs do not match the pin; not copying (re-pin or restore first)"
fi

if [[ $RUN -ne 1 ]]; then
  cat <<EOF

DRY RUN: nothing was written. With --run this would execute:
  mkdir -p '$DEST/game' '$DEST/tools' '$DEST/meta' && touch '$MARKER'
  rsync -rt --info=progress2 '$GAME_SRC/' '$DEST/game/'
  rsync -rt --info=progress2 '$TOOLS_SRC/' '$DEST/tools/'
  python3 -I -B '$PREFLIGHT' --pin '$PIN' --check ${WAIVE[*]}            # live installs unchanged
  python3 -I -B '$PREFLIGHT' --pin '$PIN' --check ${WAIVE[*]} \\
      --game-root '$DEST/game' --tools-root '$DEST/tools' --mtime-resolution-s 1   # the copy matches the pin
  same file lists: find game/tools sources vs '$DEST/game' and '$DEST/tools'      # nothing missing or extra
  (cd '$DEST' && find game tools -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > '$DEST/SHA256SUMS'
  cp '$PIN' '$GAME_ACF' '$TOOLS_ACF' '$DEST/meta/' && rm '$MARKER'
EOF
  exit 0
fi

mkdir -p -- "$DEST/game" "$DEST/tools" "$DEST/meta"
touch -- "$MARKER"
echo "== copying game";  rsync -rt --info=progress2 -- "$GAME_SRC/"  "$DEST/game/"
echo "== copying tools"; rsync -rt --info=progress2 -- "$TOOLS_SRC/" "$DEST/tools/"

echo "== live installs vs pin, after the copy"
"${PY[@]}" "$PREFLIGHT" --pin "$PIN" --check "${WAIVE[@]}" \
  || die "the live installs changed during the copy; the copy is NOT trustworthy (marker left in place)"
echo "== copy vs pin"
"${PY[@]}" "$PREFLIGHT" --pin "$PIN" --check "${WAIVE[@]}" \
  --game-root "$DEST/game" --tools-root "$DEST/tools" --mtime-resolution-s 1 \
  || die "the copy does not match the pin (marker left in place)"
echo "== copy file lists vs source"
same_file_set game "$GAME_SRC" "$DEST/game"
same_file_set tools "$TOOLS_SRC" "$DEST/tools"

echo "== SHA256SUMS"
( cd -- "$DEST" && find game tools -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum ) > "$DEST/SHA256SUMS.tmp"
mv -- "$DEST/SHA256SUMS.tmp" "$DEST/SHA256SUMS"
cp -- "$PIN" "$GAME_ACF" "$TOOLS_ACF" "$DEST/meta/"
rm -f -- "$MARKER"
echo "done: $DEST ($(wc -l < "$DEST/SHA256SUMS") files hashed; sha256 of SHA256SUMS $(sha256sum "$DEST/SHA256SUMS" | cut -c1-64))"
