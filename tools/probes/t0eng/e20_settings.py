#!/usr/bin/env python3
"""E-20 render-settings dump: analysis of one e20 play (DC_ProbeEngine, DD-4-probe).

    python3 -I -B tools/probes/t0eng/e20_settings.py <e20 play_dir> [--compare <other e20 play_dir>]
                                                     [--profile-dir <profile dir>]

r2 4.4 criterion (E-20): GetEngineUserSettings() modules VideoUserSettings and DisplayUserSettings dumped in WB and
client; the engine settings file located in each -profile; WB and client differences listed. Checked here:
  engine_modules_dumped   every engine module of params.m_sE20EngineModules has at least one variable row
  run_footer              the probe ended normally (status complete)
INFO: the pinned keys of r2 3.4 (ResolutionScale, Fsaa, MaxFps, Vsync, OverallQuality, and any motion-blur or
depth-of-field key), the game modules, the display facts of e20_display.txt.
--compare lists every (root, module, variable) whose value differs between the two dumps, or exists in only one
(WB against client, the input for DD-9-min's engine_R1.conf).
--profile-dir lists, read-only, the files below a profile folder whose content mentions VideoUserSettings (where
the engine keeps its settings file); files over 64 MB are skipped.

Exit codes: 0 PASS, 1 FAIL, 3 INCOMPLETE, 2 usage or input error.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import probe_common as pc  # noqa: E402

LABEL = "E-20 render settings"
COLUMNS = ("root", "module", "var_index", "var_name", "var_type", "value")
PINNED = ("ResolutionScale", "Fsaa", "MaxFps", "Vsync", "OverallQuality")
PINNED_PATTERNS = ("blur", "dof", "depthoffield")
DEFAULT_ENGINE_MODULES = "VideoUserSettings,DisplayUserSettings"
PROFILE_MARKER = b"VideoUserSettings"
PROFILE_MAX_BYTES = 64 * 1024 * 1024


def load_dump(play):
    """{(root, module, var_name): (var_type, value)} and the rows"""
    rows = play.csv("e20_settings", COLUMNS)
    dump = {}
    for r in rows:
        if pc.to_int(r["var_index"], -1) < 0:
            continue
        dump[(r["root"], r["module"], r["var_name"])] = (r["var_type"], r["value"])
    return dump, rows


def is_pinned(name):
    low = name.lower()
    return name in PINNED or any(p in low for p in PINNED_PATTERNS)


def analyse(play, compare=None, profile_dir=None):
    v = pc.Verdict(LABEL, play.dir)
    engine_modules = [m.strip() for m in play.param("m_sE20EngineModules", DEFAULT_ENGINE_MODULES).split(",")
                      if m.strip()]
    if not play.has("e20_settings.csv"):
        v.check("engine_modules_dumped", None, "e20_settings.csv missing")
        dump, rows = {}, []
    else:
        dump, rows = load_dump(play)
        counts = {m: sum(1 for key in dump if key[0] == "engine" and key[1] == m) for m in engine_modules}
        empty = [m for m, n in counts.items() if n == 0]
        detail = ", ".join("%s %d variables" % (m, n) for m, n in counts.items())
        v.check("engine_modules_dumped", bool(counts) and not empty, detail or "no engine module configured")
        game = sorted({key[1] for key in dump if key[0] == "game"})
        missing = sorted({(r["root"], r["module"]) for r in rows if r["var_type"] == "missing_module"})
        if game:
            v.info("game_modules", ", ".join("%s %d variables" % (
                m, sum(1 for key in dump if key[:2] == ("game", m))) for m in game))
        if missing:
            v.info("missing_modules", ", ".join("%s/%s" % m for m in missing))
        pinned = [(key, val) for key, val in sorted(dump.items()) if is_pinned(key[2])]
        for (root, module, name), (vtype, value) in pinned:
            v.info("pinned %s.%s" % (module, name), "%s (%s)" % (value, vtype))

    if play.has("e20_display.txt"):
        display = pc.read_kv(play.path("e20_display.txt"))
        keys = [k for k in display if not k.startswith("module.")]
        v.info("display", ", ".join("%s=%s" % (k, display[k]) for k in keys))
    v.info("exe", play.header.get("engine.exe", "?"))

    if compare:
        other = pc.Play(compare)
        other_dump, _ = load_dump(other)
        only_a = sorted(set(dump) - set(other_dump))
        only_b = sorted(set(other_dump) - set(dump))
        differ = sorted(k for k in set(dump) & set(other_dump) if dump[k][1] != other_dump[k][1])
        label_a = play.header.get("engine.exe", "A")
        label_b = other.header.get("engine.exe", "B")
        v.info("compare", "%s vs %s (%s): %d different, %d only in %s, %d only in %s" % (
            label_a, label_b, other.dir, len(differ), len(only_a), label_a, len(only_b), label_b))
        for key in differ:
            v.info("differs %s/%s.%s" % key, "%s=%s %s=%s" % (label_a, dump[key][1], label_b, other_dump[key][1]))
        for key in only_a:
            v.info("only_in_%s %s/%s.%s" % ((label_a,) + key), dump[key][1])
        for key in only_b:
            v.info("only_in_%s %s/%s.%s" % ((label_b,) + key), other_dump[key][1])

    if profile_dir:
        hits = find_settings_files(profile_dir)
        v.info("settings_files", "%d file(s) under %s mention VideoUserSettings%s" % (
            len(hits), profile_dir, (": " + ", ".join(hits)) if hits else ""))
    v.check_footer(play)
    return v


def find_settings_files(profile_dir):
    """Relative paths of the files below profile_dir whose bytes contain VideoUserSettings (read-only)."""
    if not os.path.isdir(profile_dir):
        raise pc.InputError("not a directory: %s" % profile_dir)
    hits = []
    for dirpath, dirnames, filenames in os.walk(profile_dir):
        dirnames.sort()
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            try:
                if os.path.islink(path) or os.path.getsize(path) > PROFILE_MAX_BYTES:
                    continue
                with open(path, "rb") as f:
                    if PROFILE_MARKER in f.read():
                        hits.append(os.path.relpath(path, profile_dir))
            except OSError:
                continue
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description="E-20 render-settings dump (r2 4.4)")
    ap.add_argument("play_dir", help="an e20 play directory ($profile/DC_Probes/e20/<launch_id>/play_NN)")
    ap.add_argument("--compare", metavar="PLAY_DIR", help="another e20 play (for example the client's) to diff against")
    ap.add_argument("--profile-dir", metavar="DIR", help="profile folder to search for the settings file (read-only)")
    args = ap.parse_args(argv)
    return analyse(pc.Play(args.play_dir), args.compare, args.profile_dir).report()


if __name__ == "__main__":
    sys.exit(pc.run_cli(main))
