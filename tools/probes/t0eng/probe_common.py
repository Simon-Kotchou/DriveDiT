#!/usr/bin/env python3
"""Shared reader and verdict printer of the T0-ENG probe analysis scripts (DD-4-probe).

Not a script itself. Every analysis script under tools/probes/t0eng/ (and the ones in tools/validator/ and
tools/roads/ that read probe output) imports it after putting this folder on sys.path, because under `python3 -I`
the script's own directory is not on it:

    HERE = os.path.dirname(os.path.realpath(__file__)); sys.path.insert(0, HERE)
    import probe_common

What a probe writes (DC_ProbeRunner, PROBE_DESIGN.md 4.5): $profile:DC_Probes/<probe>/<launch_id>/play_NN/ with
run_header.txt (key=value, written before the probe starts), <name>.csv files (one header row, ',' separator, LF,
no quoting, empty field = missing value) and run_footer.txt (key=value, written at the stop; absent after a crash, a
kill or a Stop that tore the world down without a footer).

Printed format of every analysis (Verdict.report):

    E-10 tick counting: <play_dir>
      [PASS] steps_per_world_second: 60 +-1 in 598/598 full seconds (dt 0.016667 s)
      [FAIL] telemetry_every_k: k=3, 2 missed of 11960
      [INFO] step_source: simulate
    RESULT E-10: FAIL

Exit codes: 0 PASS, 1 FAIL, 3 INCOMPLETE, 2 usage or input error (argparse, InputError). A play without
run_footer.txt is analysed as far as possible and its result is at best INCOMPLETE.

Storage rule of this machine: the C: drive is full. Optional outputs go only under /mnt/e/ or /dev/shm/
(check_out_path), synthetic test plays only there too (write_synthetic_play, make_temp_dir), never /tmp (the WSL
root is on C:). Stdlib only; numpy is never required here.
"""

import csv
import os
import re
import tempfile
import unittest

PASS, FAIL, INCOMPLETE, INFO = "PASS", "FAIL", "INCOMPLETE", "INFO"
EXIT_CODES = {PASS: 0, FAIL: 1, INCOMPLETE: 3}
EXIT_INPUT_ERROR = 2

RUN_HEADER = "run_header.txt"
RUN_FOOTER = "run_footer.txt"
PLAY_DIR_RE = re.compile(r"^play_\d\d$")

# Roots where this tool family may write (outputs, synthetic plays, temporary directories)
WRITE_ROOTS = ("/mnt/e/", "/dev/shm/")
FORBIDDEN_ROOTS = ("/mnt/c/", "/tmp/")


class InputError(Exception):
    """A missing play file or column: the script exits with 2."""


# ------------------------------------------------------------------------------------------------
# Readers
# ------------------------------------------------------------------------------------------------
def read_kv(path):
    """key=value file -> dict. Later keys win (DC_ProbeContext.AppendHeaderLines appends to run_header.txt).
    Lines without '=' are ignored. Raises InputError when the file is absent."""
    if not os.path.isfile(path):
        raise InputError("missing file %s" % path)
    out = {}
    with open(path, "r", encoding="ascii", errors="replace", newline="") as f:
        for line in f.read().split("\n"):
            line = line.rstrip("\r")
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key.strip()] = value
    return out


def read_csv(path, required=()):
    """csv.DictReader rows (list of dicts). InputError when the file or a required column is missing."""
    if not os.path.isfile(path):
        raise InputError("missing file %s" % path)
    with open(path, "r", encoding="ascii", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        missing = [c for c in required if c not in columns]
        if missing:
            raise InputError("%s lacks column(s) %s" % (path, ", ".join(missing)))
        return list(reader)


def _number(value, conv):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return conv(value)
    except ValueError:
        return None


def floats(rows, name):
    """Column name as floats; an empty or unparsable field -> None."""
    return [_number(r.get(name), float) for r in rows]


def ints(rows, name):
    """Column name as ints; an empty or unparsable field -> None ("3.0" is accepted as 3)."""
    def conv(text):
        try:
            return int(text)
        except ValueError:
            return int(float(text))
    return [_number(r.get(name), conv) for r in rows]


def to_float(value, default=None):
    """One key=value or CSV value as float, default when empty or unparsable."""
    result = _number(value, float) if isinstance(value, str) else None
    return default if result is None else result


def to_int(value, default=None):
    """One key=value or CSV value as int, default when empty or unparsable."""
    if not isinstance(value, str):
        return default
    result = _number(value, lambda t: int(float(t)))
    return default if result is None else result


def percentile(values, q):
    """Linear interpolation between closest ranks (numpy's default), q in [0, 100]. None for an empty list."""
    data = sorted(values)
    if not data:
        return None
    if len(data) == 1:
        return float(data[0])
    pos = (len(data) - 1) * (float(q) / 100.0)
    low = int(pos)
    high = min(low + 1, len(data) - 1)
    frac = pos - low
    return float(data[low]) + (float(data[high]) - float(data[low])) * frac


class Play:
    """One play_NN directory: run_header.txt (required), run_footer.txt (optional) and the probe's files."""

    def __init__(self, play_dir):
        self.dir = os.path.realpath(play_dir)
        if not os.path.isdir(self.dir):
            raise InputError("not a directory: %s" % play_dir)
        self.header = read_kv(os.path.join(self.dir, RUN_HEADER))
        footer_path = os.path.join(self.dir, RUN_FOOTER)
        self.footer = read_kv(footer_path) if os.path.isfile(footer_path) else None
        self.complete = bool(self.footer) and self.footer.get("status") == "complete"
        self.probe = self.header.get("probe", "")

    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    def has(self, name):
        return os.path.isfile(self.path(name))

    def csv(self, name, required=()):
        """<play_dir>/<name>.csv as DictReader rows; InputError when missing."""
        return read_csv(self.path(name + ".csv"), required)

    def param(self, key, default=None):
        """params.<key> of the run header (probe attributes and derived values)."""
        return self.header.get("params." + key, default)


def find_plays(probe_root, probe=None):
    """Sorted play_NN dirs below <probe_root> (a DC_Probes folder): <probe>/<launch_id>/play_NN, every probe when
    probe is None."""
    probes = [probe] if probe else sorted(os.listdir(probe_root)) if os.path.isdir(probe_root) else []
    out = []
    for name in probes:
        probe_dir = os.path.join(probe_root, name)
        if not os.path.isdir(probe_dir):
            continue
        for launch in sorted(os.listdir(probe_dir)):
            launch_dir = os.path.join(probe_dir, launch)
            if not os.path.isdir(launch_dir):
                continue
            for play in sorted(os.listdir(launch_dir)):
                if PLAY_DIR_RE.match(play) and os.path.isdir(os.path.join(launch_dir, play)):
                    out.append(os.path.join(launch_dir, play))
    return sorted(out)


# ------------------------------------------------------------------------------------------------
# Verdict
# ------------------------------------------------------------------------------------------------
class Verdict:
    """Collects PASS / FAIL / INCOMPLETE checks and INFO lines, prints them in the common format."""

    def __init__(self, label, target=None):
        self.label = label                  # e.g. "E-10 tick counting"
        self.target = target                # play dir shown in the first line
        self.items = []                     # (tag, name, detail)

    def check(self, name, ok, detail):
        """ok True -> PASS, False -> FAIL, None -> INCOMPLETE."""
        tag = {True: PASS, False: FAIL, None: INCOMPLETE}[ok if ok is None else bool(ok)]
        self.items.append((tag, name, detail))
        return ok

    def info(self, name, detail):
        self.items.append((INFO, name, detail))

    def check_footer(self, play):
        """The play ended normally (footer status complete). failed -> FAIL; stopped or no footer -> INCOMPLETE."""
        if play.footer is None:
            return self.check("run_footer", None, "no run_footer.txt (crash, kill or Stop before the probe ended)")
        status = play.footer.get("status", "")
        detail = "status=%s reason=%s" % (status, play.footer.get("reason", ""))
        note = play.footer.get("note", "").strip()
        if note:
            detail += " note=%s" % note
        if status == "complete":
            return self.check("run_footer", True, detail)
        if status == "failed":
            return self.check("run_footer", False, detail)
        return self.check("run_footer", None, detail)

    def result(self):
        tags = [t for t, _, _ in self.items]
        if FAIL in tags:
            return FAIL
        if INCOMPLETE in tags:
            return INCOMPLETE
        if PASS not in tags:
            return INCOMPLETE
        return PASS

    def short_label(self):
        return self.label.split()[0] if self.label.split() else self.label

    def report(self, target=None):
        """Prints the block and returns the exit code of the result."""
        target = target or self.target
        print("%s: %s" % (self.label, target) if target else self.label)
        for tag, name, detail in self.items:
            print("  [%s] %s: %s" % (tag, name, detail))
        result = self.result()
        print("RESULT %s: %s" % (self.short_label(), result))
        return EXIT_CODES[result]


def run_cli(main, argv=None):
    """Runs main(argv) -> exit code; an InputError prints one line and returns 2."""
    try:
        return main(argv)
    except InputError as e:
        print("input error: %s" % e)
        return EXIT_INPUT_ERROR


# ------------------------------------------------------------------------------------------------
# Output and test helpers
# ------------------------------------------------------------------------------------------------
def _inside_write_roots(path):
    real = os.path.realpath(path)
    probe = real if real.endswith("/") else real + "/"
    if any(probe.startswith(root) for root in FORBIDDEN_ROOTS):
        return False
    return any(probe.startswith(root) for root in WRITE_ROOTS)


def check_out_path(path):
    """Optional outputs only under /mnt/e/ or /dev/shm/, never /mnt/c (or /tmp, which is on C:). Returns the path."""
    if not path or not _inside_write_roots(path):
        raise InputError("refusing output path %r: only /mnt/e/... or /dev/shm/... (never /mnt/c or /tmp)" % path)
    return path


def _csv_text(columns, rows):
    lines = [",".join(columns)]
    for row in rows:
        if isinstance(row, dict):
            values = [row.get(c) for c in columns]
        else:
            values = list(row)
        lines.append(",".join("" if v is None else str(v) for v in values))
    return "\n".join(lines) + "\n"


def _kv_text(mapping):
    return "".join("%s=%s\n" % (k, "" if v is None else v) for k, v in mapping.items())


def write_synthetic_play(play_dir, header, csvs, footer=None, files=None):
    """Test helper: writes a play directory the way the engine side does.
    header / footer: dicts -> key=value files (footer None: no run_footer.txt); csvs: {name: (columns, rows)} with rows
    as lists (or dicts by column), None -> empty field; files: {relpath: bytes}. Refuses any play_dir outside
    /dev/shm/ or /mnt/e/ and anything under /mnt/c or /tmp. Returns the play dir."""
    if not _inside_write_roots(play_dir):
        raise ValueError("refusing synthetic play outside /dev/shm/ or /mnt/e/: %r" % play_dir)
    os.makedirs(play_dir, exist_ok=True)
    with open(os.path.join(play_dir, RUN_HEADER), "w", encoding="ascii", newline="") as f:
        f.write(_kv_text(header))
    for name, (columns, rows) in (csvs or {}).items():
        with open(os.path.join(play_dir, name + ".csv"), "w", encoding="ascii", newline="") as f:
            f.write(_csv_text(columns, rows))
    for rel, data in (files or {}).items():
        path = os.path.join(play_dir, rel)
        if not _inside_write_roots(path):
            raise ValueError("refusing file outside the play: %r" % rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    if footer is not None:
        with open(os.path.join(play_dir, RUN_FOOTER), "w", encoding="ascii", newline="") as f:
            f.write(_kv_text(footer))
    return play_dir


def make_temp_dir(prefix):
    """Test helper (tools/test_deploy_addon.py tmp_base rule): a new directory under $TMPDIR when that lies in
    /dev/shm/ or /mnt/e/, else under /dev/shm/. unittest.SkipTest when neither is usable. Never /tmp. The caller
    removes it (shutil.rmtree in a finally or addCleanup)."""
    base = os.path.realpath(os.environ.get("TMPDIR") or "/")
    if not (_inside_write_roots(base) and os.path.isdir(base) and os.access(base, os.W_OK)):
        base = "/dev/shm"
        if not (os.path.isdir(base) and os.access(base, os.W_OK)):
            raise unittest.SkipTest("no writable $TMPDIR under /dev/shm/ or /mnt/e/ and no /dev/shm; not using /tmp")
    return tempfile.mkdtemp(prefix=prefix, dir=base)
