#!/usr/bin/env python3
"""Tests for tools/harvester/harvest.py (DD-6-min).

    TMPDIR=/dev/shm/<x> PYTHONDONTWRITEBYTECODE=1 python3 -I -B -m pytest --noconftest -p no:cacheprovider -q \
        tests/test_harvest.py

Every test builds a synthetic data root (the layout of /mnt/e/DriveDiT: wb_profile/, client_profiles/iNN/, logs/,
pin/) in a temporary directory under $TMPDIR (on /dev/shm/ or /mnt/e/, else /dev/shm/) and passes it with
--test-root, so nothing is written under /mnt/e/DriveDiT. Sessions come from the DD-5-min generator
(tests/fixtures/capture_dc2/make_synthetic.py, kinds golden, golden_5hz and client_1408). The generator also writes
the harvester's sidecars; the tests move them where the engine and the launchers put their originals instead: the
launch record to <profile>/profile/dc_launch/ and pin/launches/, the console log to <profile>/logs/logs_<local
time>/ (with lines of the same session outside its window, a VME and a script error outside it, a crash.log of a
later crash and an older and a newer log run), and harvest.json is deleted. The pin is a synthetic dd-pin/1 file
with a SHA256SUMS of its copy, so build_key/1 is computed.

Acceptance of the ticket shown here:
  * a T0b Workbench session is renamed on the same volume to captures/arland/<date>/<trial>/<id>, its report is
    written, there is one 512x288 derivative per decoded slot and the checksums verify;
  * a packaged-client session under client_profiles/i01 harvests the same way (1408x720 -> the 1280x720 core);
  * a re-run is a no-op; a run killed at each hook leaves the session complete at its source or its destination,
    and the next run finishes it; an unsealed session is refused; an Everon session is refused with no report, no
    log value and no stdout beyond the refusal;
  * pin_id, build_key and use_class are read before validation and land in catalog/sessions.jsonl; a run of mixed
    build_keys is refused unless --allow-mixed-builds --reason; a -exp session goes to compat19/<pin_id>/.
The 2026-10-22 review (t0c/review/codex-dd6) added: a symlinked profile is not read and a symlinked catalog refuses
the run; a session moved but not catalogued is catalogued only after the map check and a full SHA256SUMS
verification; every kill hook ends with the run manifest written; a cached _harvest/ is rebuilt when its build_key
no longer matches; a headerless crash.log reaches C20; a neighbouring Play's lines stay out of the slice; a torn
catalog line is cut off; off-contract derivative fields hold the session; a second harvester is refused.
DC_TEST_E_SCRATCH (an existing directory under /mnt/e, outside /mnt/e/DriveDiT) also runs the Workbench harvest on
E: (drvfs rename); without it that test skips.
"""

import calendar
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
HARVEST = os.path.join(REPO, "tools", "harvester", "harvest.py")
GOLDEN_JSON = os.path.join(REPO, "tests", "fixtures", "capture_dc2", "golden.json")
WB_ID = "arland_T0b_20261025T091512Z_wb_i00_5e00"
WB5_ID = "arland_T0b_20261025T093000Z_wb_i00_5e05"
CL_ID = "arland_T0b_20261025T094500Z_cl_i01_c140"
LOCAL_OFFSET_S = -7 * 3600          # the engine logs' local time is UTC-7, as on the capture machine
PIN = {"schema": "dd-pin/1", "pin_id": "1.8.0.13", "ticket": "synthetic",
       "game": {"appid": 1874880, "name": "Arma Reforger", "buildid": 24903726},
       "tools": {"appid": 1874910, "name": "Arma Reforger Tools", "buildid": 24870687}}
EXES = {"game/ArmaReforgerSteam.exe": b"synthetic client exe",
        "game/ArmaReforgerSteamDiag.exe": b"synthetic diag client exe",
        "tools/Workbench/ArmaReforgerWorkbenchSteamDiag.exe": b"synthetic workbench exe"}
OTHER_KEY = hashlib.sha256(b"another build").hexdigest()


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harvest = load("dd6_harvest", "tools/harvester/harvest.py")
synth = load("dc2_make_synthetic_h", "tests/fixtures/capture_dc2/make_synthetic.py")
reader = harvest.reader


def scratch_ok(path):
    """True for an existing, writable directory under /dev/shm/ or /mnt/e/ (symlinks resolved) that is not on the
    C: or WSL-image device and not inside /mnt/e/DriveDiT (the production root the tests never write)."""
    real = os.path.realpath(path or "/")
    low = real.lower()
    if not (low.startswith("/dev/shm/") or low.startswith("/mnt/e/")) or low == "/mnt/e/drivedit" \
            or low.startswith("/mnt/e/drivedit/"):
        return False
    return os.path.isdir(real) and os.access(real, os.W_OK) and harvest.forbidden_device_reason(real) is None


def temp_dir(prefix):
    base = os.environ.get("TMPDIR")
    if not scratch_ok(base):
        base = "/dev/shm"
        if not (os.path.isdir(base) and os.access(base, os.W_OK)):
            raise unittest.SkipTest("no usable temporary base under /dev/shm/ or /mnt/e/")
    return tempfile.mkdtemp(prefix=prefix, dir=os.path.realpath(base))


def run(root, *argv):
    """(exit code, stdout) of harvest.main in this process."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = harvest.main(["--test-root", root] + list(argv))
    return code, out.getvalue()


def run_subprocess(root, kill_at=None, *argv):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop(harvest.KILL_ENV, None)
    if kill_at:
        env[harvest.KILL_ENV] = kill_at
    return subprocess.run([sys.executable, "-I", "-B", HARVEST, "--test-root", root] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)


def json_bytes(obj):
    return (json.dumps(obj, indent=2) + "\n").encode("ascii")


def read_json(path):
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def snapshot(root):
    """{relative path: sha256} of every file under root (directories with no file are listed with None)."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if not dirnames and not filenames:
            out[os.path.relpath(dirpath, root) + "/"] = None
        for name in filenames:
            full = os.path.join(dirpath, name)
            with open(full, "rb") as f:
                out[os.path.relpath(full, root)] = hashlib.sha256(f.read()).hexdigest()
    return out


def expected_build_key():
    """build_key/1 of PIN and EXES written out by hand (the test vector of the harvest.build_key_v1 text)."""
    text = ("build_key/1\ngame.appid=1874880\ngame.buildid=24903726\ngame.exe_sha256=%s\n"
            "tools.appid=1874910\ntools.buildid=24870687\ntools.exe_sha256=%s\n") % (
        hashlib.sha256(EXES["game/ArmaReforgerSteam.exe"]).hexdigest(),
        hashlib.sha256(EXES["tools/Workbench/ArmaReforgerWorkbenchSteamDiag.exe"]).hexdigest())
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def write_pin(root, sums=True):
    """pin/pin_1.8.0.13.json (synthetic dd-pin/1) and, with sums, pin/1.8.0.13/SHA256SUMS of its copy. Returns the
    pin file's sha256 (the sessions' pin.pin_sha256)."""
    data = json_bytes(PIN)
    write(os.path.join(root, "pin", "pin_1.8.0.13.json"), data)
    if sums:
        lines = ["%s  %s" % (hashlib.sha256(v).hexdigest(), k) for k, v in sorted(EXES.items())]
        write(os.path.join(root, "pin", "1.8.0.13", "SHA256SUMS"), ("\n".join(lines) + "\n").encode("ascii"))
    return hashlib.sha256(data).hexdigest()


def stamp(unix_s):
    return time.strftime("%H:%M:%S", time.gmtime(unix_s + LOCAL_OFFSET_S)) + ".%03d" % int(round(
        (unix_s % 1) * 1000))


def log_file(path, start_utc, lines):
    """An engine log in the engine's format: the header with local and UTC start time, then lines."""
    local = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_utc + LOCAL_OFFSET_S))
    utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_utc))
    head = ["-" * 45, "Log E:\\DriveDiT\\synthetic\\%s started at %s (%s UTC)" % (os.path.basename(path), local, utc),
            ""]
    write(path, ("\r\n".join(head + lines) + "\r\n").encode("ascii"))


def run_dir_name(start_utc):
    return "logs_" + time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime(start_utc + LOCAL_OFFSET_S))


def write_engine_logs(log_base, sid, start_unix, end_unix, session_lines, before=(), after=()):
    """The log run that holds the session (started 3 h before it, the previous local day, so the slice rolls over
    midnight), an older run and a newer run. Inside the window: the session's own console lines. Outside it: a
    Screenshot-saved line of this very session (C11 would count it), a script error and a VME of ours (C20 would
    fail on them) and a later crash.log."""
    run_start = start_unix - 3 * 3600
    rdir = os.path.join(log_base, run_dir_name(run_start))
    console = ["%s ENGINE       : startup" % stamp(run_start + 5),
               "%s RESOURCES    : Screenshot saved to $profile:DC_Sessions/%s/frames/f_00009999.png"
               % (stamp(start_unix - 300), sid)]
    console += list(before) + session_lines + list(after)
    console += ["%s SCRIPT    (E): Virtual Machine Exception" % stamp(end_unix + 120),
                "Reason: NULL pointer to instance", "Class: 'DC_Session'", "Function: 'Tick'"]
    log_file(os.path.join(rdir, "console.log"), run_start, console)
    log_file(os.path.join(rdir, "script.log"), run_start + 2, [
        "%s  SCRIPT    (E): Scripts/Game/DataCapture/DC_Session.c(12): error before" % stamp(start_unix - 200),
        "%s  SCRIPT       : [DC] heartbeat" % stamp(start_unix + 5)])
    log_file(os.path.join(rdir, "crash.log"), end_unix + 600, ["", "Assertion failed"])
    old = start_unix - 2 * 86400
    log_file(os.path.join(log_base, run_dir_name(old), "console.log"), old, ["%s ENGINE : old" % stamp(old + 1)])
    log_file(os.path.join(log_base, run_dir_name(old), "crash.log"), old + 60, ["Assertion failed"])
    new = end_unix + 3600
    log_file(os.path.join(log_base, run_dir_name(new), "console.log"), new, ["%s ENGINE : new" % stamp(new + 1)])


def make_session(root, kind="golden", variant=None, pin_sha=None, edit=None, client="i01", logs=None):
    """Generate one session into its profile's DC_Sessions folder and move the generator's sidecars where the engine
    and the launchers put their originals. edit(meta, end, record) may change session.json, session_end.json and
    the launch record; logs(start, end) may return console lines (before, after) to put around the session's own.
    Returns the session directory."""
    is_client = kind == "client_1408"
    profile = os.path.join(root, "client_profiles", client) if is_client else os.path.join(root, "wb_profile")
    sessions = os.path.join(profile, "profile", "DC_Sessions")
    os.makedirs(sessions, exist_ok=True)
    sdir = synth.generate(sessions, kind=kind, variant=variant)["session_dir"]
    meta = read_json(os.path.join(sdir, "session.json"))
    end = read_json(os.path.join(sdir, "session_end.json"))
    lid = meta["pin"]["launch_id"]
    record = read_json(os.path.join(sdir, "_launch", "launch_%s.json" % lid))
    if pin_sha:
        meta["pin"]["pin_sha256"] = pin_sha
        record["pin"]["sha256"] = pin_sha
    if edit:
        edit(meta, end, record)
    write(os.path.join(sdir, "session.json"), json_bytes(meta))
    write(os.path.join(sdir, "session_end.json"), json_bytes(end))
    lid = meta["pin"]["launch_id"]
    for d in (os.path.join(profile, "profile", "dc_launch"), os.path.join(root, "pin", "launches")):
        write(os.path.join(d, "launch_%s.json" % lid), json_bytes(record))
    with open(os.path.join(sdir, "_engine_logs", "console.log"), "rb") as f:
        utc_lines = f.read().decode("ascii").splitlines()
    day = calendar.timegm(time.gmtime(meta["start"]["unix_s"])[:3] + (0, 0, 0))
    local = []
    for line in utc_lines:
        h, m, s = int(line[0:2]), int(line[3:5]), float(line[6:12])
        local.append(stamp(day + h * 3600 + m * 60 + s) + line[12:])
    log_base = os.path.join(root, "logs", client) if is_client else os.path.join(profile, "logs")
    around = logs(meta["start"]["unix_s"], end["end"]["unix_s"]) if logs else ((), ())
    write_engine_logs(log_base, meta["session_id"], meta["start"]["unix_s"], end["end"]["unix_s"], local, *around)
    shutil.rmtree(os.path.join(sdir, "_launch"))
    shutil.rmtree(os.path.join(sdir, "_engine_logs"))
    os.remove(os.path.join(sdir, "harvest.json"))
    return sdir


def dest_of(root, sid):
    return os.path.join(root, "captures", "arland", "2026-10-25", "T0b", sid)


def catalog(root):
    path = os.path.join(root, "catalog", "sessions.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "rb") as f:
        return [json.loads(line) for line in f.read().decode("ascii").splitlines() if line]


def verify_sums_independently(test, sdir):
    """Re-hash every file SHA256SUMS lists and check the listing is complete (not through harvest.verify_sums)."""
    with open(os.path.join(sdir, "_harvest", "SHA256SUMS"), "rb") as f:
        lines = f.read().decode("ascii").splitlines()
    listed = {}
    for line in lines:
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
    present = set()
    for dirpath, _, filenames in os.walk(sdir):
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), sdir)
            if rel != "_harvest/SHA256SUMS":
                present.add(rel)
    test.assertEqual(present, set(listed))
    for rel, digest in listed.items():
        with open(os.path.join(sdir, rel), "rb") as f:
            test.assertEqual(hashlib.sha256(f.read()).hexdigest(), digest, rel)
    test.assertEqual(lines, sorted(lines, key=lambda line: line.split("  ", 1)[1].encode("ascii")))


def decode_jpeg(path):
    import cv2

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class HarvestWorkbenchTest(unittest.TestCase):
    """One T0b Workbench session harvested once; each test checks one property of the result."""

    @classmethod
    def setUpClass(cls):
        cls.root = temp_dir("dd6_wb_")
        cls.pin_sha = write_pin(cls.root)
        cls.src = make_session(cls.root, pin_sha=cls.pin_sha)
        cls.raw = snapshot(cls.src)
        cls.ino = os.stat(os.path.join(cls.src, "session.json")).st_ino
        cls.code, cls.out = run(cls.root)
        cls.dest = dest_of(cls.root, WB_ID)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_renamed_to_the_dated_trial_directory(self):
        self.assertEqual(self.code, 0, self.out)
        self.assertFalse(os.path.exists(self.src))
        self.assertTrue(os.path.isdir(self.dest))
        # a rename, not a copy: the raw files keep their inode and content
        self.assertEqual(os.stat(os.path.join(self.dest, "session.json")).st_ino, self.ino)
        moved = snapshot(self.dest)
        for rel, digest in self.raw.items():
            self.assertEqual(moved.get(rel), digest, rel)
        self.assertIn("harvested %s: PASS -> captures/arland/2026-10-25/T0b/%s" % (WB_ID, WB_ID), self.out)

    def test_report_written(self):
        rep = read_json(os.path.join(self.dest, "_harvest", "dc_validate.json"))
        self.assertEqual((rep["verdict"], rep["exit_code"]), ("PASS", 0))
        checks = dict((c["id"], c) for c in rep["checks"])
        self.assertEqual(checks["C02"]["status"], "PASS")         # the launch record copy and build_key joined
        self.assertEqual(checks["C11"]["value"]["screenshot_saved_lines"], 130)    # the slice holds the session only
        self.assertEqual(checks["C20"]["status"], "PASS")         # the VME and script error outside the window
        result = read_json(os.path.join(self.dest, "_harvest", "result.json"))
        self.assertEqual((result["verdict"], result["route"], result["dest"]),
                         ("PASS", "captures", "captures/arland/2026-10-25/T0b/" + WB_ID))

    def test_one_512x288_derivative_per_decoded_slot(self):
        with open(os.path.join(self.dest, "_harvest", "img_decoded.csv"), "rb") as f:
            data = f.read()
        fields = reader.load_schema().raw["derived"]["img_resolved"]["fields"]
        problems = []
        columns, rows = reader.parse_csv_bytes(data, fields, "img_decoded.csv", problems)
        self.assertEqual(problems, [])
        self.assertEqual(columns, [f["name"] for f in fields])
        decoded = [r for r in rows if r["fid_ok"]]
        self.assertEqual(len(decoded), 130)
        names = sorted(os.listdir(os.path.join(self.dest, "_harvest", "deriv")))
        self.assertEqual(names, ["f_%08d.jpg" % r["slot_id"] for r in decoded])
        for r in decoded:
            self.assertEqual(r["deriv_path"], "_harvest/deriv/f_%08d.jpg" % r["slot_id"])
            self.assertEqual(decode_jpeg(os.path.join(self.dest, r["deriv_path"])).shape, (288, 512, 3))
        golden = read_json(GOLDEN_JSON)["img_resolved"]
        self.assertEqual([(r["slot_id"], r["captured_render_frame"]) for r in rows],
                         [(g["slot_id"], g["captured_render_frame"]) for g in golden])
        self.assertTrue(all(r["fid_margin"] is not None and r["fid_margin"] > 0 for r in decoded))

    def test_derivative_is_the_centre_16_9_of_the_core(self):
        import cv2

        raw = reader.fiducial.load_rgb(os.path.join(self.dest, "frames", "f_00000040.png"))
        # 256x240 capture, 64 px bands: core 128x240, centre 16:9 = 128x72 from row 84
        want = cv2.resize(raw[84:156, 64:192], (512, 288), interpolation=cv2.INTER_LINEAR).astype(float)
        got = decode_jpeg(os.path.join(self.dest, "_harvest", "deriv", "f_00000040.jpg")).astype(float)
        self.assertLess(numpy.abs(got - want).mean(), 2.0)
        band = cv2.resize(raw[84:156, 0:128], (512, 288), interpolation=cv2.INTER_LINEAR).astype(float)
        self.assertGreater(numpy.abs(got - band).mean(), 10.0)

    def test_checksums_verify(self):
        verify_sums_independently(self, self.dest)
        self.assertEqual(harvest.verify_sums(self.dest)[0], True)

    def test_sidecars(self):
        stamp_ = read_json(os.path.join(self.dest, "harvest.json"))
        self.assertEqual(stamp_["build_key"], expected_build_key())
        self.assertEqual(stamp_["build_key_source"], "computed build_key/1")
        self.assertEqual((stamp_["pin_id"], stamp_["use_class"], stamp_["channel"]), ("1.8.0.13", "dev", "stable"))
        lid = read_json(os.path.join(self.dest, "session.json"))["pin"]["launch_id"]
        with open(os.path.join(self.dest, "_launch", "launch_%s.json" % lid), "rb") as f:
            copy = f.read()
        with open(os.path.join(self.root, "pin", "launches", "launch_%s.json" % lid), "rb") as f:
            self.assertEqual(copy, f.read())
        self.assertEqual(stamp_["launch_record"]["source"], "profile dc_launch + pin launches")
        logs = sorted(os.listdir(os.path.join(self.dest, "_engine_logs")))
        self.assertEqual(logs, ["console.log", "script.log"])      # crash.log of a later crash is outside
        with open(os.path.join(self.dest, "_engine_logs", "console.log"), "rb") as f:
            console = f.read().decode("ascii")
        self.assertNotIn("f_00009999", console)
        self.assertNotIn("Virtual Machine Exception", console)
        self.assertEqual(console.count("Screenshot saved to"), 130)
        self.assertEqual(stamp_["engine_logs"]["run"], run_dir_name(read_json(
            os.path.join(self.dest, "session.json"))["start"]["unix_s"] - 3 * 3600))

    def test_catalog_row_and_run_manifest(self):
        rows = catalog(self.root)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(list(row), list(harvest.CATALOG_COLUMNS))
        self.assertEqual((row["session_id"], row["pin_id"], row["build_key"], row["use_class"], row["verdict"]),
                         (WB_ID, "1.8.0.13", expected_build_key(), "dev", "PASS"))
        with open(os.path.join(self.dest, "_harvest", "SHA256SUMS"), "rb") as f:
            self.assertEqual(row["sha256sums_sha256"], hashlib.sha256(f.read()).hexdigest())
        self.assertEqual(row["mixed_builds_reason"], None)
        runs = os.listdir(os.path.join(self.root, "manifests", "harvest_runs"))
        self.assertEqual(runs, ["harvest_%s.json" % row["run_id"]])
        doc = read_json(os.path.join(self.root, "manifests", "harvest_runs", runs[0]))
        self.assertEqual((doc["build_keys"], doc["mixed_builds"], doc["reason"]), ([expected_build_key()], False, None))
        self.assertEqual([s["session_id"] for s in doc["sessions"]], [WB_ID])

    def test_rerun_is_a_noop(self):
        before = snapshot(self.root)
        code, out = run(self.root)
        self.assertEqual(code, 0, out)
        self.assertIn("nothing", out)
        self.assertEqual(snapshot(self.root), before)
        code, out = run(self.root, "--session", self.src)          # the session by name: already harvested
        self.assertEqual(code, 0, out)
        self.assertIn("already harvested", out)
        self.assertEqual(snapshot(self.root), before)


class HarvestClientTest(unittest.TestCase):
    """A packaged-client session under client_profiles/i01: 1408x720 captures, the derivative from the core."""

    @classmethod
    def setUpClass(cls):
        cls.root = temp_dir("dd6_cl_")
        sha = write_pin(cls.root)
        cls.src = make_session(cls.root, kind="client_1408", pin_sha=sha)
        cls.code, cls.out = run(cls.root)
        cls.dest = dest_of(cls.root, CL_ID)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_harvested_like_workbench(self):
        self.assertEqual(self.code, 0, self.out)
        self.assertFalse(os.path.exists(self.src))
        rep = read_json(os.path.join(self.dest, "_harvest", "dc_validate.json"))
        self.assertEqual(rep["verdict"], "PASS")
        result = read_json(os.path.join(self.dest, "_harvest", "result.json"))
        self.assertEqual((result["decoded"], result["derivatives"]), (130, 130))
        self.assertEqual(result["derivative"]["box"], [64, 0, 1280, 720])
        verify_sums_independently(self, self.dest)
        row = catalog(self.root)[0]
        self.assertEqual((row["exe"], row["instance"], row["build_key"]), ("cl", "i01", expected_build_key()))
        self.assertTrue(os.path.isfile(os.path.join(self.dest, "_engine_logs", "console.log")))   # from logs/i01

    def test_derivative_is_the_core_without_bands(self):
        import cv2

        raw = reader.fiducial.load_rgb(os.path.join(self.dest, "frames", "f_00000070.png"))
        self.assertEqual(raw.shape, (720, 1408, 3))
        want = cv2.resize(raw[:, 64:1344], (512, 288), interpolation=cv2.INTER_AREA).astype(float)
        got = decode_jpeg(os.path.join(self.dest, "_harvest", "deriv", "f_00000070.jpg")).astype(float)
        self.assertLess(numpy.abs(got - want).mean(), 2.0)
        full = cv2.resize(raw, (512, 288), interpolation=cv2.INTER_AREA).astype(float)
        self.assertGreater(numpy.abs(got - full).mean(), numpy.abs(got - want).mean() + 2.0)


class RefusalTest(unittest.TestCase):
    """Sessions that stay in place: Everon and Kolguyev (refused before any report), unsealed (skipped), held
    sessions (missing inputs, TBD-4 names), the run-level refusals and the roots."""

    def setUp(self):
        self.root = temp_dir("dd6_ref_")
        self.pin_sha = write_pin(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def assert_nothing_written(self, before, code, out, want_code=2):
        self.assertEqual(code, want_code, out)
        self.assertEqual(snapshot(self.root), before)

    def test_everon_session_json_is_refused_with_one_line(self):
        def everon(meta, end, record):
            meta["map"].update({"map_id": "everon", "terrain": "Eden",
                                "root_world": "{853E92315D1D9EFE}worlds/Eden/Eden.ent"})
        make_session(self.root, pin_sha=self.pin_sha, edit=everon)
        before = snapshot(self.root)
        proc = run_subprocess(self.root)
        self.assertEqual(proc.returncode, 2, proc.stdout)
        self.assertEqual(proc.stderr, b"")
        lines = proc.stdout.decode("ascii").splitlines()
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith("refused wb_profile/profile/DC_Sessions/%s: not an Arland session"
                                            % WB_ID), lines[0])
        for value in ("Eden", "853E92315D1D9EFE", "dev", "1.8.0.13", "T0b_20261025T091512Z_wb_i00_5e00:"):
            self.assertNotIn(value, lines[0].split(": ", 1)[1])
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse(os.path.exists(os.path.join(self.root, "catalog")))

    def test_everon_directory_is_refused_without_reading_it(self):
        src = make_session(self.root, pin_sha=self.pin_sha)
        everon = os.path.join(os.path.dirname(src), "everon_T0b_20261025T091512Z_wb_i00_5e00")
        os.rename(src, everon)
        write(os.path.join(everon, "session.json"), b"{not json")      # read it and it would be held instead
        before = snapshot(self.root)
        code, out = run(self.root)
        self.assert_nothing_written(before, code, out)
        self.assertEqual(out.splitlines(), ["refused wb_profile/profile/DC_Sessions/everon_T0b_20261025T091512Z_"
                                            "wb_i00_5e00: not an Arland session; DD-6-min harvests Arland only "
                                            "(Everon is sealed, Kolguyev routing is DD-6); nothing was read or "
                                            "written"])

    def test_kolguyev_is_refused(self):
        def kolguyev(meta, end, record):
            meta["map"].update({"map_id": "kolguyev", "terrain": "Cain",
                                "root_world": "{1EA95DAE3230BEB0}worlds/Cain/Cain.ent"})
        make_session(self.root, pin_sha=self.pin_sha, edit=kolguyev)
        before = snapshot(self.root)
        code, out = run(self.root)
        self.assert_nothing_written(before, code, out)
        self.assertTrue(out.startswith("refused "), out)

    def test_unsealed_session_is_skipped(self):
        src = make_session(self.root, pin_sha=self.pin_sha)
        os.remove(os.path.join(src, "session_end.json"))
        before = snapshot(self.root)
        code, out = run(self.root)
        self.assert_nothing_written(before, code, out, want_code=0)
        self.assertIn("skipped wb_profile/profile/DC_Sessions/%s: not sealed" % WB_ID, out)
        code, out = run(self.root, "--session", src)
        self.assert_nothing_written(before, code, out, want_code=2)

    def test_failed_session_goes_to_rejected(self):
        make_session(self.root, variant="gap", pin_sha=self.pin_sha)
        code, out = run(self.root)
        self.assertEqual(code, 1, out)
        dest = os.path.join(self.root, "captures", "_rejected", WB_ID)
        rep = read_json(os.path.join(dest, "_harvest", "dc_validate.json"))
        self.assertEqual(rep["verdict"], "FAIL")
        self.assertEqual(sorted(c["id"] for c in rep["checks"] if c["status"] == "FAIL"), ["C06"])
        self.assertFalse(os.path.exists(os.path.join(dest, "_harvest", "deriv")))
        self.assertTrue(os.path.isfile(os.path.join(dest, "_harvest", "img_decoded.csv")))
        verify_sums_independently(self, dest)
        row = catalog(self.root)[0]
        self.assertEqual((row["verdict"], row["route"], row["derivatives"]), ("FAIL", "rejected", 0))
        self.assertFalse(os.path.exists(os.path.join(self.root, "manifests")))     # nothing went to captures/arland

    def test_experimental_session_goes_to_compat19(self):
        key = hashlib.sha256(b"synthetic 1.9 build").hexdigest()

        def experimental(meta, end, record):
            for obj in (meta["pin"], end["pin"]):
                obj.update({"pin_id": "1.9.0.10-exp", "build_key": key})
            meta["pin"].update({"channel": "experimental", "launch_schema": "dd-launch/2"})
            meta["use_class"] = "compat-preview"
            record.update({"schema": "dd-launch/2", "build_key": key, "channel": "experimental",
                           "use_class": "compat-preview"})
            record["pin"]["pin_id"] = "1.9.0.10-exp"
        make_session(self.root, pin_sha=self.pin_sha, edit=experimental)
        code, out = run(self.root)
        self.assertEqual(code, 0, out)
        dest = os.path.join(self.root, "compat19", "1.9.0.10-exp", WB_ID)
        self.assertTrue(os.path.isdir(dest), out)
        self.assertFalse(os.path.exists(os.path.join(self.root, "captures")))
        with open(os.path.join(dest, "_harvest", "dc_validate.txt"), "rb") as f:
            self.assertIn(b"refused", f.read())                       # no thresholds exist for an -exp pin
        row = catalog(self.root)[0]
        self.assertEqual((row["route"], row["verdict"], row["pin_id"], row["build_key"], row["use_class"]),
                         ("compat19", "REFUSED", "1.9.0.10-exp", key, "compat-preview"))
        self.assertEqual(row["build_key_source"], "session.json + launch_record")
        verify_sums_independently(self, dest)

    def test_mixed_builds_need_a_reason(self):
        make_session(self.root, pin_sha=self.pin_sha)

        def other_build(meta, end, record):
            meta["pin"]["build_key"] = OTHER_KEY
            end["pin"]["build_key"] = OTHER_KEY
        make_session(self.root, kind="golden_5hz", pin_sha=self.pin_sha, edit=other_build)
        before = snapshot(self.root)
        code, out = run(self.root)
        self.assert_nothing_written(before, code, out, want_code=3)
        self.assertIn("2 build_keys", out)
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            run(self.root, "--allow-mixed-builds")
        self.assertEqual(cm.exception.code, 3)
        self.assertEqual(snapshot(self.root), before)
        code, out = run(self.root, "--allow-mixed-builds", "--reason", "test of a declared mixed stage")
        self.assertEqual(code, 0, out)
        runs = os.listdir(os.path.join(self.root, "manifests", "harvest_runs"))
        doc = read_json(os.path.join(self.root, "manifests", "harvest_runs", runs[0]))
        self.assertEqual((doc["mixed_builds"], doc["reason"]), (True, "test of a declared mixed stage"))
        self.assertEqual(doc["build_keys"], sorted([expected_build_key(), OTHER_KEY]))
        self.assertEqual(sorted(r["session_id"] for r in catalog(self.root)), [WB_ID, WB5_ID])

    def test_build_key_sources(self):
        src = make_session(self.root, pin_sha=self.pin_sha)
        sums = os.path.join(self.root, "pin", "1.8.0.13", "SHA256SUMS")
        sidecar = os.path.join(self.root, "pin", "pin_1.8.0.13.build_key.json")
        code, out = run(self.root, "--dry-run")
        self.assertEqual(code, 0, out)
        self.assertIn("build_key %s (computed build_key/1)" % expected_build_key()[:16], out)
        os.remove(sums)
        write(sidecar, json_bytes({"pin_id": "1.8.0.13", "build_key": OTHER_KEY}))
        code, out = run(self.root, "--dry-run")
        self.assertIn("build_key %s (pin_sidecar)" % OTHER_KEY[:16], out)
        meta = read_json(os.path.join(src, "session.json"))
        meta["pin"]["build_key"] = expected_build_key()
        write(os.path.join(src, "session.json"), json_bytes(meta))
        code, out = run(self.root, "--dry-run")
        self.assertEqual(code, 2, out)
        self.assertIn("build_key sources disagree", out)
        meta["pin"]["build_key"] = None
        write(os.path.join(src, "session.json"), json_bytes(meta))
        os.remove(sidecar)
        code, out = run(self.root, "--dry-run")
        self.assertEqual(code, 2, out)
        self.assertIn("no build_key for pin 1.8.0.13", out)
        write(sidecar, json_bytes({"pin_id": "1.8.0.13", "build_key": OTHER_KEY}))
        write(os.path.join(self.root, "pin", "pin_1.8.0.13.json"), json_bytes(dict(PIN, ticket="edited")))
        code, out = run(self.root, "--dry-run")
        self.assertIn("is not the pin this session was launched with", out)

    def test_off_contract_derivative_is_held(self):
        def square(meta, end, record):
            meta["rig"]["derivative_wh"] = [512, 512]
        make_session(self.root, pin_sha=self.pin_sha, edit=square)
        before = snapshot(self.root)
        code, out = run(self.root)
        self.assert_nothing_written(before, code, out)
        self.assertIn("are not the contract's [512, 288] / jpeg_q95", out)

    def test_symlinked_profile_is_not_read(self):
        src = make_session(self.root, pin_sha=self.pin_sha)
        real = os.path.join(self.root, "elsewhere")
        os.rename(os.path.join(self.root, "wb_profile"), real)
        os.symlink(real, os.path.join(self.root, "wb_profile"))
        before = snapshot(self.root)
        code, out = run(self.root)
        self.assert_nothing_written(before, code, out, want_code=0)
        self.assertIn("nothing", out)
        self.assertTrue(os.path.isdir(os.path.join(real, os.path.relpath(src, os.path.join(self.root, "wb_profile")))))

    def test_symlinked_catalog_refuses_the_run(self):
        src = make_session(self.root, pin_sha=self.pin_sha)
        target = os.path.join(self.root, "target.txt")
        write(target, b"keep\n")
        os.makedirs(os.path.join(self.root, "catalog"))
        os.symlink(target, os.path.join(self.root, "catalog", "sessions.jsonl"))
        before = snapshot(self.root)
        code, out = run(self.root)
        self.assert_nothing_written(before, code, out, want_code=3)
        self.assertIn("is a symlink", out)
        self.assertTrue(os.path.isdir(src))

    def test_cached_harvest_is_rebuilt_when_the_build_key_changes(self):
        make_session(self.root, pin_sha=self.pin_sha)
        proc = run_subprocess(self.root, "after_harvest_rename")
        self.assertEqual(proc.returncode, -signal.SIGKILL)
        write(os.path.join(self.root, "pin", "pin_1.8.0.13.build_key.json"),
              json_bytes({"pin_id": "1.8.0.13", "build_key": OTHER_KEY}))
        code, out = run(self.root)
        self.assertEqual(code, 0, out)
        dest = dest_of(self.root, WB_ID)
        self.assertEqual(read_json(os.path.join(dest, "harvest.json"))["build_key"], OTHER_KEY)
        self.assertEqual([r["build_key"] for r in catalog(self.root)], [OTHER_KEY])
        verify_sums_independently(self, dest)

    def test_headerless_crash_log_reaches_c20(self):
        src = make_session(self.root, pin_sha=self.pin_sha)
        start = read_json(os.path.join(src, "session.json"))["start"]["unix_s"]
        write(os.path.join(self.root, "wb_profile", "logs", run_dir_name(start - 3 * 3600), "crash.log"),
              b"Assertion failed\r\nProgram: synthetic\r\n")
        code, out = run(self.root)
        self.assertEqual(code, 1, out)
        dest = os.path.join(self.root, "captures", "_rejected", WB_ID)
        rep = read_json(os.path.join(dest, "_harvest", "dc_validate.json"))
        c20 = [c for c in rep["checks"] if c["id"] == "C20"][0]
        self.assertEqual((c20["status"], c20["value"]["hard"]["crash_log"]), ("FAIL", 1))
        self.assertEqual(read_json(os.path.join(dest, "harvest.json"))["engine_logs"]["kept_whole"], ["crash.log"])

    def test_neighbouring_plays_stay_out_of_the_slice(self):
        other = "arland_T0b_20261025T091540Z_wb_i00_beef"

        def neighbours(start, end):
            before = ["%s SCRIPT    (E): Virtual Machine Exception" % stamp(start - 5),
                      "Class: 'DC_Session'"]
            after = ["%s SCRIPT       : [DC] shutdown done" % stamp(end + 3),
                     "%s SCRIPT       : [DC] session_start id=%s" % (stamp(end + 6), other),
                     "%s SCRIPT    (E): Virtual Machine Exception" % stamp(end + 8),
                     "Class: 'DC_Session'"]
            return before, after
        make_session(self.root, pin_sha=self.pin_sha, logs=neighbours)
        code, out = run(self.root)
        self.assertEqual(code, 0, out)
        dest = dest_of(self.root, WB_ID)
        with open(os.path.join(dest, "_engine_logs", "console.log"), "rb") as f:
            console = f.read().decode("ascii")
        self.assertIn("[DC] shutdown done", console)
        self.assertNotIn(other, console)
        self.assertNotIn("Virtual Machine Exception", console)

    def test_torn_catalog_line_is_cut_before_the_next_row(self):
        make_session(self.root, pin_sha=self.pin_sha)
        self.assertEqual(run(self.root)[0], 0)
        with open(os.path.join(self.root, "catalog", "sessions.jsonl"), "ab") as f:
            f.write(b'{"schema":"drivedit.catalog-sess')
        make_session(self.root, kind="golden_5hz", pin_sha=self.pin_sha)
        code, out = run(self.root)
        self.assertEqual(code, 0, out)
        self.assertEqual([r["session_id"] for r in catalog(self.root)], [WB_ID, WB5_ID])

    def test_uncatalogued_destination_is_checked_before_its_row(self):
        make_session(self.root, pin_sha=self.pin_sha)
        self.assertEqual(run(self.root)[0], 0)
        dest = dest_of(self.root, WB_ID)
        os.remove(os.path.join(self.root, "catalog", "sessions.jsonl"))
        with open(os.path.join(dest, "heartbeat.csv"), "ab") as f:
            f.write(b"damaged\n")
        code, out = run(self.root)
        self.assertEqual(code, 2, out)
        self.assertIn("moved but", out)
        self.assertEqual(catalog(self.root), [])
        everon = os.path.join(self.root, "captures", "_rejected", "arland_T0b_20261025T091512Z_wb_i00_0bad")
        shutil.copytree(dest, everon)
        shutil.rmtree(dest)
        meta = read_json(os.path.join(everon, "session.json"))
        meta["map"].update({"map_id": "everon", "terrain": "Eden"})
        write(os.path.join(everon, "session.json"), json_bytes(meta))
        before = snapshot(self.root)
        code, out = run(self.root)
        self.assert_nothing_written(before, code, out)
        self.assertEqual(out.splitlines(), ["refused captures/_rejected/arland_T0b_20261025T091512Z_wb_i00_0bad: "
                                            + harvest.NOT_ARLAND + " (found at a destination; not catalogued)"])

    def test_second_harvester_is_refused(self):
        src = make_session(self.root, pin_sha=self.pin_sha)
        os.makedirs(os.path.join(self.root, "catalog"))
        fd = os.open(os.path.join(self.root, "catalog", "harvest.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            before = snapshot(self.root)
            code, out = run(self.root)
            self.assert_nothing_written(before, code, out, want_code=3)
            self.assertIn("another harvester holds", out)
        finally:
            os.close(fd)
        self.assertTrue(os.path.isdir(src))

    def test_build_key_v1_vector(self):
        pin = {"game": {"appid": 1874880, "buildid": 24903726}, "tools": {"appid": 1874910, "buildid": 24870687}}
        sums = {"game/ArmaReforgerSteam.exe": "a" * 64, "tools/Workbench/ArmaReforgerWorkbenchSteamDiag.exe": "b" * 64}
        self.assertEqual(harvest.build_key_v1(pin, sums),
                         "c419427920d8584b47d41714449fb4ef183b2c9f463893367b535df8de0f44b6")
        with self.assertRaises(ValueError):
            harvest.build_key_v1(pin, {"game/ArmaReforgerSteam.exe": "a" * 64})

    def test_tbd4_other_screenshot_name_is_held(self):
        src = make_session(self.root, pin_sha=self.pin_sha)
        fdir = os.path.join(src, "frames")
        for name in os.listdir(fdir):
            os.rename(os.path.join(fdir, name), os.path.join(fdir, name[:-len(".png")]))
        before = snapshot(self.root)
        code, out = run(self.root)
        self.assert_nothing_written(before, code, out)
        self.assertIn("TBD-4: frames/ holds 130 file(s) f_<slot:08d> without extension", out)

    def test_launch_record_missing_or_different_is_held(self):
        src = make_session(self.root, pin_sha=self.pin_sha)
        lid = read_json(os.path.join(src, "session.json"))["pin"]["launch_id"]
        copy = os.path.join(self.root, "wb_profile", "profile", "dc_launch", "launch_%s.json" % lid)
        original = os.path.join(self.root, "pin", "launches", "launch_%s.json" % lid)
        with open(copy, "ab") as f:
            f.write(b" ")
        before = snapshot(self.root)
        code, out = run(self.root)
        self.assert_nothing_written(before, code, out)
        self.assertIn("launch record copies of launch_%s.json differ" % lid, out)
        os.remove(copy)
        os.remove(original)
        before = snapshot(self.root)
        code, out = run(self.root)
        self.assert_nothing_written(before, code, out)
        self.assertIn("no launch record launch_%s.json" % lid, out)

    def test_roots(self):
        self.assertEqual(harvest.DATA_ROOT, "/mnt/e/DriveDiT")
        for bad in ("/mnt/c/Users/x", "C:\\DriveDiT", "/tmp/x", "/mnt/e/DriveDiT", "/mnt/e/DriveDiT/captures",
                    "/mnt/e", "/dev/shm", "/mnt/e/__dd6_no_such_dir__", "/"):
            with self.assertRaises(harvest.Refused, msg=bad):
                harvest.data_root(bad)
        self.assertFalse(os.path.exists("/mnt/e/__dd6_no_such_dir__"))
        self.assertEqual(harvest.data_root(self.root), os.path.realpath(self.root))
        code, out = run(self.root, "--session", os.path.join(self.root, "elsewhere", WB_ID))
        self.assertEqual(code, 3, out)
        self.assertTrue(out.startswith("refused: --session"), out)

    def test_derivative_plan(self):
        def rig(cap, band, wh=(512, 288), codec="jpeg_q95"):
            return {"rig": {"capture_wh": list(cap), "core_wh": [cap[0] - 2 * band, cap[1]], "band_px": band,
                            "derivative_wh": list(wh), "derivative_codec": codec}}
        self.assertEqual(harvest.derivative_plan(rig((1408, 720), 64)),
                         {"box": [64, 0, 1280, 720], "wh": [512, 288], "codec": "jpeg_q95", "quality": 95,
                          "interpolation": "area"})
        self.assertEqual(harvest.derivative_plan(rig((1920, 1080), 64))["box"], [64, 36, 1792, 1008])
        self.assertEqual(harvest.derivative_plan(rig((2560, 1080), 64))["box"], [320, 0, 1920, 1080])
        self.assertEqual(harvest.derivative_plan(rig((256, 240), 64))["box"], [64, 84, 128, 72])
        for bad in ({"rig": {"capture_wh": [1408, 720]}}, rig((1408, 720), -1)):
            with self.assertRaises(harvest.Held):
                harvest.derivative_plan(bad)

    def test_log_slice_rolls_over_midnight_and_keeps_blocks(self):
        start = calendar.timegm((2026, 10, 25, 6, 50, 0))           # 23:50 local the day before
        data = ("\r\n".join([
            "-" * 45, "Log x started at 2026-10-24 23:50:00 (2026-10-25 06:50:00 UTC)", "",
            "23:59:59.000 A before", "00:00:01.500 B inside", "Class: 'DC_Session'",
            "00:00:09.000 C after"]) + "\r\n").encode("ascii")
        lo = calendar.timegm((2026, 10, 25, 7, 0, 0))
        lines = harvest.slice_log(data, lo, lo + 5)
        self.assertEqual(lines, [b"00:00:01.500 B inside\r", b"Class: 'DC_Session'\r"])
        self.assertIsNone(harvest.slice_log(b"no header\n", 0, 1e12))
        self.assertEqual(harvest.log_header(data)[:2], (start, LOCAL_OFFSET_S))


class KillMidRunTest(unittest.TestCase):
    """A harvester killed (SIGKILL) at each hook leaves the session whole at its source or at its destination; the
    next run finishes it (one catalog row) and the run after that changes nothing."""

    @classmethod
    def setUpClass(cls):
        cls.template = temp_dir("dd6_kill_tpl_")
        sha = write_pin(cls.template)
        cls.src_rel = os.path.relpath(make_session(cls.template, pin_sha=sha), cls.template)
        cls.raw = snapshot(os.path.join(cls.template, cls.src_rel))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.template, ignore_errors=True)

    def test_kill_at_each_hook(self):
        for hook in harvest.HOOKS:
            with self.subTest(hook=hook):
                root = temp_dir("dd6_kill_")
                try:
                    os.rmdir(root)
                    shutil.copytree(self.template, root)
                    src = os.path.join(root, self.src_rel)
                    dest = dest_of(root, WB_ID)
                    proc = run_subprocess(root, hook)
                    self.assertEqual(proc.returncode, -signal.SIGKILL, proc.stdout + proc.stderr)
                    places = [p for p in (src, dest) if os.path.isdir(p)]
                    self.assertEqual(len(places), 1, (hook, places))
                    have = snapshot(places[0])
                    for rel, digest in self.raw.items():
                        self.assertEqual(have.get(rel), digest, (hook, rel))
                    self.assertEqual(places[0] == dest, hook in ("after_move", "after_catalog"), hook)
                    code, out = run(root)
                    self.assertEqual(code, 0, (hook, out))
                    self.assertFalse(os.path.exists(src))
                    verify_sums_independently(self, dest)
                    rows = catalog(root)
                    self.assertEqual([r["session_id"] for r in rows], [WB_ID], hook)
                    doc = read_json(os.path.join(root, "manifests", "harvest_runs",
                                                 "harvest_%s.json" % rows[0]["run_id"]))
                    self.assertEqual([s["session_id"] for s in doc["sessions"]], [WB_ID], hook)
                    before = snapshot(root)
                    code, out = run(root)
                    self.assertEqual((code, snapshot(root)), (0, before), (hook, out))
                finally:
                    shutil.rmtree(root, ignore_errors=True)


class EScratchTest(unittest.TestCase):
    def test_workbench_harvest_on_e(self):
        """The Workbench harvest on E: (drvfs rename), only in a scratch directory the caller names."""
        scratch = os.environ.get("DC_TEST_E_SCRATCH")
        if not scratch or not scratch_ok(scratch) or not os.path.realpath(scratch).startswith("/mnt/e/"):
            self.skipTest("set DC_TEST_E_SCRATCH to an existing directory under /mnt/e/ (outside /mnt/e/DriveDiT) "
                          "to run this test")
        root = tempfile.mkdtemp(prefix="dd6_e_", dir=os.path.realpath(scratch))
        try:
            src = make_session(root, pin_sha=write_pin(root))
            raw = snapshot(src)
            code, out = run(root)
            self.assertEqual(code, 0, out)
            dest = dest_of(root, WB_ID)
            self.assertFalse(os.path.exists(src))
            verify_sums_independently(self, dest)
            have = snapshot(dest)
            self.assertTrue(all(have.get(rel) == digest for rel, digest in raw.items()))
            before = snapshot(root)
            self.assertEqual(run(root)[0], 0)
            self.assertEqual(snapshot(root), before)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
