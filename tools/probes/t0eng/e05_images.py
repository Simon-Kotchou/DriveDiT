#!/usr/bin/env python3
"""E-05 analysis: rig R1 camera with 10 Hz System.MakeScreenshot requests (r2 4.4 T0-ENG core; DD-4-probe).

    python3 -I -B tools/probes/t0eng/e05_images.py <e05 play_dir> [--log <console.log>]

Input: one play directory of probe e05 (DC_ProbeImages): run_header.txt, run_footer.txt, e05_requests.csv,
e05_frames.csv and frames/. With --log, the engine console log of the same Workbench session (read only).

r2 4.4 criterion of E-05, printed as checks:
  * rig_camera_current  r2 4.4 "R1 camera ...: attached, current, vFOV": cam_current and cam_attached are 1 on every
                        request MakeScreenshot was called for (issued and make_failed); otherwise the files may show
                        another camera (the ManualCamera)
  * rig_vfov            rig.vfov_read is within 0.1 deg of params.vertical_fov_deg
  * png_dims            the image dimensions are logged and equal for every file
  * missing_le_1pct     slots without a file <= 1% (skipped, make_failed and timed-out slots without a file count)
  * saved_count_eq_files the "Screenshot saved" lines between the START and END markers of this play equal the files
  * fps_p5_ge_30        p5 of the per-frame FPS (1 / frame time slice) >= 30, segments reset and settle excluded
and as INFO: the request->file latency distribution (in-engine done_tick_ms - issue_tick_ms), raw MB per frame, the
engine-written name (TBD-4), with params.drive=1 the shuttle ("drive": max veh_speed_kmh over the drive frames, the
gearbox, ForceEnableSimulation and its e01 decision, shift-policy shifts and physics wakes from the run header) and
the decision for TBD-2:
  DECISION image_hz=10 when missing <= 1% and latency p95 <= 100 ms, else image_hz=5 (fallback).

run_footer (probe_common.Verdict.check_footer): a play that did not end normally is analysed as far as possible
and its result is at best INCOMPLETE (FAIL when the footer says failed).
Exit codes: 0 PASS, 1 FAIL, 3 INCOMPLETE, 2 usage or input error. The script only reads.
"""

import argparse
import os
import struct
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import probe_common  # noqa: E402  (shared reader and verdict printer, tools/probes/t0eng/probe_common.py)

LABEL = "E-05 R1 camera and 10 Hz screenshots"
PROBE = "e05"
REQUEST_COLUMNS = (
    "slot_id", "requested_world_ms", "issue_status", "issue_render_frame", "issue_tick_ms", "path", "make_ok",
    "cam_current", "cam_attached", "done_status", "done_tick_ms", "found_suffix",
)
ATTEMPTED = ("issued", "make_failed")       # issue_status of a request MakeScreenshot was called for
VFOV_TOL_DEG = 0.1
FRAME_COLUMNS = ("render_frame", "fps", "segment")
MISSING_MAX_PCT = 1.0
FPS_P5_MIN = 30.0
LATENCY_P95_MAX_MS_10HZ = 100.0
EXCLUDED_FPS_SEGMENTS = ("reset", "settle")
SUFFIX_NOEXT = "noext"           # found_suffix of a file written exactly at the extension-less request path
SAVED_MARK = "Screenshot saved"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def image_size(path):
    """(width, height) from a PNG IHDR or a BMP header, None for anything else or an unreadable file."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(26)
    except OSError:
        return None
    if head[:8] == PNG_SIGNATURE and len(head) >= 24 and head[12:16] == b"IHDR":
        return struct.unpack(">II", head[16:24])
    if head[:2] == b"BM" and len(head) >= 26:
        width, height = struct.unpack("<ii", head[18:26])
        return width, abs(height)
    return None


class FrameFiles:
    """The files under <play>/frames, indexed by request name (f_NNNNNNNN, the part before the first '.')."""

    def __init__(self, play_dir):
        self.dir = os.path.join(play_dir, "frames")
        self.by_stem = {}
        if os.path.isdir(self.dir):
            for name in sorted(os.listdir(self.dir)):
                if os.path.isfile(os.path.join(self.dir, name)):
                    self.by_stem.setdefault(name.split(".", 1)[0], []).append(name)

    def count(self):
        return sum(len(names) for names in self.by_stem.values())

    def resolve(self, rel_path, found_suffix=""):
        """(absolute path, suffix) of the file the engine wrote for a request path, or (None, "")."""
        if not rel_path:
            return None, ""
        stem = rel_path.replace("\\", "/").split("/")[-1]
        names = self.by_stem.get(stem, [])
        if found_suffix:
            wanted = stem if found_suffix == SUFFIX_NOEXT else stem + found_suffix
            if wanted in names:
                return os.path.join(self.dir, wanted), found_suffix
        if not names:
            return None, ""
        name = names[0]
        suffix = name[len(stem):] or SUFFIX_NOEXT
        return os.path.join(self.dir, name), suffix


def count_saved_lines(log_path, probe, play_dir_value):
    """Lines containing "Screenshot saved" between this play's START marker and the following END marker.

    Returns (count, detail, saved_names); count is None when the START marker is not in the log.
    """
    start_mark = "[DC_Probe %s] START play_dir=%s" % (probe, play_dir_value)
    end_mark = "[DC_Probe %s] END reason=" % probe
    count = 0
    started = False
    ended = False
    names = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not started:
                if line.rstrip("\r\n").endswith(start_mark) or (start_mark + " ") in line:
                    started = True
                continue
            if end_mark in line:
                ended = True
                break
            if SAVED_MARK in line:
                count += 1
                names.append(line.rstrip("\r\n").replace("\\", "/").split("/")[-1])
    if not started:
        return None, "START marker '%s' not in %s" % (start_mark, log_path), names
    detail = "between START and END" if ended else "from START to the end of the log (no END marker)"
    return count, detail, names


def latency_text(values):
    if not values:
        return "no completed request"
    return "p50 %.1f / p95 %.1f / p99 %.1f / max %.1f ms over %d requests" % (
        probe_common.percentile(values, 50), probe_common.percentile(values, 95),
        probe_common.percentile(values, 99), max(values), len(values))


def drive_text(header, frames):
    """INFO drive of a shuttle play (params.drive=1): max veh_speed_kmh over the frames of segment drive, then the run
    header's vehicle.gearbox, drive.force_sim / drive.force_sim_e01, gear_policy.shifts, physics_wake.* and
    drive.stuck_forces (the last three are appended at the probe stop; a drive.force_sim appended there wins over the
    start value, so a stuck-guard latch shows as force_sim 1). None for a static capture."""
    if header.get("params.drive") != "1":
        return None
    speeds = [value for row, value in zip(frames, probe_common.floats(frames, "veh_speed_kmh"))
              if value is not None and row.get("segment") == "drive"]
    moved = "no drive frame with veh_speed_kmh"
    if speeds:
        moved = "max veh_speed_kmh %.2f over %d drive frames" % (max(speeds), len(speeds))
    text = "%s; gearbox %s; force_sim %s (e01 decision %s); gear policy shifts %s; physics wake %s calls, %s with " \
           "the body asleep before" % (
               moved, header.get("vehicle.gearbox", "n/a"), header.get("drive.force_sim", "n/a"),
               header.get("drive.force_sim_e01", "n/a"), header.get("gear_policy.shifts", "n/a"),
               header.get("physics_wake.calls", "n/a"), header.get("physics_wake.asleep_before", "n/a"))
    if "physics_wake.sleep_wakes" in header or "drive.stuck_forces" in header:
        # The review fix (R-4): re-wakes of an asleep body while driving, and the stuck guard's ForceEnableSimulation
        text += "; %s sleep re-wakes while driving; stuck force latches %s" % (
            header.get("physics_wake.sleep_wakes", "n/a"), header.get("drive.stuck_forces", "n/a"))
    return text


def evaluate(play_dir, log_path=None):
    """Builds the E-05 verdict of one e05 play directory (raises probe_common.InputError on bad input)."""
    play = probe_common.Play(play_dir)
    probe = play.header.get("probe", "")
    if probe != PROBE:
        raise probe_common.InputError("%s is a '%s' play, not %s" % (play_dir, probe, PROBE))

    requests = play.csv("e05_requests", required=REQUEST_COLUMNS)
    frames = play.csv("e05_frames", required=FRAME_COLUMNS)
    files = FrameFiles(play.dir)

    verdict = probe_common.Verdict(LABEL, play.dir)

    statuses = {}
    present = []
    suffixes = {}
    latencies = []
    for row in requests:
        statuses[row["issue_status"]] = statuses.get(row["issue_status"], 0) + 1
        path, suffix = files.resolve(row["path"], row.get("found_suffix", ""))
        if path:
            present.append(path)
            suffixes[suffix] = suffixes.get(suffix, 0) + 1
        if row["done_status"] == "done" and row["done_tick_ms"] and row["issue_tick_ms"]:
            latencies.append(float(row["done_tick_ms"]) - float(row["issue_tick_ms"]))

    attempted = [row for row in requests if row["issue_status"] in ATTEMPTED]
    if not attempted:
        verdict.check("rig_camera_current", None, "no request was issued")
    else:
        off = [row["slot_id"] for row in attempted if row["cam_current"] != "1" or row["cam_attached"] != "1"]
        detail = "%d of %d issued requests with cam_current or cam_attached 0" % (len(off), len(attempted))
        if off:
            detail += " (first slots: %s)" % ", ".join(off[:5])
        verdict.check("rig_camera_current", not off, detail)

    vfov_read = probe_common.to_float(play.header.get("rig.vfov_read"))
    vfov_set = probe_common.to_float(play.header.get("params.vertical_fov_deg"))
    if vfov_read is None or vfov_set is None:
        verdict.check("rig_vfov", None, "rig.vfov_read=%s, params.vertical_fov_deg=%s" % (
            play.header.get("rig.vfov_read", "(absent)"), play.header.get("params.vertical_fov_deg", "(absent)")))
    else:
        verdict.check("rig_vfov", abs(vfov_read - vfov_set) <= VFOV_TOL_DEG, "read %.4f deg, set %.4f deg (limit "
                      "%.1f deg)" % (vfov_read, vfov_set, VFOV_TOL_DEG))

    slots = len(requests)
    verdict.info("slots", "%d slots: %s; %d files present, %d files in frames/" % (
        slots, ", ".join("%s %d" % item for item in sorted(statuses.items())) or "none", len(present), files.count()))

    sizes = {}
    total_bytes = []
    for path in present:
        size = image_size(path)
        if size:
            sizes[size] = sizes.get(size, 0) + 1
        total_bytes.append(os.path.getsize(path))
    if not sizes:
        verdict.check("png_dims", None, "no readable PNG or BMP header among %d files" % len(present))
    else:
        text = ", ".join("%dx%d (%d files)" % (w, h, n) for (w, h), n in sorted(sizes.items()))
        verdict.check("png_dims", len(sizes) == 1, text)

    if slots == 0:
        missing_pct = None
        verdict.check("missing_le_1pct", None, "no slot in e05_requests.csv")
    else:
        missing = slots - len(present)
        missing_pct = 100.0 * missing / slots
        verdict.check("missing_le_1pct", missing_pct <= MISSING_MAX_PCT,
                      "%d of %d slots without a file (%.2f%%, limit %.1f%%)" % (missing, slots, missing_pct,
                                                                                 MISSING_MAX_PCT))

    if not log_path:
        verdict.check("saved_count_eq_files", None, "no --log given")
    else:
        if not os.path.isfile(log_path):
            raise probe_common.InputError("console log %s not found" % log_path)
        saved, where, saved_names = count_saved_lines(log_path, PROBE, play.header.get("play_dir", ""))
        if saved is None:
            verdict.check("saved_count_eq_files", None, where)
        else:
            verdict.check("saved_count_eq_files", saved == len(present),
                          "%d 'Screenshot saved' lines %s, %d files" % (saved, where, len(present)))
            engine_names = sorted({name.split(".", 1)[1] if "." in name else SUFFIX_NOEXT for name in saved_names})
            verdict.info("log_written_suffixes", ", ".join("." + s if s != SUFFIX_NOEXT else s
                                                          for s in engine_names) or "none")

    fps = [value for row, value in zip(frames, probe_common.floats(frames, "fps"))
           if value is not None and row["segment"] not in EXCLUDED_FPS_SEGMENTS]
    if not fps:
        verdict.check("fps_p5_ge_30", None, "no frame with an FPS value outside reset/settle")
    else:
        p5 = probe_common.percentile(fps, 5)
        verdict.check("fps_p5_ge_30", p5 >= FPS_P5_MIN, "FPS p5 %.1f, p50 %.1f over %d frames (limit %.0f)" % (
            p5, probe_common.percentile(fps, 50), len(fps), FPS_P5_MIN))

    verdict.info("latency", latency_text(latencies))
    if total_bytes:
        mb = [value / 1e6 for value in total_bytes]
        verdict.info("raw_mb_per_frame", "mean %.3f MB, p95 %.3f MB over %d files" % (
            sum(mb) / len(mb), probe_common.percentile(mb, 95), len(mb)))
    else:
        verdict.info("raw_mb_per_frame", "no file")
    verdict.info("engine_suffix", ", ".join("%s (%d)" % item for item in sorted(suffixes.items())) or "none found")
    drive = drive_text(play.header, frames)
    if drive is not None:
        verdict.info("drive", drive)

    p95 = None
    if latencies:
        p95 = probe_common.percentile(latencies, 95)
    if missing_pct is not None and missing_pct <= MISSING_MAX_PCT and p95 is not None \
            and p95 <= LATENCY_P95_MAX_MS_10HZ:
        decision = "image_hz=10 (missing %.2f%% <= 1%%, latency p95 %.1f ms <= 100 ms)" % (missing_pct, p95)
    else:
        decision = "image_hz=5 (fallback: missing %s, latency p95 %s)" % (
            "n/a" if missing_pct is None else "%.2f%%" % missing_pct, "n/a" if p95 is None else "%.1f ms" % p95)
    verdict.info("DECISION", decision)
    verdict.check_footer(play)
    return verdict


def parse_args(argv):
    parser = argparse.ArgumentParser(description="E-05 analysis of one e05 play directory (r2 4.4).")
    parser.add_argument("play_dir", help="$profile:DC_Probes/e05/<launch_id>/play_NN as a WSL path")
    parser.add_argument("--log", help="engine console.log of the same session (for the 'Screenshot saved' count)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return evaluate(args.play_dir, args.log).report()


if __name__ == "__main__":
    sys.exit(probe_common.run_cli(main))
