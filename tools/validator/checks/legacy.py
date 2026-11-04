"""--legacy: the dc-1.0 self-test (r1 [TL] 5): a dc-1.0 session must FAIL, for its known reasons.

dc-1.0 sessions ($profile:DrivingData/session_<n>/: telemetry.csv with one row per vehicle per tick, CRLF,
summary.txt, frames/frame_<frame_id:06d>.bmp.png) are never read as dc-2.0 data: the reader raises on them. This
module parses only what is needed to name the reasons, with the stdlib csv module:

  C01  no session.json / session_end.json: not a dc-2.0 session (schema);
  C05  telemetry.csv header against the dc-2.0 telemetry stream (missing required and unknown columns);
  C08  cadence: the tick interval of timestamp_ms per frame_id is bimodal (or there are no rows);
  C10  image presence: telemetry frame_ids without frames/frame_<id:06d>.bmp.png;
  C13a dc-1.0 images carry no fiducial.
"""

import csv
import io
import os

from .base import FAIL, HARD, NA, PASS, first, percentile

BIMODAL_DEV = 0.25          # a tick interval more than 25% away from the median is off the main mode
BIMODAL_MIN_SHARE = 0.05    # bimodal when at least 5% of the intervals are off the main mode
IMAGE_NAMES = ("frame_%06d.bmp.png", "frame_%06d.png", "frame_%06d.bmp")


def _res(check_id, status, value, threshold, detail):
    return {"id": check_id, "severity": HARD, "status": status, "value": value, "threshold": threshold,
            "detail": detail, "provisional": False}


def run_legacy(session_dir, schema, thresholds):
    out = []
    present = sorted(os.listdir(session_dir)) if os.path.isdir(session_dir) else []
    has_json = "session.json" in present
    out.append(_res("C01", FAIL if not has_json else PASS, present[:8], "session.json + session_end.json",
                    "dc-1.0 layout (%s): no session.json / session_end.json" % ", ".join(present[:6])
                    if not has_json else "session.json present"))
    tel_path = os.path.join(session_dir, "telemetry.csv")
    rows = []
    header = []
    if os.path.isfile(tel_path):
        with open(tel_path, "rb") as f:
            text = f.read().decode("ascii", errors="replace")
        reader = csv.reader(io.StringIO(text, newline=""))
        lines = list(reader)
        header = lines[0] if lines else []
        rows = [dict(zip(header, r)) for r in lines[1:] if r]
    fields = schema.fields("telemetry")
    required = [f["name"] for f in fields if f["required"]]
    known = set(f["name"] for f in fields)
    missing = [c for c in required if c not in header]
    unknown = [c for c in header if c not in known]
    if not header:
        out.append(_res("C05", FAIL, None, 0, "no telemetry.csv header"))
    else:
        out.append(_res("C05", FAIL if (missing or unknown) else PASS,
                        {"missing_required": len(missing), "unknown": len(unknown)}, 0,
                        "telemetry.csv is not the dc-2.0 telemetry stream: %d required column(s) missing (%s), %d "
                        "unknown (%s)" % (len(missing), first(missing, 4), len(unknown), first(unknown, 4))))
    ticks = {}
    for r in rows:
        try:
            ticks.setdefault(int(r["frame_id"]), float(r["timestamp_ms"]))
        except (KeyError, ValueError):
            continue
    ordered = sorted(ticks.items())
    dts = [b[1] - a[1] for a, b in zip(ordered, ordered[1:])]
    if len(dts) < 2:
        out.append(_res("C08", FAIL, {"ticks": len(ordered)}, "unimodal tick interval",
                        "%d tick(s) with frame_id and timestamp_ms: no cadence" % len(ordered)))
    else:
        med = percentile(dts, 50)
        off = [d for d in dts if med <= 0 or abs(d - med) > BIMODAL_DEV * med]
        share = len(off) / float(len(dts))
        modes = sorted(set(int(round(d / 50.0)) * 50 for d in dts))
        out.append(_res("C08", FAIL if share >= BIMODAL_MIN_SHARE else PASS,
                        {"median_dt_ms": med, "off_mode_share": share, "dt_ms_bins": modes[:8]},
                        "off-mode share < %s" % BIMODAL_MIN_SHARE,
                        "bimodal tick interval: median %.1f ms, %.1f%% of %d intervals off by > %d%% (bins %s)"
                        % (med, 100 * share, len(dts), int(100 * BIMODAL_DEV), modes[:6])
                        if share >= BIMODAL_MIN_SHARE else "tick interval unimodal (median %.1f ms)" % med))
    fdir = os.path.join(session_dir, "frames")
    files = set(os.listdir(fdir)) if os.path.isdir(fdir) else set()
    if not ordered:
        out.append(_res("C10", NA, {"files": len(files)}, None,
                        "no telemetry frame to pair with the %d image file(s)" % len(files)))
    else:
        lost = [fid for fid, _ in ordered if not any((n % fid) in files for n in IMAGE_NAMES)]
        limit = thresholds.get("C10", {}).get("T0", {}).get("max_missing_frac", 0.01)
        frac = len(lost) / float(len(ordered))
        suffix = "" if os.path.isdir(fdir) else " (no frames/ directory)"
        out.append(_res("C10", FAIL if frac > limit else PASS, {"missing": len(lost), "frames": len(ordered)},
                        limit, "%d of %d telemetry frames have no image%s" % (len(lost), len(ordered), suffix)))
    out.append(_res("C13a", FAIL, None, "fiducial v2", "dc-1.0 images carry no fiducial (no captured render frame)"))
    return out
