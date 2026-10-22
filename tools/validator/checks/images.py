"""C10 (image presence), C11 (integrity), C12 (content), C13a (fiducial pairing) and C13b (motion lag).

ImageCollector.visit is the reader's image_visitor: it sees each readable image once, in slot order, and keeps only
small per-image features (luma statistics, the difference to the previous image, the horizontal shift of the upper
core between consecutive slots) plus running per-pixel moments of the moving frames for the overlay test, so a
30-minute session is never held in memory.
"""

import hashlib
import math
import os
import re

import numpy

from .base import FAIL, HARD, INFO, NA, NOT_EVALUATED, PASS, first, percentile, result, unreadable

LUMA = numpy.array([0.299, 0.587, 0.114], dtype=numpy.float32)
UPPER_FRAC = 0.4            # C13b: the upper part of the core (horizon and above) carries the yaw flow
STALE_SPEED_MPS = 2.0       # r1 C12: stale frames count only while moving faster than this
STALE_MAD = 0.5 / 255.0     # r1 C12: mean absolute difference below this is a stale frame
OVERLAY_MIN_FRAMES = 200    # r1 C12: overlays are judged over at least 200 moving frames
OVERLAY_STD = 1.0           # r1 C12: per-pixel temporal std below 1 (of 255) is static
SAVED_RE = re.compile(r"Screenshot saved to (\S+)")


def profile_shift(p0, p1, max_shift):
    """Sub-pixel d with p1[u + d] ~ p0[u] (normalised cross-correlation of two 1-D profiles, parabolic peak);
    returns (d, peak) or (None, None)."""
    a = p0 - p0.mean()
    b = p1 - p1.mean()
    n = len(a)
    scores = []
    for d in range(-max_shift, max_shift + 1):
        x, y = (a[:n - d], b[d:]) if d >= 0 else (a[-d:], b[:n + d])
        den = math.sqrt(float(numpy.dot(x, x)) * float(numpy.dot(y, y)))
        scores.append(float(numpy.dot(x, y)) / den if den > 0 else -1.0)
    i = int(numpy.argmax(scores))
    if i in (0, len(scores) - 1):
        return None, None
    c0, c1, c2 = scores[i - 1], scores[i], scores[i + 1]
    den = c0 - 2.0 * c1 + c2
    frac = 0.5 * (c0 - c2) / den if den != 0 else 0.0
    return i - max_shift + frac, c1


class ImageCollector:
    """Per-image features for C11-C13b, filled by the reader's image_visitor hook."""

    def __init__(self):
        self.features = {}          # slot_id -> dict
        self.shifts = []            # (slot_prev, slot, shift_px, ncc_peak)
        self.overlay_n = 0
        self._sum = None
        self._sq = None
        self._prev = None           # (slot_id, core luma float32, profile, sha256 of the core)

    def visit(self, session, row, rgb):
        band = int(session.meta_get("rig.band_px", 0) or 0)
        core = rgb[:, band:rgb.shape[1] - band, :3]
        luma = core.astype(numpy.float32) @ LUMA
        speed = sun = None
        if row.get("hf_index") is not None:
            t = session.hf_tel[row["hf_index"]]
            if t is not None:
                tel = session.streams["telemetry"]
                speed = abs(float(tel.col("vel_local_z")[t]))
                sun = tel.col("sun_elev_deg")[t]
        upper = luma[:max(int(luma.shape[0] * UPPER_FRAC), 1)]
        profile = upper.mean(axis=0)
        sha = hashlib.sha256(numpy.ascontiguousarray(core).tobytes()).hexdigest()
        feat = {"luma_mean": float(luma.mean()), "luma_std": float(luma.std()), "speed": speed, "sun_elev": sun,
                "mad_prev": None, "same_prev": False}
        prev = self._prev
        if prev is not None and prev[0] == row["slot_id"] - 1 and prev[1].shape == luma.shape:
            feat["mad_prev"] = float(numpy.abs(luma - prev[1]).mean()) / 255.0
            feat["same_prev"] = prev[3] == sha
            d, peak = profile_shift(prev[2], profile, max(8, profile.shape[0] // 8))
            if d is not None:
                self.shifts.append((prev[0], row["slot_id"], d, peak))
        if speed is not None and speed > STALE_SPEED_MPS:
            data = luma.astype(numpy.float64)
            if self._sum is None or self._sum.shape != data.shape:
                self._sum = numpy.zeros_like(data)
                self._sq = numpy.zeros_like(data)
                self.overlay_n = 0
            self._sum += data
            self._sq += data * data
            self.overlay_n += 1
        self.features[row["slot_id"]] = feat
        self._prev = (row["slot_id"], luma, profile, sha)

    def static_regions(self, min_area):
        """Connected regions (8-neighbour) of the moving-frame std map below OVERLAY_STD with at least min_area
        pixels: list of areas."""
        import cv2
        n = self.overlay_n
        mean = self._sum / n
        std = numpy.sqrt(numpy.maximum(self._sq / n - mean * mean, 0.0))
        mask = (std < OVERLAY_STD).astype(numpy.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)
                if stats[i, cv2.CC_STAT_AREA] >= min_area]


def _present(row):
    return row["issue_status"] == "issued" and row["file_exists"] == 1 and row["png_w"] is not None


def check_c10(ctx):
    """Image presence per requested slot (r2 5.1 C10): a slot is missing when it was skipped_busy or make_failed or
    its file is absent or unreadable. Missing share <= C10.T0.max_missing_frac in T0 trials, else
    C10.default.max_missing_frac; longest run of missing slots <= C10.max_missing_run; session_end.json images
    counts agree with the img rows."""
    if ctx.load_error:
        return unreadable(ctx, "C10")
    s = ctx.session
    thr = ctx.thr("C10")
    limit = (thr.get("T0") if ctx.tier == "T0" else thr.get("default"))["max_missing_frac"]
    rows = s.images
    if not rows:
        return result(ctx, "C10", HARD, FAIL, None, limit, "no image slot rows")
    missing = [not _present(r) for r in rows]
    run = longest = 0
    for m in missing:
        run = run + 1 if m else 0
        longest = max(longest, run)
    frac = sum(missing) / float(len(rows))
    issues = []
    if frac > limit:
        issues.append("missing %d of %d slots (%.4f > %s)" % (sum(missing), len(rows), frac, limit))
    if longest > thr.get("max_missing_run", 2):
        issues.append("longest missing run %d > %d" % (longest, thr.get("max_missing_run", 2)))
    images = s.end_get("images") if s.end is not None else None
    if isinstance(images, dict):
        status = s.streams["img"].col("issue_status")
        have = {"requested": len(rows), "issued": status.count("issued"),
                "skipped_busy": status.count("skipped_busy"), "make_failed": status.count("make_failed")}
        for key, value in have.items():
            if images.get(key) != value:
                issues.append("session_end images.%s=%r, img rows %d" % (key, images.get(key), value))
    value = {"missing": sum(missing), "slots": len(rows), "missing_frac": frac, "longest_run": longest}
    return result(ctx, "C10", HARD, FAIL if issues else PASS, value,
                  {"max_missing_frac": limit, "max_missing_run": thr.get("max_missing_run"),
                   "target_missing_frac": thr.get("target_missing_frac")},
                  first(issues) if issues else "%d of %d slots present" % (len(rows) - sum(missing), len(rows)))


def _log_lines(ctx):
    """Lines of every *.log file in the engine-log directory (--engine-logs, default <session>/_engine_logs), or
    None when there is none."""
    ldir = getattr(ctx.options, "engine_logs", None)
    if not ldir and ctx.session is not None:
        candidate = os.path.join(ctx.session.path, "_engine_logs")
        ldir = candidate if os.path.isdir(candidate) else None
    if not ldir or not os.path.isdir(ldir):
        return None, None
    files = {}
    for name in sorted(os.listdir(ldir)):
        path = os.path.join(ldir, name)
        if name.endswith(".log") and os.path.isfile(path):
            with open(path, "rb") as f:
                files[name] = f.read().decode("utf-8", errors="replace").splitlines()
    return ldir, files


def check_c11(ctx):
    """Image integrity (r2 5.1 C11): every issued file decodes with the size of rig.capture_wh (constant over the
    session); the client class is exactly C11.client_png_wh with core C11.core_wh; capture width = core width +
    2 x band_px with band_px >= C11.band_px_min and both fiducial bands inside the side bands; Workbench sessions
    are tagged exe=wb with source_class wb_viewport (clients cl/cld with client_1408x720); no image without
    fiducial geometry (out_of_bounds); the files in frames/ equal the issued slots and the engine log's
    "Screenshot saved to" lines for this session."""
    if ctx.load_error:
        return unreadable(ctx, "C11")
    s = ctx.session
    thr = ctx.thr("C11")
    fid = ctx.reader.fiducial
    issues = []
    cap = s.meta_get("rig.capture_wh")
    core = s.meta_get("rig.core_wh")
    band = s.meta_get("rig.band_px")
    source = s.meta_get("rig.source_class")
    exe = s.meta_get("exe.kind")
    if (exe == "wb") != (source == "wb_viewport"):
        issues.append("exe.kind %r with rig.source_class %r" % (exe, source))
    if source == "client_1408x720":
        if cap != thr.get("client_png_wh") or core != thr.get("core_wh"):
            issues.append("client capture_wh %r / core_wh %r, expected %r / %r" % (cap, core, thr.get("client_png_wh"),
                                                                                 thr.get("core_wh")))
    if not (isinstance(cap, list) and isinstance(core, list) and isinstance(band, int) and len(cap) == 2):
        issues.append("rig.capture_wh / core_wh / band_px unusable")
    else:
        if cap[0] != core[0] + 2 * band or cap[1] != core[1]:
            issues.append("capture_wh %r is not core_wh %r plus two %d px bands" % (cap, core, band))
        if band < thr.get("band_px_min", 64):
            issues.append("band_px %d < %d" % (band, thr.get("band_px_min", 64)))
        geometry = fid.layout(cap[0], cap[1])
        left_x = geometry["left"][0][0]
        right_x = geometry["right"][0][0]
        if left_x + fid.PATCH_PX > band or right_x < cap[0] - band or geometry["left"][0][1] < 0:
            issues.append("fiducial patches do not fit the %d px bands of a %dx%d image" % (band, cap[0], cap[1]))
    unreadable_n = sum(1 for r in s.images if r["issue_status"] == "issued" and r["file"] and not r["file_exists"]
                       and os.path.isfile(os.path.join(s.path, r["file"])))
    if unreadable_n:
        issues.append("%d image file(s) do not decode" % unreadable_n)
    sizes = set((r["png_w"], r["png_h"]) for r in s.images if r["png_w"] is not None)
    if isinstance(cap, list) and sizes - {tuple(cap)}:
        issues.append("image sizes %s differ from rig.capture_wh %r" % (sorted(sizes - {tuple(cap)}), cap))
    oob = sum(1 for r in s.images if r["fid_status"] == fid.STATUS_OUT_OF_BOUNDS)
    if oob:
        issues.append("%d image(s) without room for the fiducial bands" % oob)
    fdir = os.path.join(s.path, s.schema.frames_dir)
    on_disk = set(n for n in os.listdir(fdir) if os.path.isfile(os.path.join(fdir, n))) if os.path.isdir(fdir) \
        else set()
    referenced = set(os.path.basename(r["file"]) for r in s.images if r["file"])
    extra = sorted(on_disk - referenced)
    if extra:
        issues.append("%d file(s) in frames/ without an img row (%s)" % (len(extra), first(extra, 2)))
    ldir, logs = _log_lines(ctx)
    saved = None
    if logs is not None:
        marker = "/%s/" % s.session_id
        saved = sum(1 for lines in logs.values() for line in lines
                    if "Screenshot saved to" in line and marker in line)
        if saved != len(on_disk):
            issues.append("engine log: %d 'Screenshot saved to' line(s) for this session, %d file(s) in frames/"
                          % (saved, len(on_disk)))
    value = {"files": len(on_disk), "sizes": sorted(sizes), "screenshot_saved_lines": saved}
    detail = first(issues) if issues else "%d files, %s, bands ok%s" % (
        len(on_disk), "x".join(str(v) for v in cap) if isinstance(cap, list) else cap,
        "" if saved is not None else "; Screenshot-saved count not compared (no engine logs)")
    return result(ctx, "C11", HARD, FAIL if issues else PASS, value, thr, detail)


def check_c12(ctx):
    """Image content (r1 [TL] 5.1 C12 with the r2 thresholds): in the day bin (sun_elev_deg > 0 at capture, or
    unknown) every image's core luma mean lies in C12.luma_mean_day and its std >= C12.luma_std_min (night bins get
    thresholds at T2: reported only); stale frames (identical, or mean abs difference < 0.5/255 to the previous
    slot, while faster than 2 m/s) <= C12.stale_max_frac; overlays: no static region (temporal std < 1 over >= 200
    moving frames) of at least C12.overlay_min_region_frac of the image outside the fiducial bands."""
    if ctx.load_error:
        return unreadable(ctx, "C12")
    if ctx.images is None or not ctx.images.features:
        return result(ctx, "C12", HARD, NOT_EVALUATED, None, ctx.thr("C12"), "no readable image")
    thr = ctx.thr("C12")
    feats = ctx.images.features
    lo, hi = thr.get("luma_mean_day", [8, 247])
    day = [f for f in feats.values() if f["sun_elev"] is None or f["sun_elev"] > 0]
    night = len(feats) - len(day)
    bad_luma = [f for f in day if not (lo <= f["luma_mean"] <= hi) or f["luma_std"] < thr.get("luma_std_min", 4)]
    moving = [f for f in feats.values() if f["speed"] is not None and f["speed"] > STALE_SPEED_MPS
              and f["mad_prev"] is not None]
    stale = [f for f in moving if f["same_prev"] or f["mad_prev"] < STALE_MAD]
    stale_frac = len(stale) / float(len(moving)) if moving else 0.0
    issues = []
    if bad_luma:
        issues.append("%d day image(s) outside luma mean %s / std >= %s" % (len(bad_luma), [lo, hi],
                                                                            thr.get("luma_std_min")))
    if stale_frac > thr.get("stale_max_frac", 0.002):
        issues.append("stale frames %.4f > %s (%d of %d moving)" % (stale_frac, thr.get("stale_max_frac"),
                                                                   len(stale), len(moving)))
    skipped = []
    regions = None
    if ctx.images.overlay_n >= OVERLAY_MIN_FRAMES:
        cap = ctx.session.meta_get("rig.capture_wh") or [0, 0]
        min_area = thr.get("overlay_min_region_frac", 0.005) * cap[0] * cap[1]
        regions = ctx.images.static_regions(max(int(math.ceil(min_area)), 1))
        if regions:
            issues.append("%d static region(s) over %d moving frames (areas %s px)" % (
                len(regions), ctx.images.overlay_n, first(sorted(regions, reverse=True), 3)))
    else:
        skipped.append("overlay (%d moving frames < %d)" % (ctx.images.overlay_n, OVERLAY_MIN_FRAMES))
    value = {"images": len(feats), "night_images": night, "stale_frac": stale_frac, "static_regions": regions}
    detail = first(issues) if issues else "%d images: luma ok, %d stale of %d moving%s" % (
        len(feats), len(stale), len(moving), "" if regions is not None else "; overlay not judged (%d moving "
        "frames)" % ctx.images.overlay_n)
    return result(ctx, "C12", HARD, FAIL if issues else PASS, value, thr, detail, not_evaluated=skipped)


def check_c13a(ctx):
    """Fiducial pairing (r2 5.1 C13a): of the readable images, >= C13a.min_decode_frac decode in both bands with
    agreeing 18-bit codes (C13a.code_bits); every decoded captured_render_frame (the code unwrapped against
    issue_render_frame) joins an hf row exactly (0 join failures). The capture offset (captured - issued frame) is
    reported."""
    if ctx.load_error:
        return unreadable(ctx, "C13a")
    s = ctx.session
    thr = ctx.thr("C13a")
    fid = ctx.reader.fiducial
    readable = [r for r in s.images if r["png_w"] is not None]
    if not readable:
        return result(ctx, "C13a", HARD, NOT_EVALUATED, None, thr, "no readable image")
    decoded = [r for r in readable if r["fid_ok"]]
    frac = len(decoded) / float(len(readable))
    unjoined = [r["slot_id"] for r in decoded if r["hf_index"] is None]
    offsets = [r["captured_render_frame"] - r["issue_render_frame"] for r in decoded]
    issues = []
    if thr.get("code_bits") != fid.CODE_BITS:
        issues.append("decoder has %d code bits, thresholds %r" % (fid.CODE_BITS, thr.get("code_bits")))
    if frac < thr.get("min_decode_frac", 0.995):
        issues.append("decoded %d of %d (%.4f < %s)" % (len(decoded), len(readable), frac,
                                                       thr.get("min_decode_frac")))
    if unjoined:
        issues.append("%d decoded image(s) whose captured frame has no hf row (slots %s)" % (len(unjoined),
                                                                                           first(unjoined, 5)))
    margins = [r["fid_margin"] for r in decoded if r["fid_margin"] is not None]
    value = {"decode_frac": frac, "join_failures": len(unjoined), "offset_p50": percentile(offsets, 50),
             "offset_p99": percentile(offsets, 99), "negative_offsets": sum(1 for o in offsets if o < 0),
             "min_margin": min(margins) if margins else None}
    return result(ctx, "C13a", HARD, FAIL if issues else PASS, value, thr,
                  first(issues) if issues else "%d of %d decoded and joined; capture offset p50 %s p99 %s frames"
                  % (len(decoded), len(readable), value["offset_p50"], value["offset_p99"]))


def check_c13b(ctx):
    """Motion lag (r2 5.1 C13b), independent of the fiducial: the horizontal shift of the upper core between
    consecutive slots (1-D normalised cross-correlation of column means) is fitted against the camera heading change
    hf cam_fwd(captured + L) - cam_fwd(previous captured + L) for L in E06A.shift_range; the best lag (smallest
    residual) must satisfy |L| <= C13b.max_abs_lag_frames. HARD when the session's yaw-rate std >=
    C13b.min_yaw_rate_std_rps, else INFO."""
    if ctx.load_error:
        return unreadable(ctx, "C13b")
    s = ctx.session
    thr = ctx.thr("C13b")
    yaw = s.telemetry_export["yaw_rate_rps"]
    yaw = yaw[~numpy.isnan(yaw)]
    yaw_std = float(yaw.std()) if len(yaw) else 0.0
    severity = HARD if yaw_std >= thr.get("min_yaw_rate_std_rps", 0.05) else INFO
    lo, hi = ctx.thr("E06A", "shift_range", [-3, 3])
    if ctx.images is None:
        return result(ctx, "C13b", severity, NOT_EVALUATED, None, thr, "no image features")
    hf = s.streams["hf"]
    heading = {f: math.atan2(x, z) for f, x, z in zip(hf.col("render_frame"), hf.col("cam_fwd_x"),
                                                       hf.col("cam_fwd_z"))}
    captured = {r["slot_id"]: r["captured_render_frame"] for r in s.images if r["fid_ok"]}
    pairs = []
    for prev, slot, shift, peak in ctx.images.shifts:
        if peak is None or peak < 0.5 or prev not in captured or slot not in captured:
            continue
        a, b = captured[prev], captured[slot]
        if all((a + L) in heading and (b + L) in heading for L in range(lo, hi + 1)):
            pairs.append((a, b, shift))
    if len(pairs) < 10:
        return result(ctx, "C13b", severity, NOT_EVALUATED if severity == HARD else NA,
                      {"pairs": len(pairs), "yaw_rate_std": yaw_std}, thr,
                      "%d usable consecutive image pairs (< 10)" % len(pairs))
    residuals = {}
    gains = {}
    shifts = numpy.array([p[2] for p in pairs])
    for L in range(lo, hi + 1):
        dpsi = numpy.array([math.atan2(math.sin(heading[b + L] - heading[a + L]),
                                       math.cos(heading[b + L] - heading[a + L])) for a, b, _ in pairs])
        den = float(numpy.dot(dpsi, dpsi))
        if den <= 0:
            continue
        gain = float(numpy.dot(dpsi, shifts)) / den
        residuals[L] = float(numpy.sqrt(numpy.mean((shifts - gain * dpsi) ** 2)))
        gains[L] = gain
    if not residuals:
        return result(ctx, "C13b", severity, NOT_EVALUATED if severity == HARD else NA, None, thr,
                      "no heading change between images")
    best = min(residuals, key=lambda L: (residuals[L], abs(L)))
    ok = abs(best) <= thr.get("max_abs_lag_frames", 1)
    value = {"lag_frames": best, "residual_px": {str(L): round(r, 4) for L, r in sorted(residuals.items())},
             "px_per_rad": gains[best], "pairs": len(pairs), "yaw_rate_std": yaw_std}
    return result(ctx, "C13b", severity, PASS if ok else FAIL, value, thr,
                  "best lag %+d frames (residual %.3f px over %d pairs; yaw-rate std %.3f rad/s)"
                  % (best, residuals[best], len(pairs), yaw_std))
