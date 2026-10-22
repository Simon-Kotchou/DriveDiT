#!/usr/bin/env python3
"""E-06A: marker reprojection alignment of one e06a play (r2 3.5 and 4.4 T0-ENG core; DD-4-probe).

    python3 -I -B tools/validator/e06a_reproject.py <e06a play_dir> [--kmax 3] [--rms-px 2.0] [--margin 0.5]
        [--gate-px 20] [--color FF00FF] [--color-tol 60] [--min-motion-px 2] [--min-area 3] [--fps-tol 0.15]

Input: one play directory of probe e06a (DC_ProbeMarkers): run_header.txt (rig.vfov_read, fiducial.*),
e06a_markers.csv, e06a_markers_end.csv, e06a_frames.csv (camera transform per render frame), e06a_requests.csv
(fps_setting, segment), e06a_proj.csv and frames/.

Method:
  * camera model: pinhole, fy = fx = (H / 2) / tan(vfov / 2) with rig.vfov_read, cx = W / 2, cy = H / 2, in
    continuous pixel coordinates (pixel i covers [i, i + 1)); a world point X has camera coordinates
    x = (X - pos) . right, y = (X - pos) . up, z = (X - pos) . fwd and projects to u = cx + fx x / z,
    v = cy - fy y / z. The camera rows are the e06a_frames.csv cam_* columns (CameraBase.GetWorldCameraTransform);
  * model check (INFO, FAIL "camera_model" when p95 > 1 px): the model against the engine's own
    ProjWorldToScreenNative in e06a_proj.csv;
  * per image: decode fiducial v2 (data/dc2/fiducial.py, loaded by file path), F = unwrap(code,
    issue_render_frame); detect the marker blobs (colour within --color-tol of --color, outside the fiducial bands,
    cv2.connectedComponentsWithStats, centroid, area >= --min-area); for k in -kmax..kmax project every marker
    (moved prefab markers excluded) with the camera of frame F + k; a blob's residual at k is the distance to the
    nearest projection. A blob counts when its smallest residual over all k is within --gate-px, and then it counts
    at every k (residuals capped at 5 x gate), so every k is scored on the same blobs. Only images whose markers
    move >= --min-motion-px between F - 1 and F + 1 count (a static camera cannot discriminate k). Segments
    slalom, backlog, brake and curve are used; reset and settle are not;
  * per condition (each fps_setting, backlog included): RMS(k).

r2 PASS per condition: RMS(0) <= 2 px, k = 0 is the unique minimum, and RMS(0) <= 0.5 x min(RMS(-1), RMS(+1)),
at 60 and at 30 FPS (every value of params.fps_phases). A condition whose measured FPS p50 is not within --fps-tol of
its setting is INCOMPLETE (the setting did not take); no counted blob is INCOMPLETE (switch m_eMarkerKind to PREFAB).
Exit codes: 0 PASS, 1 FAIL, 3 INCOMPLETE, 2 usage or input error. The script only reads.

Library use (tests): project(markers, cam_rows, vfov_deg, width, height) and evaluate(observations, cam_by_frame,
markers, vfov_deg, width, height, kmax, gate_px, min_motion_px=2.0) -> dict, judge(result, ...) and
detect_blobs(image, color, tol, exclude, min_area).
"""

import argparse
import importlib.util
import math
import os
import sys

import numpy

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools", "probes", "t0eng"))
import probe_common  # noqa: E402  (shared reader and verdict printer)
import e05_images  # noqa: E402  (FrameFiles: request path -> engine-written file)

FIDUCIAL_PATH = os.path.join(REPO, "data", "dc2", "fiducial.py")
LABEL = "E-06A marker reprojection"
PROBE = "e06a"
USED_SEGMENTS = ("slalom", "backlog", "brake", "curve")
RESIDUAL_CAP_GATES = 5.0
MODEL_P95_MAX_PX = 1.0
CAM_PREFIX = "cam"
AXES = ("right", "up", "fwd")


def load_fiducial():
    """data/dc2/fiducial.py by file path (importing the data package would load torch)."""
    spec = importlib.util.spec_from_file_location("dc2_fiducial", FIDUCIAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------------------------
# camera model


def focal_px(vfov_deg, height):
    return (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)


def project(markers, cam_rows, vfov_deg, width, height):
    """Pinhole projection of world points.

    markers: (N, 3) world positions; cam_rows: (4, 3) rows right, up, forward, position (the Enforce vector mat[4]
    order). Returns (uv, depth): (N, 2) continuous pixel coordinates (NaN behind the camera) and (N,) z along the
    forward axis.
    """
    markers = numpy.asarray(markers, dtype=float).reshape(-1, 3)
    cam = numpy.asarray(cam_rows, dtype=float).reshape(4, 3)
    offset = markers - cam[3]
    x = offset @ cam[0]
    y = offset @ cam[1]
    z = offset @ cam[2]
    f = focal_px(vfov_deg, height)
    uv = numpy.full((markers.shape[0], 2), numpy.nan)
    front = z > 1e-6
    uv[front, 0] = width / 2.0 + f * x[front] / z[front]
    uv[front, 1] = height / 2.0 - f * y[front] / z[front]
    return uv, z


def visible(uv, width, height):
    return (numpy.isfinite(uv[:, 0]) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
            & (uv[:, 1] >= 0) & (uv[:, 1] < height))


def cam_from_row(row, prefix=CAM_PREFIX):
    """(4, 3) camera rows right, up, fwd, pos from the 12 <prefix>_* columns of a frames CSV row."""
    rows = [[float(row["%s_%s_%s" % (prefix, axis, c)]) for c in "xyz"] for axis in AXES]
    rows.append([float(row["%s_pos_%s" % (prefix, c)]) for c in "xyz"])
    return numpy.array(rows)


# ---------------------------------------------------------------------------------------------------------------
# blobs


def parse_color(text):
    text = text.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError("colour %r is not RRGGBB" % text)
    return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))


def band_exclusion(width, header, fiducial):
    """x ranges [x0, x1) covered by the fiducial bands (patch plus its margin on both sides)."""
    patch = int(float(header.get("fiducial.patch_px", fiducial.PATCH_PX)))
    margin = int(float(header.get("fiducial.margin_px", fiducial.MARGIN_PX)))
    left_x = header.get("fiducial.left_x", "")
    right_x = header.get("fiducial.right_x", "")
    if left_x != "" and right_x != "":
        left_x, right_x = int(float(left_x)), int(float(right_x))
    else:
        left_x, right_x = margin, width - margin - patch
    return [(0, left_x + patch + margin), (max(0, right_x - margin), width)]


def detect_blobs(image, color, tol, exclude, min_area):
    """Centroids (continuous pixel coordinates) of the connected regions within tol of color, outside exclude."""
    import cv2  # only here: projection and evaluation need numpy alone

    diff = numpy.abs(image[:, :, :3].astype(numpy.int16) - numpy.array(color, dtype=numpy.int16))
    mask = (diff.max(axis=2) <= tol).astype(numpy.uint8)
    for x0, x1 in exclude:
        mask[:, max(0, x0):max(0, x1)] = 0
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            blobs.append((centroids[label][0] + 0.5, centroids[label][1] + 0.5))
    return numpy.array(blobs, dtype=float).reshape(-1, 2)


# ---------------------------------------------------------------------------------------------------------------
# evaluation


def evaluate(observations, cam_by_frame, markers, vfov_deg, width, height, kmax, gate_px, min_motion_px=2.0):
    """RMS(k) per condition.

    observations: iterable of {"condition": key, "frame": F (decoded and unwrapped), "blobs": (B, 2)};
    cam_by_frame: {render_frame: (4, 3) camera rows}; markers: (N, 3).
    Returns {condition: {"rms": {k: float or None}, "blobs": n, "images": n, "used": n, "static": n, "edge": n,
    "no_blob": n}}.
    """
    markers = numpy.asarray(markers, dtype=float).reshape(-1, 3)
    ks = list(range(-kmax, kmax + 1))
    cap = RESIDUAL_CAP_GATES * gate_px
    out = {}
    for obs in observations:
        result = out.setdefault(obs["condition"], {
            "sq": {k: [] for k in ks}, "images": 0, "used": 0, "static": 0, "edge": 0, "no_blob": 0})
        result["images"] += 1
        frame = int(obs["frame"])
        needed = [frame + k for k in ks] + [frame - 1, frame + 1]
        if any(f not in cam_by_frame for f in needed):
            result["edge"] += 1
            continue
        uv_before, _ = project(markers, cam_by_frame[frame - 1], vfov_deg, width, height)
        uv_after, _ = project(markers, cam_by_frame[frame + 1], vfov_deg, width, height)
        both = visible(uv_before, width, height) & visible(uv_after, width, height)
        if not both.any():
            result["static"] += 1
            continue
        motion = float(numpy.median(numpy.linalg.norm(uv_after[both] - uv_before[both], axis=1)))
        if motion < min_motion_px:
            result["static"] += 1
            continue
        blobs = numpy.asarray(obs["blobs"], dtype=float).reshape(-1, 2)
        if blobs.shape[0] == 0:
            result["no_blob"] += 1
            continue
        residuals = []
        for k in ks:
            uv, _ = project(markers, cam_by_frame[frame + k], vfov_deg, width, height)
            front = numpy.isfinite(uv[:, 0])
            if not front.any():
                residuals.append(numpy.full(blobs.shape[0], numpy.inf))
                continue
            distances = numpy.linalg.norm(blobs[:, None, :] - uv[None, front, :], axis=2)
            residuals.append(distances.min(axis=1))
        residuals = numpy.array(residuals)          # (len(ks), B)
        counted = residuals.min(axis=0) <= gate_px
        if not counted.any():
            result["no_blob"] += 1
            continue
        result["used"] += 1
        for index, k in enumerate(ks):
            capped = numpy.minimum(residuals[index, counted], cap)
            result["sq"][k].extend((capped ** 2).tolist())
    for result in out.values():
        result["rms"] = {k: (math.sqrt(sum(v) / len(v)) if v else None) for k, v in result["sq"].items()}
        result["blobs"] = len(result["sq"][0])
        del result["sq"]
    return out


def judge(result, rms_px=2.0, margin=0.5):
    """(ok, detail) of one condition: ok True (PASS), False (FAIL) or None (INCOMPLETE, no counted blob)."""
    rms = result["rms"]
    if not result["blobs"] or rms.get(0) is None:
        return None, "no counted blob (images %d, static %d, edge %d, without blobs %d)" % (
            result["images"], result["static"], result["edge"], result["no_blob"])
    others = {k: v for k, v in rms.items() if k != 0 and v is not None}
    best_k = min(rms, key=lambda k: (rms[k] is None, rms[k] if rms[k] is not None else 0.0))
    unique = all(rms[0] < v for v in others.values())
    neighbours = [v for k, v in others.items() if abs(k) == 1]
    margin_ok = bool(neighbours) and rms[0] <= margin * min(neighbours)
    ok = rms[0] <= rms_px and unique and margin_ok
    table = " ".join("k=%+d:%s" % (k, "-" if v is None else "%.2f" % v) for k, v in sorted(rms.items()))
    detail = "RMS(0) %.2f px (limit %.1f), argmin k=%+d, unique %s, RMS(0) <= %.1f x min(RMS(+-1)) %s; %d blobs in " \
             "%d images; %s" % (rms[0], rms_px, best_k, "yes" if unique else "no", margin,
                                 "yes" if margin_ok else "no", result["blobs"], result["used"], table)
    return ok, detail


# ---------------------------------------------------------------------------------------------------------------
# play directory


def header_float(header, key):
    value = header.get(key, "")
    try:
        return float(value)
    except ValueError:
        raise probe_common.InputError("run_header.txt %s=%r is missing or not a number" % (key, value))


def fps_phases(header, requests):
    text = header.get("params.fps_phases", header.get("fps.phases", ""))
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(int(float(token)))
    if not values:
        values = sorted({int(row["fps_setting"]) for row in requests if row.get("fps_setting")})
    return values


def load_markers(play):
    rows = play.csv("e06a_markers", required=("marker_id", "x", "y", "z"))
    moved = set()
    path = play.path("e06a_markers_end.csv")
    if os.path.isfile(path):
        for row in play.csv("e06a_markers_end", required=("marker_id", "moved")):
            if row["moved"] == "1":
                moved.add(int(row["marker_id"]))
    ids, points = [], []
    for row in rows:
        marker_id = int(row["marker_id"])
        if marker_id in moved:
            continue
        ids.append(marker_id)
        points.append((float(row["x"]), float(row["y"]), float(row["z"])))
    return ids, numpy.array(points, dtype=float).reshape(-1, 3), moved


def model_check(play, ids, markers, cam_by_frame, vfov_deg, width, height):
    """Python model vs ProjWorldToScreenNative: list of pixel errors (empty when e06a_proj.csv has no usable row)."""
    path = play.path("e06a_proj.csv")
    if not os.path.isfile(path):
        return []
    index = {marker_id: i for i, marker_id in enumerate(ids)}
    errors = []
    for row in play.csv("e06a_proj", required=("render_frame", "marker_id", "screen_x", "screen_y")):
        frame = int(row["render_frame"])
        marker_id = int(row["marker_id"])
        if frame not in cam_by_frame or marker_id not in index:
            continue
        uv, _ = project(markers[index[marker_id]], cam_by_frame[frame], vfov_deg, width, height)
        if numpy.isfinite(uv[0, 0]):
            errors.append(math.hypot(uv[0, 0] - float(row["screen_x"]), uv[0, 1] - float(row["screen_y"])))
    return errors


def analyse(play_dir, kmax=3, rms_px=2.0, margin=0.5, gate_px=20.0, color="FF00FF", color_tol=60,
            min_motion_px=2.0, min_area=3, fps_tol=0.15):
    """Builds the E-06A verdict of one e06a play directory."""
    fiducial = load_fiducial()
    play = probe_common.Play(play_dir)
    probe = play.header.get("probe", "")
    if probe != PROBE:
        raise probe_common.InputError("%s is a '%s' play, not %s" % (play_dir, probe, PROBE))
    header = play.header
    vfov = header_float(header, "rig.vfov_read")
    try:
        target = parse_color(color)
    except ValueError as exc:
        raise probe_common.InputError(str(exc))

    frames = play.csv("e06a_frames", required=("render_frame", "fps", "fps_setting", "segment", "cam_pos_x"))
    requests = play.csv("e06a_requests", required=("issue_status", "issue_render_frame", "path", "found_suffix",
                                                    "fps_setting", "segment"))
    ids, markers, moved = load_markers(play)
    cam_by_frame = {int(row["render_frame"]): cam_from_row(row) for row in frames}
    files = e05_images.FrameFiles(play.dir)

    verdict = probe_common.Verdict(LABEL, play.dir)
    verdict.info("markers", "%d used, %d excluded as moved" % (len(ids), len(moved)))

    patch = int(float(header.get("fiducial.patch_px", fiducial.PATCH_PX)))
    fid_margin = int(float(header.get("fiducial.margin_px", fiducial.MARGIN_PX)))
    gap = int(float(header.get("fiducial.gap_px", fiducial.GAP_PX)))
    observations = []
    by_segment = []
    size = None
    undecodable = 0
    for row in requests:
        if row["issue_status"] != "issued" or row["segment"] not in USED_SEGMENTS:
            continue
        path, _ = files.resolve(row["path"], row.get("found_suffix", ""))
        if not path:
            continue
        try:
            image = fiducial.load_rgb(path)
        except ValueError:
            undecodable += 1
            continue
        height, width = image.shape[:2]
        size = size or (width, height)
        decoded = fiducial.decode(image, patch=patch, margin=fid_margin, gap=gap)
        if decoded.status != fiducial.STATUS_OK:
            undecodable += 1
            continue
        frame = fiducial.unwrap(decoded.code, int(row["issue_render_frame"]))
        blobs = detect_blobs(image, target, color_tol, band_exclusion(width, header, fiducial), min_area)
        observations.append({"condition": int(row["fps_setting"]), "frame": frame, "blobs": blobs})
        by_segment.append({"condition": "%s/%s" % (row["fps_setting"], row["segment"]), "frame": frame,
                           "blobs": blobs})
    if size is None:
        verdict.check("images", None, "no screenshot of a used segment could be read")
        verdict.check_footer(play)
        return verdict
    width, height = size
    verdict.info("images", "%d decoded, %d undecodable or unreadable, %dx%d px, vfov %.3f deg" % (
        len(observations), undecodable, width, height, vfov))

    model_w = int(float(header.get("fiducial.physical_w", width) or width))
    model_h = int(float(header.get("fiducial.physical_h", height) or height))
    errors = model_check(play, ids, markers, cam_by_frame, vfov, model_w, model_h)
    if errors:
        p95 = probe_common.percentile(errors, 95)
        verdict.check("camera_model", p95 <= MODEL_P95_MAX_PX, "model vs ProjWorldToScreenNative p50 %.3f, p95 %.3f px"
                      " over %d projections (limit p95 %.1f px)" % (probe_common.percentile(errors, 50), p95,
                                                                   len(errors), MODEL_P95_MAX_PX))
    else:
        verdict.info("camera_model", "no usable e06a_proj.csv row")
    if (model_w, model_h) != (width, height):
        verdict.info("size", "screenshots %dx%d differ from the workspace %dx%d" % (width, height, model_w, model_h))

    results = evaluate(observations, cam_by_frame, markers, vfov, width, height, kmax, gate_px, min_motion_px)
    for condition in fps_phases(header, requests):
        fps = [value for row, value in zip(frames, probe_common.floats(frames, "fps"))
               if value is not None and row["fps_setting"] == str(condition) and row["segment"] in USED_SEGMENTS]
        name = "alignment_%dfps" % condition
        if not fps:
            verdict.check(name, None, "no frame at fps_setting %d" % condition)
            continue
        p50 = probe_common.percentile(fps, 50)
        if abs(p50 - condition) > fps_tol * condition:
            verdict.check(name, None, "measured FPS p50 %.1f is not within %.0f%% of the setting %d (MaxFps did not "
                                      "take)" % (p50, 100 * fps_tol, condition))
            continue
        result = results.get(condition)
        if result is None:
            verdict.check(name, None, "no decoded image at fps_setting %d" % condition)
            continue
        ok, detail = judge(result, rms_px, margin)
        verdict.check(name, ok, "FPS p50 %.1f; %s" % (p50, detail))

    for condition, result in sorted(evaluate(by_segment, cam_by_frame, markers, vfov, width, height, kmax, gate_px,
                                             min_motion_px).items()):
        rms = result["rms"]
        verdict.info("segment_%s" % condition, "RMS(-1/0/+1) %s px, %d blobs in %d images" % (
            "/".join("-" if rms.get(k) is None else "%.2f" % rms[k] for k in (-1, 0, 1)), result["blobs"],
            result["used"]))
    verdict.check_footer(play)
    return verdict


def parse_args(argv):
    parser = argparse.ArgumentParser(description="E-06A marker reprojection alignment of one e06a play (r2 3.5).")
    parser.add_argument("play_dir", help="$profile:DC_Probes/e06a/<launch_id>/play_NN as a WSL path")
    parser.add_argument("--kmax", type=int, default=3, help="frame shifts -kmax..kmax (default 3)")
    parser.add_argument("--rms-px", type=float, default=2.0, help="RMS(0) limit in px (default 2.0)")
    parser.add_argument("--margin", type=float, default=0.5, help="RMS(0) <= margin x min(RMS(+-1)) (default 0.5)")
    parser.add_argument("--gate-px", type=float, default=20.0, help="blob-to-projection gate in px (default 20)")
    parser.add_argument("--color", default="FF00FF", help="marker colour RRGGBB (default FF00FF)")
    parser.add_argument("--color-tol", type=int, default=60, help="per-channel colour tolerance (default 60)")
    parser.add_argument("--min-motion-px", type=float, default=2.0,
                        help="median marker motion between F-1 and F+1 needed to count an image (default 2)")
    parser.add_argument("--min-area", type=int, default=3, help="smallest blob in px (default 3)")
    parser.add_argument("--fps-tol", type=float, default=0.15, help="measured FPS p50 tolerance (default 0.15)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return analyse(args.play_dir, args.kmax, args.rms_px, args.margin, args.gate_px, args.color, args.color_tol,
                   args.min_motion_px, args.min_area, args.fps_tol).report()


if __name__ == "__main__":
    sys.exit(probe_common.run_cli(main))
