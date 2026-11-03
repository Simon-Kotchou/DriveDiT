#!/usr/bin/env python3
"""Fiducial v2 carriers of one probe play: which carriers it drew, where, and how each one decodes (DD-4-probe).

Not a script itself. Shared by e05_images.py, e06_fiducial.py and tools/validator/e06a_reproject.py, which put this
folder on sys.path (see probe_common.py). data/dc2/fiducial.py is loaded by file path (load_fiducial).

Carriers (run header, appended by DC_FiducialCarriers in src/enfusion/datacapture/DC_ProbeFiducial.c):
  * rig.fiducial.carrier = widget | shape | both | none. A play written before the shape carrier has no such key: it
    drew the widget carrier unless fiducial.created=0;
  * widget: DC_Fiducial ImageWidgets, geometry fiducial.patch_px / margin_px / gap_px, fiducial.left_x / right_x /
    y0..y5;
  * shape: DC_FiducialShape debug Shapes, geometry fiducial.shape.* (same keys), margin 64 by default;
  * widget diagnostics: fiducial.control.* (three control widgets, colour 255,128,0, rectangles "x y w h" in
    physical px) and the stop totals fiducial.diag.*; shape placement totals fiducial.shape.place_* / last_* /
    residual_*; the runner's footer key clock.postframe_calls.

Per carrier and file: data/dc2/fiducial.decode_origins (presence gate) then fiducial.resolve (plausibility gate
against the request's issue_render_frame and done_render_frame). CarrierStats keeps the per-status counts, the
offsets of plausible decodes, the done - 1 agreement of single-flight requests and the smallest margin.
"""

import importlib.util
import os

import probe_common

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FIDUCIAL_PATH = os.path.join(REPO, "data", "dc2", "fiducial.py")

WIDGET = "widget"
SHAPE = "shape"
CARRIERS = (WIDGET, SHAPE)
PREFIX = {WIDGET: "fiducial.", SHAPE: "fiducial.shape."}
DEFAULT_MARGIN_PX = {WIDGET: 16, SHAPE: 64}
UNREADABLE = "unreadable"

# Control widgets in a screenshot (fiducial.control.*): the flat block is an exact colour, the text is partly
# covered by glyphs, the textured block is a texture tinted (255, 128, 0): its blue channel is 0 everywhere (dark
# texels included), and it is not black as a whole
CONTROL_COLOR = (255, 128, 0)
CONTROL_TOL = 16
CONTROL_FLAT_MIN = 0.9
CONTROL_TEXT_MIN = 0.02
CONTROL_TEXTURED_BLUE_MAX = 4
CONTROL_TEXTURED_MIN = 0.9
CONTROL_TEXTURED_RED_MEAN_MIN = 8
CONTROLS = ("flat", "textured", "text")


def load_fiducial():
    """data/dc2/fiducial.py by file path (importing the data package would load torch)."""
    spec = importlib.util.spec_from_file_location("dc2_fiducial", FIDUCIAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def header_int(header, key, default=None):
    value = header.get(key, "")
    if value == "":
        if default is None:
            raise probe_common.InputError("run_header.txt has no %s" % key)
        return default
    try:
        return int(float(value))
    except ValueError:
        raise probe_common.InputError("run_header.txt %s=%r is not a number" % (key, value))


def carriers(header):
    """The carriers a play drew, in CARRIERS order (rig.fiducial.carrier; module docstring)."""
    value = header.get("rig.fiducial.carrier", "").strip()
    if value == "":
        if header.get("fiducial.created", "1").strip() == "0":
            return []
        return [WIDGET]
    if value == "both":
        return [WIDGET, SHAPE]
    if value in CARRIERS:
        return [value]
    if value == "none":
        return []
    raise probe_common.InputError("run_header.txt rig.fiducial.carrier=%r is not widget, shape, both or none" % value)


def carrier_params(header, fiducial, carrier):
    """(patch, margin, gap) of a carrier from its header keys (defaults of DC_ProbeImages / DC_ProbeMarkers)."""
    prefix = PREFIX[carrier]
    patch = header_int(header, prefix + "patch_px", fiducial.PATCH_PX)
    margin = header_int(header, prefix + "margin_px", DEFAULT_MARGIN_PX[carrier])
    gap = header_int(header, prefix + "gap_px", fiducial.GAP_PX)
    return patch, margin, gap


def geometry_from_header(header, fiducial, carrier=WIDGET):
    """(left origins, right origins, patch px) from a carrier's left_x, right_x and y0..y5 keys."""
    prefix = PREFIX[carrier]
    patch = header_int(header, prefix + "patch_px")
    left_x = header_int(header, prefix + "left_x")
    right_x = header_int(header, prefix + "right_x")
    ys = [header_int(header, prefix + "y%d" % i) for i in range(fiducial.PATCH_COUNT)]
    return [(left_x, y) for y in ys], [(right_x, y) for y in ys], patch


def origins(header, fiducial, carrier, width, height, header_geometry=False):
    """(left origins, right origins, patch px) of a carrier on a width x height image: the layout of the image's own
    size with the carrier's patch, margin and gap, or with header_geometry its header keys."""
    if header_geometry:
        return geometry_from_header(header, fiducial, carrier)
    patch, margin, gap = carrier_params(header, fiducial, carrier)
    geometry = fiducial.layout(width, height, patch, margin, gap)
    return geometry["left"], geometry["right"], patch


def band_exclusion(width, header, fiducial):
    """x ranges [x0, x1) covering the bands of every carrier the play drew, plus one widget margin of clearance on
    the inner side (the E-06A blob detector skips them: value 5 is the marker magenta)."""
    inner_left = 0
    inner_right = width
    clearance = carrier_params(header, fiducial, WIDGET)[1]
    for carrier in carriers(header) or [WIDGET]:
        patch, margin, _ = carrier_params(header, fiducial, carrier)
        prefix = PREFIX[carrier]
        left_x = header.get(prefix + "left_x", "")
        right_x = header.get(prefix + "right_x", "")
        if left_x != "" and right_x != "":
            left_x, right_x = int(float(left_x)), int(float(right_x))
        else:
            left_x, right_x = margin, width - margin - patch
        inner_left = max(inner_left, left_x + patch + clearance)
        inner_right = min(inner_right, right_x - clearance)
    return [(0, inner_left), (max(0, inner_right), width)]


def decode_all(fiducial, image, header, carrier_list, issue_frame, done_frame=None,
               max_offset=None, header_geometry=False):
    """{carrier: (FiducialResult, Capture)} of one image."""
    if max_offset is None:
        max_offset = fiducial.MAX_CAPTURE_OFFSET
    height, width = image.shape[:2]
    out = {}
    for carrier in carrier_list:
        left, right, patch = origins(header, fiducial, carrier, width, height, header_geometry)
        result = fiducial.decode_origins(image, left, right, patch=patch)
        out[carrier] = (result, fiducial.resolve(result, issue_frame, done_frame, max_offset))
    return out


def agreed_frame(fiducial, decodes):
    """(frame, reason): the captured frame every plausible carrier agrees on; reason "none" when no carrier decoded
    plausibly, "disagree" when two plausible carriers name different frames."""
    frames = {capture.frame for _, capture in decodes.values() if capture.status == fiducial.STATUS_OK}
    if not frames:
        return None, "none"
    if len(frames) > 1:
        return None, "disagree"
    return frames.pop(), "ok"


class CarrierStats:
    """Decode results of one carrier over the files of a play."""

    def __init__(self, carrier):
        self.carrier = carrier
        self.files = 0
        self.counts = {}
        self.offsets = []
        self.implausible = []
        self.single = 0
        self.single_done_minus_1 = 0
        self.margins = []
        self.any_present = 0

    def add(self, result, capture, done_frame=None, single_flight=False):
        self.files += 1
        self.counts[capture.status] = self.counts.get(capture.status, 0) + 1
        if result.present_left or result.present_right:
            self.any_present += 1
        if capture.status == "ok":
            self.offsets.append(capture.offset)
            if result.margin is not None:
                self.margins.append(result.margin)
            if single_flight and done_frame is not None:
                self.single += 1
                if capture.frame == done_frame - 1:
                    self.single_done_minus_1 += 1
        elif capture.status == "implausible":
            self.implausible.append(capture.offset)

    def add_unreadable(self):
        self.files += 1
        self.counts[UNREADABLE] = self.counts.get(UNREADABLE, 0) + 1

    def decoded(self):
        return len(self.offsets)

    def rate(self):
        if self.files == 0:
            return None
        return 100.0 * self.decoded() / self.files

    def status_text(self):
        return ", ".join("%s %d" % item for item in sorted(self.counts.items())) or "none"

    def offset_text(self):
        if not self.offsets:
            return "no plausible decode"
        return "offset p50 %.1f, p99 %.1f, min %d, max %d frames" % (
            probe_common.percentile(self.offsets, 50), probe_common.percentile(self.offsets, 99),
            min(self.offsets), max(self.offsets))

    def summary(self, limit_pct=None):
        """One text line: decoded n of m (pct), the statuses, the offsets, done - 1 agreement, margin, presence."""
        rate = self.rate()
        pct = "n/a" if rate is None else "%.2f%%" % rate
        text = "decoded %d of %d files (%s" % (self.decoded(), self.files, pct)
        if limit_pct is not None:
            text += ", limit %.1f%%" % limit_pct
        text += "); %s; %s" % (self.status_text(), self.offset_text())
        if self.implausible:
            text += "; implausible offsets %d..%d" % (min(self.implausible), max(self.implausible))
        if self.single:
            text += "; done - 1 = captured on %d of %d single-flight decodes" % (self.single_done_minus_1, self.single)
        if self.margins:
            text += "; margin min %.1f" % min(self.margins)
        text += "; files with >= 1 present patch %d" % self.any_present
        return text


def parse_rect(text):
    """(x, y, w, h) of a fiducial.control.* value "x y w h", None when unusable."""
    parts = (text or "").split()
    if len(parts) != 4:
        return None
    try:
        return tuple(int(float(part)) for part in parts)
    except ValueError:
        return None


class ControlStats:
    """Presence of the widget-diagnostics control widgets in the screenshots (fiducial.control.* rectangles)."""

    def __init__(self, header):
        self.rects = {}
        for name in CONTROLS:
            rect = parse_rect(header.get("fiducial.control." + name, ""))
            if rect:
                self.rects[name] = rect
        self.color = CONTROL_COLOR
        color = header.get("fiducial.control.color", "")
        if color:
            try:
                self.color = tuple(int(part) for part in color.split(","))
            except ValueError:
                pass
        self.files = 0
        self.present = {name: 0 for name in self.rects}

    def active(self):
        return bool(self.rects)

    def fractions(self, image):
        """{control: fraction of its rectangle's pixels that match it} for one image."""
        import numpy  # only here: the carrier list and the texts need the stdlib alone

        height, width = image.shape[:2]
        out = {}
        target = numpy.array(self.color, dtype=numpy.int16)
        for name, (x, y, w, h) in self.rects.items():
            x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
            if x1 <= x0 or y1 <= y0:
                out[name] = 0.0
                continue
            block = image[y0:y1, x0:x1, 0:3].astype(numpy.int16).reshape(-1, 3)
            if name == "textured":
                if float(block[:, 0].mean()) < CONTROL_TEXTURED_RED_MEAN_MIN:
                    out[name] = 0.0
                    continue
                match = block[:, 2] <= CONTROL_TEXTURED_BLUE_MAX
            else:
                match = numpy.abs(block - target).max(axis=1) <= CONTROL_TOL
            out[name] = float(match.mean())
        return out

    def add(self, image):
        self.files += 1
        for name, fraction in self.fractions(image).items():
            limit = {"flat": CONTROL_FLAT_MIN, "text": CONTROL_TEXT_MIN, "textured": CONTROL_TEXTURED_MIN}[name]
            if fraction >= limit:
                self.present[name] += 1

    def summary(self):
        parts = ["%s %d" % (name, self.present[name]) for name in CONTROLS if name in self.present]
        return "in %s of %d files (colour %s; flat >= %.0f%% within %d, text >= %.0f%% within %d, textured blue <= "\
               "%d on >= %.0f%% and red mean >= %d)" % (
                   ", ".join(parts), self.files, ",".join(str(c) for c in self.color), 100 * CONTROL_FLAT_MIN,
                   CONTROL_TOL, 100 * CONTROL_TEXT_MIN, CONTROL_TOL, CONTROL_TEXTURED_BLUE_MAX,
                   100 * CONTROL_TEXTURED_MIN, CONTROL_TEXTURED_RED_MEAN_MIN)


def readback_text(header):
    """The widget readback totals (fiducial.diag.*), None when the play has none."""
    if "fiducial.diag.readbacks" not in header:
        return None
    get = header.get
    return "%s readbacks: patches visible_in_hierarchy %s..%s of 12, at layout %s..%s, max offset %s px; controls " \
           "visible text %s, textured %s, flat %s (replaced %s); last: %s" % (
               get("fiducial.diag.readbacks"), get("fiducial.diag.patches_visible_min", "n/a"),
               get("fiducial.diag.patches_visible_max", "n/a"), get("fiducial.diag.patches_at_layout_min", "n/a"),
               get("fiducial.diag.patches_at_layout_max", "n/a"), get("fiducial.diag.patch_offset_max_px", "n/a"),
               get("fiducial.diag.control_text_visible", "n/a"), get("fiducial.diag.control_textured_visible", "n/a"),
               get("fiducial.diag.control_flat_visible", "n/a"), get("fiducial.diag.control_replacements", "n/a"),
               get("fiducial.diag.last", "n/a"))


def placement_text(header, footer=None):
    """The shape placement totals (fiducial.shape.place_* / last_* / residual_*), None when the play has none."""
    if "fiducial.shape.residual_frames" not in header:
        return None
    get = header.get
    postframe_calls = "n/a"
    if footer:
        postframe_calls = footer.get("clock.postframe_calls", "n/a")
    return "%s frames, last placed by postframe %s, poststep %s, frame %s; residual vs the world camera max %s px " \
           "(postframe %s, poststep %s, frame %s), mean %s px, over 1 px %s; vs the rig camera max %s px; placements " \
           "frame %s, poststep %s, postframe %s; depth %s m; runner clock.postframe_calls %s" % (
               get("fiducial.shape.residual_frames"), get("fiducial.shape.last_postframe", "n/a"),
               get("fiducial.shape.last_poststep", "n/a"), get("fiducial.shape.last_frame", "n/a"),
               get("fiducial.shape.residual_max_px", "n/a"), get("fiducial.shape.residual_max_px_postframe", "n/a"),
               get("fiducial.shape.residual_max_px_poststep", "n/a"),
               get("fiducial.shape.residual_max_px_frame", "n/a"),
               get("fiducial.shape.residual_mean_px", "n/a"), get("fiducial.shape.residual_over_1px", "n/a"),
               get("fiducial.shape.residual_rig_max_px", "n/a"), get("fiducial.shape.place_frame", "n/a"),
               get("fiducial.shape.place_poststep", "n/a"), get("fiducial.shape.place_postframe", "n/a"),
               get("rig.fiducial.depth_m", "n/a"), postframe_calls)


def is_single_flight(row):
    """A request issued with nothing else in flight and outside a backlog phase (inflight_at_issue 0, backlog 0)."""
    return row.get("inflight_at_issue", "0") in ("0", "") and row.get("backlog", "0") in ("0", "")
