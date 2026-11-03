"""Fiducial v2 (r2 3.4): the frame code drawn into every capture image, its synthetic encoder and its decoder.

The in-engine side is src/enfusion/datacapture/DC_Fiducial.c (widget carrier) and DC_FiducialShape.c (debug-Shape
carrier, at its own margin: fiducial.shape.* in the run header). All follow the same rule:

  * code = render_frame mod 2^18 (unique for 73 min at 60 FPS; sessions are at most 30 min);
  * 6 patches x 3 bits; patch 0 is the top patch and carries the 3 least significant bits:
    value_i = (code >> (3 * i)) & 7;
  * a patch value is a corner of the RGB cube: bit 0 red, bit 1 green, bit 2 blue, each channel 0 or 255;
  * the left band holds the 6 patches, the right band mirrors them (same values, same order);
  * layout in physical pixels of the W x H image, integer arithmetic with truncating division (Enforce int '/'):
    patch P = 32, margin M = 16, gap G = 8; left x0 = M; right x0 = W - M - P;
    y_i = (H - (6 P + 5 G)) / 2 + i (P + G), i = 0..5 from the top; (x0, y_i) is the patch's top-left pixel.
    For 1408x720: left x0 = 16, right x0 = 1360, y_0 = 244.

Decode rule (r2 3.4, hardened after probe run 1): per patch the centred 8 x 8 block.
  * value: a channel is 1 when its median is >= 255 - channel_margin, 0 when <= channel_margin, else the patch has no
    value;
  * presence gate: a patch is present (drawn) when it has a value, its block is flat (per-channel standard deviation
    <= uniform_std_max) and >= min_corner_fraction of its pixels lie within pixel_tolerance of the value's RGB-cube
    corner on every channel. A drawn patch is an exact corner (run 1: debug shapes within 2 levels), scene texture is
    not: run 1 read dark tree bark in both bands as six black patches and reported code 0 "ok" on every e05 frame
    (t0/run1/FIDUCIAL.md 1.2). With the defaults below no band of the run-1 and run-2 e05 and e06a frames passes;
  * a band decodes ("ok") when all 6 patches are present, is "absent" when none is, else "ambiguous". The image
    status is "ok" when both bands decode and agree, "absent" when both bands are absent (no carrier at these
    origins), else ambiguous_left / ambiguous_right / mirror_mismatch / out_of_bounds;
  * plausibility gate (capture_window, resolve): a decoded code is unwrapped against the request's
    issue_render_frame; the captured frame must lie in [issue, issue + max_offset] and, when the request's
    done_render_frame is known, <= done - 1 (the file of render frame C appears at the EOnFrame of C + 1 at the
    earliest). Outside: status "implausible". The code domain stays v2 (code 0 and equal bands are valid codewords):
    a constant dark band can still decode as code 0 in a dark scene, the window then rejects it.

Image input is an H x W x 3 uint8 RGB numpy array (extra channels are ignored). The module needs only the stdlib
and numpy; cv2 is imported inside load_rgb. It has no import side effects and no relative imports, so callers load
it by file path:

    spec = importlib.util.spec_from_file_location("dc2_fiducial", "<repo>/data/dc2/fiducial.py")
    fiducial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fiducial)
"""

import collections

import numpy

CODE_BITS = 18
CODE_MODULUS = 1 << CODE_BITS
PATCH_COUNT = 6
BITS_PER_PATCH = 3
PATCH_PX = 32
MARGIN_PX = 16
GAP_PX = 8
CENTRE_PX = 8
# Presence gate (module docstring). Run 1 (forest) and run 2 (runway) e05 / e06a frames, ~300 files each, at the
# widget (x 16) and shape (x 64) origins: 0 bands pass; the old rule (margin 80, no gate) passed 616 of 616 run-1
# e05 bands. Real patches: exact corners (debug shapes within 2 levels, run 1).
CHANNEL_MARGIN = 32
PIXEL_TOLERANCE = 32
UNIFORM_STD_MAX = 6.0
MIN_CORNER_FRACTION = 0.9
# Plausibility gate: frames from the request to the captured frame (single flight, run 1: done - 1 = issue + 0 or + 1)
MAX_CAPTURE_OFFSET = 2

STATUS_OK = "ok"
STATUS_ABSENT = "absent"
STATUS_AMBIGUOUS_LEFT = "ambiguous_left"
STATUS_AMBIGUOUS_RIGHT = "ambiguous_right"
STATUS_MIRROR_MISMATCH = "mirror_mismatch"
STATUS_OUT_OF_BOUNDS = "out_of_bounds"
STATUS_IMPLAUSIBLE = "implausible"
STATUSES = (STATUS_OK, STATUS_ABSENT, STATUS_AMBIGUOUS_LEFT, STATUS_AMBIGUOUS_RIGHT, STATUS_MIRROR_MISMATCH,
            STATUS_OUT_OF_BOUNDS, STATUS_IMPLAUSIBLE)

# Band reasons (decode_band)
BAND_OK = "ok"
BAND_ABSENT = "absent"
BAND_AMBIGUOUS = "ambiguous"
BAND_OUT_OF_BOUNDS = "out_of_bounds"

# code: the agreed code or None; left / right: each band's code or None; status: one of STATUSES;
# present_left / present_right: patches that passed the presence gate (0..6); margin: the smallest distance of a
# decoded patch's channel median from its decision threshold (channel_margin - |median - corner|), None unless "ok"
# (dc-2.0 img_resolved.fid_margin)
FiducialResult = collections.namedtuple("FiducialResult", "code left right status present_left present_right margin",
                                        defaults=(0, 0, None))
# One band: code or None, reason (BAND_*), present patches, margin (None unless BAND_OK)
BandResult = collections.namedtuple("BandResult", "code reason present margin")
# Plausibility gate: frame = unwrapped decoded frame (None without a decode), offset = frame - issue, status = the
# decode status or STATUS_IMPLAUSIBLE, window = (lo, hi)
Capture = collections.namedtuple("Capture", "frame offset status window")


def _div_trunc(numerator, denominator):
    """Integer division truncating toward zero, as Enforce int '/' does (Python '//' floors)."""
    quotient = abs(numerator) // denominator
    if numerator < 0:
        return -quotient
    return quotient


def code_for(render_frame):
    """render_frame mod 2^18, always non-negative."""
    return int(render_frame) % CODE_MODULUS


def patch_values(code):
    """The 6 patch values (0..7) of a code, index 0 = top patch."""
    code = int(code)
    if code < 0 or code >= CODE_MODULUS:
        raise ValueError("code %d outside 0..%d" % (code, CODE_MODULUS - 1))
    return [(code >> (BITS_PER_PATCH * i)) & 7 for i in range(PATCH_COUNT)]


def code_from_values(values):
    """Inverse of patch_values."""
    if len(values) != PATCH_COUNT:
        raise ValueError("expected %d patch values, got %d" % (PATCH_COUNT, len(values)))
    code = 0
    for i, value in enumerate(values):
        if value < 0 or value > 7:
            raise ValueError("patch value %r outside 0..7" % (value,))
        code |= int(value) << (BITS_PER_PATCH * i)
    return code


def patch_color(value):
    """(r, g, b) of a patch value: bit 0 red, bit 1 green, bit 2 blue; each channel 0 or 255."""
    value = int(value)
    if value < 0 or value > 7:
        raise ValueError("patch value %d outside 0..7" % value)
    return (255 * (value & 1), 255 * ((value >> 1) & 1), 255 * ((value >> 2) & 1))


def layout(width, height, patch=PATCH_PX, margin=MARGIN_PX, gap=GAP_PX):
    """Top-left pixel of every patch: {"left": [(x0, y0)] * 6, "right": [(x0, y0)] * 6}, index 0 = top."""
    width, height, patch, margin, gap = int(width), int(height), int(patch), int(margin), int(gap)
    y_top = _div_trunc(height - (PATCH_COUNT * patch + (PATCH_COUNT - 1) * gap), 2)
    ys = [y_top + i * (patch + gap) for i in range(PATCH_COUNT)]
    right_x = width - margin - patch
    return {"left": [(margin, y) for y in ys], "right": [(right_x, y) for y in ys]}


def _check_image(image):
    if not isinstance(image, numpy.ndarray) or image.ndim != 3 or image.shape[2] < 3 or image.dtype != numpy.uint8:
        raise ValueError("expected an H x W x 3 uint8 RGB array")


def render(image, code, patch=PATCH_PX, margin=MARGIN_PX, gap=GAP_PX):
    """Synthetic encoder (tests): paints both bands of the code into image, in place."""
    _check_image(image)
    height, width = image.shape[:2]
    geometry = layout(width, height, patch, margin, gap)
    values = patch_values(code)
    for band in ("left", "right"):
        for (x0, y0), value in zip(geometry[band], values):
            if x0 < 0 or y0 < 0 or x0 + patch > width or y0 + patch > height:
                raise ValueError("patch at (%d, %d) does not fit a %dx%d image" % (x0, y0, width, height))
            image[y0:y0 + patch, x0:x0 + patch, 0:3] = patch_color(value)


def _patch(block, channel_margin, pixel_tolerance, uniform_std_max, min_corner_fraction):
    """(value or None, present, margin) of one centre block (N x 3 uint8)."""
    medians = numpy.median(block, axis=0)
    value = 0
    for bit in range(3):
        channel = float(medians[bit])
        if channel >= 255 - channel_margin:
            value |= 1 << bit
        elif channel > channel_margin:
            return None, False, None
    corner = numpy.array(patch_color(value), dtype=numpy.int16)
    pixels = block.astype(numpy.int16)
    near = (numpy.abs(pixels - corner).max(axis=1) <= pixel_tolerance).mean()
    flat = float(pixels.std(axis=0).max()) <= uniform_std_max
    margin = float(channel_margin) - float(numpy.abs(medians - corner).max())
    return value, bool(flat and near >= min_corner_fraction), margin


def band_details(image, origins, patch=PATCH_PX, centre=CENTRE_PX, channel_margin=CHANNEL_MARGIN,
                 pixel_tolerance=PIXEL_TOLERANCE, uniform_std_max=UNIFORM_STD_MAX,
                 min_corner_fraction=MIN_CORNER_FRACTION):
    """BandResult of one band from its 6 patch origins (top-left pixels, index 0 = top)."""
    _check_image(image)
    if len(origins) != PATCH_COUNT:
        raise ValueError("expected %d patch origins, got %d" % (PATCH_COUNT, len(origins)))
    height, width = image.shape[:2]
    offset = (int(patch) - int(centre)) // 2
    code = 0
    present = 0
    margins = []
    for x0, y0 in origins:
        xs = int(x0) + offset
        ys = int(y0) + offset
        if xs < 0 or ys < 0 or xs + centre > width or ys + centre > height:
            return BandResult(None, BAND_OUT_OF_BOUNDS, 0, None)
    for i, (x0, y0) in enumerate(origins):
        xs = int(x0) + offset
        ys = int(y0) + offset
        block = image[ys:ys + centre, xs:xs + centre, 0:3].reshape(-1, 3)
        value, is_present, margin = _patch(block, channel_margin, pixel_tolerance, uniform_std_max,
                                           min_corner_fraction)
        if is_present:
            present += 1
            margins.append(margin)
            code |= value << (BITS_PER_PATCH * i)
    if present == PATCH_COUNT:
        return BandResult(code, BAND_OK, present, min(margins))
    if present == 0:
        return BandResult(None, BAND_ABSENT, 0, None)
    return BandResult(None, BAND_AMBIGUOUS, present, None)


def decode_band(image, origins, patch=PATCH_PX, centre=CENTRE_PX, channel_margin=CHANNEL_MARGIN,
                pixel_tolerance=PIXEL_TOLERANCE, uniform_std_max=UNIFORM_STD_MAX,
                min_corner_fraction=MIN_CORNER_FRACTION):
    """Decodes one band: (code, "ok"), (None, "absent"), (None, "ambiguous") or (None, "out_of_bounds")."""
    band = band_details(image, origins, patch, centre, channel_margin, pixel_tolerance, uniform_std_max,
                        min_corner_fraction)
    return band.code, band.reason


def decode_origins(image, left_origins, right_origins, patch=PATCH_PX, centre=CENTRE_PX,
                   channel_margin=CHANNEL_MARGIN, pixel_tolerance=PIXEL_TOLERANCE, uniform_std_max=UNIFORM_STD_MAX,
                   min_corner_fraction=MIN_CORNER_FRACTION):
    """Decodes both bands at the given origins (for example the fiducial.* geometry of a run header)."""
    gates = (channel_margin, pixel_tolerance, uniform_std_max, min_corner_fraction)
    left = band_details(image, left_origins, patch, centre, *gates)
    right = band_details(image, right_origins, patch, centre, *gates)
    if left.reason == BAND_OUT_OF_BOUNDS or right.reason == BAND_OUT_OF_BOUNDS:
        status = STATUS_OUT_OF_BOUNDS
    elif left.reason == BAND_ABSENT and right.reason == BAND_ABSENT:
        status = STATUS_ABSENT
    elif left.code is None:
        status = STATUS_AMBIGUOUS_LEFT
    elif right.code is None:
        status = STATUS_AMBIGUOUS_RIGHT
    elif left.code != right.code:
        status = STATUS_MIRROR_MISMATCH
    else:
        status = STATUS_OK
    code = None
    margin = None
    if status == STATUS_OK:
        code = left.code
        margin = min(left.margin, right.margin)
    return FiducialResult(code, left.code, right.code, status, left.present, right.present, margin)


def decode(image, patch=PATCH_PX, margin=MARGIN_PX, gap=GAP_PX, centre=CENTRE_PX, channel_margin=CHANNEL_MARGIN,
           pixel_tolerance=PIXEL_TOLERANCE, uniform_std_max=UNIFORM_STD_MAX,
           min_corner_fraction=MIN_CORNER_FRACTION):
    """Decodes an image with the patch origins computed by layout() from the image's own size."""
    _check_image(image)
    height, width = image.shape[:2]
    geometry = layout(width, height, patch, margin, gap)
    return decode_origins(image, geometry["left"], geometry["right"], patch, centre, channel_margin, pixel_tolerance,
                          uniform_std_max, min_corner_fraction)


def unwrap(code, reference_frame):
    """The render_frame congruent to code (mod 2^18) nearest to reference_frame (ties resolve to the earlier one)."""
    half = CODE_MODULUS // 2
    reference_frame = int(reference_frame)
    delta = (int(code) - reference_frame + half) % CODE_MODULUS - half
    return reference_frame + delta


def capture_window(issue_frame, done_frame=None, max_offset=MAX_CAPTURE_OFFSET):
    """(lo, hi): the render frames an image of a request issued in issue_frame can show. lo = issue_frame; hi =
    issue_frame + max_offset, and at most done_frame - 1 when the frame that saw the file is known."""
    lo = int(issue_frame)
    hi = lo + int(max_offset)
    if done_frame is not None:
        hi = min(hi, int(done_frame) - 1)
    return lo, hi


def resolve(result, issue_frame, done_frame=None, max_offset=MAX_CAPTURE_OFFSET):
    """Plausibility gate of one decode: Capture(frame, offset, status, window). A decode outside capture_window gets
    status "implausible" (its frame and offset are kept for the report); any other status passes through."""
    window = capture_window(issue_frame, done_frame, max_offset)
    if result.status != STATUS_OK:
        return Capture(None, None, result.status, window)
    frame = unwrap(result.code, issue_frame)
    status = STATUS_OK
    if frame < window[0] or frame > window[1]:
        status = STATUS_IMPLAUSIBLE
    return Capture(frame, frame - int(issue_frame), status, window)


def load_rgb(path):
    """H x W x 3 uint8 RGB array of an image file (cv2.imread + BGR to RGB); ValueError when unreadable."""
    import cv2  # only here: the rest of the module needs numpy alone

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot read image %s" % path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
