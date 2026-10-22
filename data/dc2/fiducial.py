"""Fiducial v2 (r2 3.4): the frame code drawn into every capture image, its synthetic encoder and its decoder.

The in-engine side is src/enfusion/datacapture/DC_Fiducial.c. Both follow the same rule:

  * code = render_frame mod 2^18 (unique for 73 min at 60 FPS; sessions are at most 30 min);
  * 6 patches x 3 bits; patch 0 is the top patch and carries the 3 least significant bits:
    value_i = (code >> (3 * i)) & 7;
  * a patch value is a corner of the RGB cube: bit 0 red, bit 1 green, bit 2 blue, each channel 0 or 255;
  * the left band holds the 6 patches, the right band mirrors them (same values, same order);
  * layout in physical pixels of the W x H image, integer arithmetic with truncating division (Enforce int '/'):
    patch P = 32, margin M = 16, gap G = 8; left x0 = M; right x0 = W - M - P;
    y_i = (H - (6 P + 5 G)) / 2 + i (P + G), i = 0..5 from the top; (x0, y_i) is the patch's top-left pixel.
    For 1408x720: left x0 = 16, right x0 = 1360, y_0 = 244.

Decode rule (r2 3.4): per patch the per-channel median of the centred 8 x 8 block; a channel is 1 when its median is
>= 255 - channel_margin, 0 when <= channel_margin, otherwise the patch is ambiguous. A band decodes when all 6
patches decode; the image decodes (status "ok") when both bands decode and agree.

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
CHANNEL_MARGIN = 80

STATUS_OK = "ok"
STATUS_AMBIGUOUS_LEFT = "ambiguous_left"
STATUS_AMBIGUOUS_RIGHT = "ambiguous_right"
STATUS_MIRROR_MISMATCH = "mirror_mismatch"
STATUS_OUT_OF_BOUNDS = "out_of_bounds"
STATUSES = (STATUS_OK, STATUS_AMBIGUOUS_LEFT, STATUS_AMBIGUOUS_RIGHT, STATUS_MIRROR_MISMATCH, STATUS_OUT_OF_BOUNDS)

# code: the agreed code or None; left / right: each band's code or None; status: one of STATUSES
FiducialResult = collections.namedtuple("FiducialResult", "code left right status")


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


def decode_band(image, origins, patch=PATCH_PX, centre=CENTRE_PX, channel_margin=CHANNEL_MARGIN):
    """Decodes one band from its 6 patch origins (top-left pixels, index 0 = top).

    Returns (code, "ok"), (None, "ambiguous") or (None, "out_of_bounds").
    """
    _check_image(image)
    if len(origins) != PATCH_COUNT:
        raise ValueError("expected %d patch origins, got %d" % (PATCH_COUNT, len(origins)))
    height, width = image.shape[:2]
    offset = (int(patch) - int(centre)) // 2
    code = 0
    for i, (x0, y0) in enumerate(origins):
        xs = int(x0) + offset
        ys = int(y0) + offset
        if xs < 0 or ys < 0 or xs + centre > width or ys + centre > height:
            return None, "out_of_bounds"
        block = image[ys:ys + centre, xs:xs + centre, 0:3].reshape(-1, 3)
        medians = numpy.median(block, axis=0)
        value = 0
        for bit in range(3):
            channel = float(medians[bit])
            if channel >= 255 - channel_margin:
                value |= 1 << bit
            elif channel > channel_margin:
                return None, "ambiguous"
        code |= value << (BITS_PER_PATCH * i)
    return code, "ok"


def decode_origins(image, left_origins, right_origins, patch=PATCH_PX, centre=CENTRE_PX,
                   channel_margin=CHANNEL_MARGIN):
    """Decodes both bands at the given origins (for example the fiducial.* geometry of a run header)."""
    left, left_reason = decode_band(image, left_origins, patch, centre, channel_margin)
    right, right_reason = decode_band(image, right_origins, patch, centre, channel_margin)
    if left_reason == "out_of_bounds" or right_reason == "out_of_bounds":
        status = STATUS_OUT_OF_BOUNDS
    elif left is None:
        status = STATUS_AMBIGUOUS_LEFT
    elif right is None:
        status = STATUS_AMBIGUOUS_RIGHT
    elif left != right:
        status = STATUS_MIRROR_MISMATCH
    else:
        status = STATUS_OK
    code = None
    if status == STATUS_OK:
        code = left
    return FiducialResult(code, left, right, status)


def decode(image, patch=PATCH_PX, margin=MARGIN_PX, gap=GAP_PX, centre=CENTRE_PX, channel_margin=CHANNEL_MARGIN):
    """Decodes an image with the patch origins computed by layout() from the image's own size."""
    _check_image(image)
    height, width = image.shape[:2]
    geometry = layout(width, height, patch, margin, gap)
    return decode_origins(image, geometry["left"], geometry["right"], patch, centre, channel_margin)


def unwrap(code, reference_frame):
    """The render_frame congruent to code (mod 2^18) nearest to reference_frame (ties resolve to the earlier one)."""
    half = CODE_MODULUS // 2
    reference_frame = int(reference_frame)
    delta = (int(code) - reference_frame + half) % CODE_MODULUS - half
    return reference_frame + delta


def load_rgb(path):
    """H x W x 3 uint8 RGB array of an image file (cv2.imread + BGR to RGB); ValueError when unreadable."""
    import cv2  # only here: the rest of the module needs numpy alone

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot read image %s" % path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
