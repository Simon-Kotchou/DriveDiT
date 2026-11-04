"""Reference of the dc-2.0 collision detector (docs/capture_contract/dc-2.0.md section 9.5.1), stdlib only.

    spec = importlib.util.spec_from_file_location("dc2_crash_dv", "<repo>/data/dc2/crash_dv.py")

The capture core (DD-4-min DC_CrashWatch.c) implements the same rule in Enforce; this module is its executable
specification. The synthetic fixtures compute their collision rows and telemetry.dv_max_mps with it, the tests check
it against the E-01 braking profile with merged steps, and offline tools re-apply a new TBD-10 threshold with it.

The rule. Rows are frame-bound (freeze decision D-A): one callback covers 1 to 3 physics steps of 1/30 s on 1.8, so
every window is world time, never a count of steps or callbacks.
  * each callback stores a sample (t_ms, vx, vz): world_time_ms and the horizontal world-frame velocity;
  * dv_h(t) = |v_h(t) - v_ref|, v_ref = the velocity at t - window_ms, linearly interpolated between the newest
    sample at or before that instant and the sample after it;
  * gap rule: when the two samples around the instant are more than window_ms + tol_ms apart (a frame hitch; a
    callback of 3 merged steps spans 100 ms), the callback is not evaluated and the history restarts at the current
    sample; a history that does not reach back to the instant is not evaluated either; a callback whose world time
    does not advance replaces the newest sample;
  * onset: the first evaluated callback with dv_h >= dv_mps outside the refractory and the suppression; its v_ref is
    kept, dv_mps of the row is the peak of |v_h(t) - v_ref| over peak_ms after the onset.
"""

import math

WINDOW_MS = 100          # session.json events.collision_window_ms (TBD-10 defaults)
WINDOW_TOL_MS = 50       # events.collision_window_tol_ms
DV_MPS = 4.0             # events.collision_dv_mps (PROVISIONAL)
PEAK_MS = 200            # events.collision_peak_ms
REFRACTORY_MS = 1000     # events.collision_refractory_ms
CAPACITY = 32            # samples kept per slot (the Enforce ring); enough for the window at any step >= 5 ms


class Detector:
    """The detector of one vehicle slot. step() returns the callback's dv_h (None when not evaluated)."""

    def __init__(self, window_ms=WINDOW_MS, tol_ms=WINDOW_TOL_MS, dv_mps=DV_MPS, peak_ms=PEAK_MS,
                 refractory_ms=REFRACTORY_MS, capacity=CAPACITY):
        self.window_ms, self.tol_ms, self.dv_mps = window_ms, tol_ms, dv_mps
        self.peak_ms, self.refractory_ms, self.capacity = peak_ms, refractory_ms, capacity
        self.samples = []            # (t_ms, vx, vz), oldest first
        self.suppress_until_ms = -math.inf
        self.refractory_until_ms = -math.inf
        self.pending = None          # the onset whose peak window is still open
        self.onsets = []             # every onset: {step, t_ms, ref, peak, peak_vec}

    def reset(self):
        """A reposition (or a gap): the history is cleared."""
        self.samples = []

    def suppress(self, until_ms):
        self.suppress_until_ms = max(self.suppress_until_ms, until_ms)

    def reference(self, t_ms):
        """(vx, vz) at t_ms - window_ms by interpolation, or None (not evaluated). Applies the gap rule."""
        target = t_ms - self.window_ms
        for i in range(len(self.samples) - 2, -1, -1):
            ta, ax, az = self.samples[i]
            if ta <= target:
                tb, bx, bz = self.samples[i + 1]
                if tb - ta > self.window_ms + self.tol_ms:
                    self.samples = self.samples[-1:]
                    return None
                f = (target - ta) / float(tb - ta)
                return ax + (bx - ax) * f, az + (bz - az) * f
        return None

    def step(self, t_ms, vx, vz, physics_step=None):
        if self.samples and t_ms <= self.samples[-1][0]:
            self.samples[-1] = (self.samples[-1][0], vx, vz)
            return None
        self.samples.append((t_ms, vx, vz))
        if len(self.samples) > self.capacity:
            self.samples.pop(0)
        if self.pending is not None and t_ms <= self.pending["t_ms"] + self.peak_ms:
            rx, rz = self.pending["ref"]
            peak = math.hypot(vx - rx, vz - rz)
            if peak > self.pending["peak"]:
                self.pending["peak"], self.pending["peak_vec"] = peak, (vx - rx, vz - rz)
        elif self.pending is not None:
            self.pending = None
        ref = self.reference(t_ms)
        if ref is None:
            return None
        dv = math.hypot(vx - ref[0], vz - ref[1])
        if dv >= self.dv_mps and self.pending is None and t_ms >= self.suppress_until_ms \
                and t_ms >= self.refractory_until_ms:
            self.pending = {"step": physics_step, "t_ms": t_ms, "ref": ref, "peak": dv,
                            "peak_vec": (vx - ref[0], vz - ref[1])}
            self.onsets.append(self.pending)
            self.refractory_until_ms = t_ms + self.refractory_ms
        return dv


def frame_bound(samples, pattern):
    """Engine-step samples (t_ms, vx, vz, ...) merged into frame-bound callbacks: callback j covers pattern[j %
    len(pattern)] consecutive steps and reports the last one (the state after its steps)."""
    out, i, j = [], 0, 0
    while i < len(samples):
        n = pattern[j % len(pattern)]
        i = min(i + n, len(samples))
        out.append(samples[i - 1])
        j += 1
    return out


def dv_series(samples, **kwargs):
    """[(t_ms, dv or None)] of a detector run over callbacks (t_ms, vx, vz)."""
    det = Detector(**kwargs)
    return [(s[0], det.step(s[0], s[1], s[2])) for s in samples], det
