"""Shared pieces of the dc_validate checks: result records, severities, the run context and small statistics.

A check is a function check(ctx) -> result dict:

    {"id": "C06", "severity": "HARD"|"SOFT"|"INFO", "status": "PASS"|"FAIL"|"N/A"|"NOT_EVALUATED",
     "value": ..., "threshold": ..., "detail": "...", "provisional": bool, "not_evaluated": [sub-checks]}

  * severity decides what a FAIL means: HARD fails the session, SOFT warns, INFO never counts (r1 [TL] 5: exit 0
    PASS, 1 FAIL, 2 PASS-with-warnings, 3 validator error);
  * N/A: the check does not apply to this session (for example C17 without mode-E episodes);
  * NOT_EVALUATED: it applies but an input is missing (no engine logs, no route file); a HARD or SOFT check left
    NOT_EVALUATED is a warning;
  * provisional: the check depends on a TBD-by-probe value of contract dc-2.0 (TBD-1 to TBD-9) and the contract is
    not frozen yet, so its threshold or rule may still change at the T0-ENG freeze.

Severities follow the trial gates (severity()): CAPTURE_PROGRAM 4.4 T0-MIN makes C12, C13b and C21 SOFT and T1 makes
C21 HARD; C31 is reported (INFO) before T1. C17 (behaviour.py) and C13b's yaw condition (images.py) add their own
rules. A refusal (Refused: an Everon route outside the sealed path, a lower --trial than the session's) ends the run
with exit 3.
"""

import importlib.util
import math
import os
import re

HARD, SOFT, INFO = "HARD", "SOFT", "INFO"
PASS, FAIL, NA, NOT_EVALUATED = "PASS", "FAIL", "N/A", "NOT_EVALUATED"
ROUTE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")      # drivedit_nav route ids (route_schema.json)

# Severity by trial tier (CAPTURE_PROGRAM 4.4). T0-MIN and S1 (rank < 2; S1's GO is "all T0-MIN HARD checks"):
# HARD C01 C02 C05 C06 C07 C08 C10 C11 C13a C15 C28 C34, SOFT C12 C13b C21. T1 on (rank >= 2): the T0-MIN set plus
# C21, C31 and C35 HARD; C12 and C13b stay SOFT. Every check not named here is HARD at every tier. C20 and C35 are
# HARD from T0 on, stricter than the T0-MIN list by choice: a script error of the capture code or an ego slot that
# changes entity never makes usable data. C31 is reported (INFO) before T1. C17 has its own tier rule (behaviour.py).
SEVERITY_BEFORE_T1 = {"C12": SOFT, "C13b": SOFT, "C21": SOFT, "C31": INFO}
SEVERITY_FROM_T1 = {"C12": SOFT, "C13b": SOFT, "C21": HARD, "C31": HARD}

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
READER_PATH = os.path.join(REPO, "data", "dc2", "reader.py")

# Trial tiers in ladder order (CAPTURE_PROGRAM 4.4). A trial id maps to its tier by prefix: T0b -> T0, T4-K -> T4.
TIER_RANK = {"T0": 0, "S1": 1, "T1": 2, "T2": 3, "T3": 4, "T4": 5, "S2": 6, "T5": 6}
TIER_RE = re.compile(r"^(T0|T1|T2|T3|T4|T5|S1|S2)")

# The TBD-by-probe values each check depends on (dc-2.0.md section 1).
CHECK_TBDS = {
    "C05": (), "C06": ("TBD-1",), "C07": ("TBD-1",), "C08": ("TBD-1", "TBD-2"), "C10": ("TBD-2", "TBD-4"),
    "C11": ("TBD-4",), "C12": (), "C13a": ("TBD-2",), "C13b": (), "C15": ("TBD-5",), "C17": ("TBD-5",),
    "C20": (), "C21": (), "C28": ("TBD-1",), "C31": ("TBD-1", "TBD-2"), "C34": ("TBD-3",), "C35": (),
    "C01": (), "C02": ("TBD-6",),
}

_READER = []


class Refused(Exception):
    """A refusal: dc_validate prints one line and exits 3 (an Everon route or session outside the sealed evaluation
    path, a C: path, a report directory outside /mnt/e, a lower --trial than the session's)."""


def severity(check_id, rank):
    """HARD, SOFT or INFO of a check at a tier rank (SEVERITY_BEFORE_T1 / SEVERITY_FROM_T1)."""
    table = SEVERITY_FROM_T1 if rank >= 2 else SEVERITY_BEFORE_T1
    return table.get(check_id, HARD)


def load_reader():
    """data/dc2/reader.py, loaded by file path (importing the data package would import torch)."""
    if not _READER:
        spec = importlib.util.spec_from_file_location("dc2_reader", READER_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _READER.append(module)
    return _READER[0]


def tier_of(trial_id):
    """(tier, rank) of a trial id; an unknown id is ranked like T5 (strictest) and keeps its own name."""
    m = TIER_RE.match(trial_id or "")
    if not m:
        return (trial_id or "?"), max(TIER_RANK.values())
    return m.group(1), TIER_RANK[m.group(1)]


class Context:
    """What the checks see: the reader Session (or None with load_error), thresholds, tier, options, collected
    image features and the routes."""

    def __init__(self, session, thresholds, trial_id, options, contract_frozen, load_error=None, images=None,
                 routes=None):
        self.session = session
        self.thresholds = thresholds
        self.trial_id = trial_id
        self.tier, self.rank = tier_of(trial_id)
        self.options = options
        self.contract_frozen = contract_frozen
        self.load_error = load_error
        self.images = images
        self.reader = load_reader()
        self._routes = dict(routes or {})

    def severity(self, check_id):
        return severity(check_id, self.rank)

    def thr(self, check, key=None, default=None):
        block = self.thresholds.get(check, {})
        if key is None:
            return block
        return block.get(key, default)

    def route(self, route_id):
        """(Route, path, error) for a route id: the --route files (loaded and checked by dc_validate before any
        session file is read, keyed by their route_id), then <--routes-dir>/<route_id>.json. A route id that is not
        a drivedit_nav id is an error, never a path. An Everon route (map_id everon, an everon_* split, a path under
        heldout/ or SEALED) outside the sealed evaluation path raises Refused: it is never skipped."""
        if route_id in self._routes:
            return self._routes[route_id]
        import drivedit_nav
        routes_dir = getattr(self.options, "routes_dir", None)
        if not (isinstance(route_id, str) and ROUTE_ID_RE.match(route_id)):
            out = (None, None, "route_id %r is not a route id" % (route_id,))
        elif routes_dir and os.path.isfile(os.path.join(routes_dir, route_id + ".json")):
            found = os.path.join(routes_dir, route_id + ".json")
            try:
                out = (drivedit_nav.load_route(found, sealed_eval=bool(getattr(self.options, "sealed_eval", False))),
                       found, None)
            except drivedit_nav.SealedMapError as e:
                raise Refused(str(e))
            except (OSError, ValueError) as e:
                out = (None, found, "route %s: %s" % (found, e))
        else:
            out = (None, None, "no route file for %s (--route / --routes-dir)" % route_id)
        self._routes[route_id] = out
        return out


def result(ctx, check_id, severity, status, value=None, threshold=None, detail="", not_evaluated=None):
    provisional = bool(CHECK_TBDS.get(check_id)) and not ctx.contract_frozen
    out = {"id": check_id, "severity": severity, "status": status, "value": value, "threshold": threshold,
           "detail": detail, "provisional": provisional}
    if not_evaluated:
        out["not_evaluated"] = list(not_evaluated)
    return out


def unreadable(ctx, check_id, severity=None):
    return result(ctx, check_id, severity or ctx.severity(check_id), NOT_EVALUATED,
                  detail="session not readable: %s" % ctx.load_error)


def percentile(values, q):
    """Linear interpolation between closest ranks (numpy's default), q in [0, 100]; None for no values."""
    data = sorted(v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v)))
    if not data:
        return None
    if len(data) == 1:
        return float(data[0])
    pos = (len(data) - 1) * (float(q) / 100.0)
    low = int(pos)
    high = min(low + 1, len(data) - 1)
    return float(data[low]) + (float(data[high]) - float(data[low])) * (pos - low)


def first(items, n=3):
    """At most n items joined for a detail line, with a count of the rest."""
    items = list(items)
    text = "; ".join(str(i) for i in items[:n])
    if len(items) > n:
        text += "; ... (%d more)" % (len(items) - n)
    return text


def wrap_angle(a):
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi
