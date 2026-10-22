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
"""

import importlib.util
import json
import math
import os
import re

HARD, SOFT, INFO = "HARD", "SOFT", "INFO"
PASS, FAIL, NA, NOT_EVALUATED = "PASS", "FAIL", "N/A", "NOT_EVALUATED"

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

    def __init__(self, session, thresholds, trial_id, options, contract_frozen, load_error=None, images=None):
        self.session = session
        self.thresholds = thresholds
        self.trial_id = trial_id
        self.tier, self.rank = tier_of(trial_id)
        self.options = options
        self.contract_frozen = contract_frozen
        self.load_error = load_error
        self.images = images
        self.reader = load_reader()
        self._routes = {}

    def thr(self, check, key=None, default=None):
        block = self.thresholds.get(check, {})
        if key is None:
            return block
        return block.get(key, default)

    def route(self, route_id):
        """(Route, path, error) for a route id: --route files (by their route_id) first, then --routes-dir."""
        if route_id in self._routes:
            return self._routes[route_id]
        import drivedit_nav
        found = None
        for path in getattr(self.options, "route", None) or []:
            try:
                with open(path, "rb") as f:
                    if json.loads(f.read().decode("utf-8")).get("route_id") == route_id:
                        found = path
                        break
            except (OSError, ValueError, AttributeError):
                continue
        routes_dir = getattr(self.options, "routes_dir", None)
        if found is None and routes_dir:
            candidate = os.path.join(routes_dir, route_id + ".json")
            if os.path.isfile(candidate):
                found = candidate
        if found is None:
            out = (None, None, "no route file for %s (--route / --routes-dir)" % route_id)
        else:
            try:
                out = (drivedit_nav.load_route(found, sealed_eval=bool(getattr(self.options, "sealed_eval", False))),
                       found, None)
            except (ValueError, PermissionError) as e:
                out = (None, found, "route %s: %s" % (found, e))
        self._routes[route_id] = out
        return out


def result(ctx, check_id, severity, status, value=None, threshold=None, detail="", not_evaluated=None):
    provisional = bool(CHECK_TBDS.get(check_id)) and not ctx.contract_frozen
    out = {"id": check_id, "severity": severity, "status": status, "value": value, "threshold": threshold,
           "detail": detail, "provisional": provisional}
    if not_evaluated:
        out["not_evaluated"] = list(not_evaluated)
    return out


def unreadable(ctx, check_id, severity=HARD):
    return result(ctx, check_id, severity, NOT_EVALUATED, detail="session not readable: %s" % ctx.load_error)


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
