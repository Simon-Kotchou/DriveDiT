"""dc_validate checks (DD-5-min): the T0-MIN and T1 set of CAPTURE_PROGRAM r2 5.1, in report order.

    C01 C02 (files)  C05 C06 C07 C34 C35 (streams)  C08 C28 (timing)  C10 C11 C12 C13a C13b (images)
    C15 (dynamics)  C17 (behaviour, SOFT at T1)  C20 C21 (health)  C31 (usable yield)

Each module documents its checks; base.py holds the result record, the severities and the run context.
"""

from . import behaviour, dynamics, files, health, images, streams, timing, usable
from .base import FAIL, HARD, NOT_EVALUATED, SOFT

REGISTRY = (
    ("C01", files.check_c01),
    ("C02", files.check_c02),
    ("C05", streams.check_c05),
    ("C06", streams.check_c06),
    ("C07", streams.check_c07),
    ("C08", timing.check_c08),
    ("C10", images.check_c10),
    ("C11", images.check_c11),
    ("C12", images.check_c12),
    ("C13a", images.check_c13a),
    ("C13b", images.check_c13b),
    ("C15", dynamics.check_c15),
    ("C17", behaviour.check_c17),
    ("C20", health.check_c20),
    ("C21", health.check_c21),
    ("C28", timing.check_c28),
    ("C31", usable.check_c31),
    ("C34", streams.check_c34),
    ("C35", streams.check_c35),
)
CHECK_IDS = tuple(cid for cid, _ in REGISTRY)

PASS_VERDICT, FAIL_VERDICT, WARN_VERDICT = "PASS", "FAIL", "PASS_WITH_WARNINGS"
EXIT_CODES = {PASS_VERDICT: 0, FAIL_VERDICT: 1, WARN_VERDICT: 2}
EXIT_VALIDATOR_ERROR = 3


def run_all(ctx, only=None):
    """Results of every registered check (or of the ids in only), in registry order."""
    return [fn(ctx) for cid, fn in REGISTRY if only is None or cid in only]


def verdict(results):
    """FAIL when a HARD check failed; PASS_WITH_WARNINGS when a SOFT check failed or a HARD or SOFT check could not
    be evaluated; else PASS. Returns (verdict, exit code)."""
    if any(r["severity"] == HARD and r["status"] == FAIL for r in results):
        v = FAIL_VERDICT
    elif any((r["severity"] == SOFT and r["status"] == FAIL)
             or (r["severity"] in (HARD, SOFT) and r["status"] == NOT_EVALUATED) for r in results):
        v = WARN_VERDICT
    else:
        v = PASS_VERDICT
    return v, EXIT_CODES[v]
