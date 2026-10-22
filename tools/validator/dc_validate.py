#!/usr/bin/env python3
"""dc_validate: validator of capture contract dc-2.0 sessions (DD-5-min; CAPTURE_PROGRAM r2 section 5).

    python3 -I -B tools/validator/dc_validate.py --session DIR [--profile capture] [--trial ID]
        [--report-dir DIR | --no-report] [--engine-logs DIR] [--route FILE ...] [--routes-dir DIR]
        [--launch-record FILE] [--allow-list FILE] [--vehicle-card FILE] [--thresholds FILE] [--sealed-eval]
        [--checks C01,C06,...] [--json] [--legacy]

What it checks: C01 C02 C05 C06 C07 C08 C10 C11 C12 C13a C13b C15 C17 C20 C21 C28 C31 C34 C35 (the T0-MIN and T1
set; tools/validator/checks/ documents each). The session is read by data/dc2/reader.py (strict=False, so every
problem is reported, not raised). Thresholds are CAPTURE_PROGRAM r2 5.2 verbatim (thresholds_dc2.json). Trial
rules (C02 dirty flag from T1, C10 T0 limit, C17 SOFT at T1 / HARD at T2-T3 / INFO elsewhere, C21 memory and C31
from T1) follow the session's trial.trial_id, or --trial. A check that depends on a TBD-by-probe value is marked
"provisional" while dc-2.0 is a DRAFT. --profile capture is the only profile so far (eval_lockstep and eval_async
come with the bridge, DD-8).

--legacy runs the dc-1.0 self-test instead (checks/legacy.py): a dc-1.0 session must FAIL for its known reasons.

Exit codes (r1 [TL] 5): 0 PASS, 1 FAIL (a HARD check failed), 2 PASS with warnings (a SOFT check failed, or a HARD
or SOFT check could not be evaluated, e.g. no engine logs), 3 validator error or refusal (bad arguments, an Everon
session without --sealed-eval, a path on C:, a report directory outside /mnt/e).

Report: <report-dir>/<session_id>/dc_validate.json = {report_schema, session_id, session_dir, trial, tier,
profile, validator_version, contract_version, contract_status, thresholds_sha256, verdict, exit_code, checks:
[{id, severity, status, value, threshold, detail, provisional, not_evaluated?}], stats, created_utc}. Reports are
written only under /mnt/e/ (default /mnt/e/DriveDiT/reports); an Everon report only under a SEALED directory
(L12). The validator never writes anywhere else, and reads no C: path.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for path in (HERE, REPO):
    if path not in sys.path:
        sys.path.insert(0, path)
import checks  # noqa: E402
from checks import base  # noqa: E402
from checks.images import ImageCollector  # noqa: E402
from checks.legacy import run_legacy  # noqa: E402

VALIDATOR_VERSION = "dc_validate/0.1 (DD-5-min)"
REPORT_SCHEMA = "drivedit.dc-validate-report/1"
DEFAULT_THRESHOLDS = os.path.join(HERE, "thresholds_dc2.json")
DEFAULT_REPORT_DIR = "/mnt/e/DriveDiT/reports"
REPORT_ROOT = "/mnt/e/"
PROFILES = ("capture", "eval_lockstep", "eval_async")


class Refused(Exception):
    """A refusal: exit 3 with one line."""


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print("usage error: %s" % message, file=sys.stderr)
        sys.exit(checks.EXIT_VALIDATOR_ERROR)


def _under(path, root):
    real = os.path.realpath(path)
    probe = real if real.endswith("/") else real + "/"
    return probe.startswith(root)


def check_read_path(path, what):
    if _under(path, "/mnt/c/"):
        raise Refused("refusing %s %s: no repo tool receives a C: path (use the E: copy)" % (what, path))
    return os.path.realpath(path)


def check_report_dir(path, sealed=False):
    """Reports only under /mnt/e/ (never /mnt/c or anything else); an Everon report only under a SEALED dir."""
    if _under(path, "/mnt/c/") or not _under(path, REPORT_ROOT):
        raise Refused("refusing report directory %s: reports are written only under %s" % (path, REPORT_ROOT))
    real = os.path.realpath(path)
    if sealed and "sealed" not in real.lower().split("/"):
        raise Refused("refusing report directory %s for an Everon session: sealed reports go under a SEALED "
                      "directory (L12)" % path)
    return real


def load_thresholds(path):
    with open(path, "rb") as f:
        raw = f.read()
    data = json.loads(raw.decode("ascii"))
    return {k: v for k, v in data.items() if not k.startswith("_")}, hashlib.sha256(raw).hexdigest()


def parse_args(argv):
    ap = _Parser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", required=True, help="dc-2.0 session directory (or a dc-1.0 one with --legacy)")
    ap.add_argument("--profile", choices=PROFILES, default="capture")
    ap.add_argument("--trial", default=None, help="trial id for the trial rules (default: session.json trial_id)")
    ap.add_argument("--report-dir", default=DEFAULT_REPORT_DIR, help="report root under /mnt/e (default %(default)s)")
    ap.add_argument("--no-report", action="store_true", help="print only, write no report")
    ap.add_argument("--engine-logs", default=None, help="sliced engine logs (default <session>/_engine_logs)")
    ap.add_argument("--route", action="append", default=[], help="route file (drivedit_nav schema); repeatable")
    ap.add_argument("--routes-dir", default=None, help="directory of <route_id>.json route files")
    ap.add_argument("--launch-record", default=None, help="the session's launch_<id>.json (DD-0 preflight)")
    ap.add_argument("--allow-list", default=None, help='JSON {"expert_sha": [...], "llc_sha": [...]}')
    ap.add_argument("--vehicle-card", default=None, help="vehicle card JSON (gear indices, TBD-7)")
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--sealed-eval", action="store_true", help="the sealed evaluation path (Everon)")
    ap.add_argument("--checks", default=None, help="comma-separated subset of check ids")
    ap.add_argument("--json", action="store_true", help="print the report JSON instead of the summary")
    ap.add_argument("--legacy", action="store_true", help="dc-1.0 self-test: the session must FAIL")
    return ap.parse_args(argv)


def validate(args):
    """(report dict, exit code) for parsed arguments; raises Refused."""
    if args.profile != "capture":
        raise Refused("profile %s is not implemented yet (DD-5-min ships the capture profile)" % args.profile)
    session_dir = check_read_path(args.session, "session")
    for name in ("engine_logs", "routes_dir", "launch_record", "allow_list", "vehicle_card"):
        value = getattr(args, name)
        if value:
            setattr(args, name, check_read_path(value, name.replace("_", "-")))
    args.route = [check_read_path(r, "route") for r in args.route]
    if not args.no_report:
        check_report_dir(args.report_dir, sealed=args.sealed_eval)
    thresholds, thr_sha = load_thresholds(args.thresholds)
    reader = base.load_reader()
    schema = reader.load_schema()
    only = set(c.strip() for c in args.checks.split(",")) if args.checks else None
    if only and only - set(checks.CHECK_IDS):
        raise Refused("unknown check id(s) %s" % sorted(only - set(checks.CHECK_IDS)))
    session = None
    collector = ImageCollector()
    load_error = None
    trial_id = args.trial
    if args.legacy:
        results = run_legacy(session_dir, schema, thresholds)
        session_id = os.path.basename(os.path.normpath(session_dir))
        trial_id = trial_id or "legacy"
    else:
        try:
            session = reader.open_session(session_dir, sealed_eval=args.sealed_eval, strict=False,
                                          vehicle_card=args.vehicle_card, image_visitor=collector.visit)
        except reader.SealedMapError as e:
            raise Refused(str(e))
        except reader.LegacySessionError as e:
            load_error = "%s (run --legacy for the dc-1.0 self-test)" % e
        except reader.ReaderError as e:
            load_error = str(e)
        session_id = session.session_id if session is not None else os.path.basename(os.path.normpath(session_dir))
        if trial_id is None:
            trial_id = session.trial_id if session is not None else None
        ctx = base.Context(session, thresholds, trial_id, args, schema.frozen, load_error, collector)
        results = checks.run_all(ctx, only)
    verdict, code = checks.verdict(results)
    tier, _ = base.tier_of(trial_id)
    stats = {}
    if session is not None:
        stats = {"rows": {name: len(st) for name, st in session.streams.items() if st.present},
                 "images": len(session.images), "decision_steps": len(session.decisions),
                 "segments": len(session.segments), "notes": session.notes,
                 "build_key": session.build_key(), "map_id": session.map_id}
    report = {"report_schema": REPORT_SCHEMA, "session_id": session_id, "session_dir": session_dir,
              "trial": trial_id, "tier": tier, "profile": args.profile, "legacy": bool(args.legacy),
              "validator_version": VALIDATOR_VERSION, "contract_version": reader.CONTRACT_VERSION,
              "contract_status": schema.status, "thresholds_sha256": thr_sha, "verdict": verdict, "exit_code": code,
              "checks": results, "stats": stats,
              "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    return report, code


def write_report(report, report_dir, sealed):
    root = check_report_dir(report_dir, sealed=sealed)
    out_dir = os.path.join(root, report["session_id"])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dc_validate.json")
    with open(path, "wb") as f:
        f.write((json.dumps(report, indent=1, default=_json_default) + "\n").encode("ascii"))
    return path


def _json_default(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def summary(report):
    lines = ["dc_validate %s (profile %s, trial %s, tier %s%s)" % (
        report["session_id"], report["profile"], report["trial"], report["tier"],
        ", legacy self-test" if report["legacy"] else "")]
    for r in report["checks"]:
        lines.append("  [%s] %-4s %-4s%s: %s" % (r["status"], r["id"], r["severity"],
                                                 " provisional" if r.get("provisional") else "", r["detail"]))
    lines.append("VERDICT: %s (exit %d)" % (report["verdict"], report["exit_code"]))
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    try:
        report, code = validate(args)
        if not args.no_report:
            report["report_path"] = write_report(report, args.report_dir, args.sealed_eval)
        if args.json:
            print(json.dumps(report, indent=1, default=_json_default))
        else:
            print(summary(report))
            if report.get("report_path"):
                print("report: %s" % report["report_path"])
        return code
    except Refused as e:
        print("refused: %s" % e)
        return checks.EXIT_VALIDATOR_ERROR
    except Exception:  # validator error: exit 3, never a silent PASS
        traceback.print_exc()
        return checks.EXIT_VALIDATOR_ERROR


if __name__ == "__main__":
    sys.exit(main())
