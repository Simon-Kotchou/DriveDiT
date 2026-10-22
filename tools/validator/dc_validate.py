#!/usr/bin/env python3
"""dc_validate: validator of capture contract dc-2.0 sessions (DD-5-min; CAPTURE_PROGRAM r2 section 5).

    python3 -I -B tools/validator/dc_validate.py --session DIR [--profile capture] [--trial ID]
        [--report-dir DIR | --no-report] [--engine-logs DIR] [--route FILE ...] [--routes-dir DIR]
        [--launch-record FILE] [--allow-list FILE] [--vehicle-card FILE] [--thresholds FILE] [--sealed-eval]
        [--checks C01,C06,...] [--image-suffix S] [--json] [--legacy]

What it checks: C01 C02 C05 C06 C07 C08 C10 C11 C12 C13a C13b C15 C17 C20 C21 C28 C31 C34 C35 (the T0-MIN and T1
set; tools/validator/checks/ documents each). The session is read by data/dc2/reader.py (strict=False, so every
problem is reported, not raised). Thresholds are keyed by pin: thresholds_dc2.<pin_id>.json (a <pin_id>-<suffix>
pin of the same build falls back to thresholds_dc2.<pin_id>.json), each CAPTURE_PROGRAM r2 5.2 verbatim; a session
whose pin has no thresholds file is refused (exit 3); --thresholds overrides the choice. Severities and trial rules
follow the session's trial.trial_id, or a --trial of the same or a higher tier (a lower one is refused): C02 dirty
flag from T1, C10 T0 limit, C12 / C13b SOFT, C21 SOFT before T1 and HARD from T1, C31 from T1, C17 SOFT at T1 /
HARD at T2-T3 / INFO elsewhere. A check that depends on a TBD-by-probe value is marked "provisional" while dc-2.0
is a DRAFT. --profile capture is the only profile so far (eval_lockstep and eval_async come with the bridge, DD-8).
--image-suffix names the TBD-4 file suffix the engine appends to an img path (one of the reader's IMAGE_SUFFIXES,
default .png); the harvester passes its --screenshot-suffix through it.

The sealed admission runs before any session file is parsed: an Everon session (map_id, terrain, an Eden-rooted
root_world, the session id, a path under heldout/ or SEALED; also with --legacy) and every --route file (map_id
everon, an everon_* split, a sealed path) need --sealed-eval, else exit 3. An Everon route met through
--routes-dir is refused the same way, never skipped.

--legacy runs the dc-1.0 self-test instead (checks/legacy.py): a dc-1.0 session must FAIL for its known reasons.

Exit codes (r1 [TL] 5): 0 PASS, 1 FAIL (a HARD check failed), 2 PASS with warnings (a SOFT check failed, or a HARD
or SOFT check could not be evaluated, e.g. no engine logs or no launch record), 3 validator error or refusal (bad
arguments, an Everon session or route without --sealed-eval, a path on C:, a report directory outside /mnt/e, a
pin without thresholds, a lower --trial).

Report: <report-dir>/<session directory name>/dc_validate.json (the directory name is the session id of a valid
session; it must be a plain name, never a path) = {report_schema, session_id, session_dir, trial, tier, profile,
validator_version, contract_version, contract_status, thresholds_file, thresholds_sha256, verdict, exit_code,
checks: [{id, severity, status, value, threshold, detail, provisional, not_evaluated?}], stats, created_utc}.
Reports are written only under /mnt/e/ (default /mnt/e/DriveDiT/reports), never through a symlink out of the
report root; an Everon report only under a SEALED directory (L12). The validator never writes anywhere else, and
reads no C: path.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import traceback

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for path in (HERE, REPO):
    if path not in sys.path:
        sys.path.insert(0, path)
import checks  # noqa: E402
from checks import base  # noqa: E402
from checks.base import Refused  # noqa: E402
from checks.images import ImageCollector  # noqa: E402
from checks.legacy import run_legacy  # noqa: E402

VALIDATOR_VERSION = "dc_validate/0.2 (DD-5-min)"
REPORT_SCHEMA = "drivedit.dc-validate-report/1"
THRESHOLDS_DIR = HERE
THRESHOLDS_NAME = "thresholds_dc2.%s.json"
# dc-1.0 sessions carry no pin. They were captured on 1.8 builds, so the legacy self-test judges them with the
# interim 1.8.0.13 thresholds (it reads only C10's T0 limit).
LEGACY_THRESHOLDS_PIN = "1.8.0.13"
PIN_ID_RE = re.compile(r"^[0-9]+(\.[0-9]+){1,3}(-[0-9A-Za-z]+)*$")
REPORT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
DEFAULT_REPORT_DIR = "/mnt/e/DriveDiT/reports"
REPORT_ROOT = "/mnt/e/"
PROFILES = ("capture", "eval_lockstep", "eval_async")


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


def thresholds_path(pin_id):
    """thresholds_dc2.<pin_id>.json for a pin id, else that of its base pin (<pin_id>-<suffix> pins of one build,
    e.g. 1.8.0.13-cold); Refused when neither exists or the pin id is not a pin id."""
    if not (isinstance(pin_id, str) and PIN_ID_RE.match(pin_id)):
        raise Refused("pin.pin_id %r is not a pin id: no thresholds file can be chosen" % (pin_id,))
    for candidate in (pin_id, pin_id.split("-", 1)[0]):
        path = os.path.join(THRESHOLDS_DIR, THRESHOLDS_NAME % candidate)
        if os.path.isfile(path):
            return path
    raise Refused("no thresholds file for pin %s (%s): a session of a build without thresholds is not validated"
                  % (pin_id, THRESHOLDS_NAME % pin_id))


def load_routes(paths, sealed_eval):
    """{route_id: (Route, path, None)} of the --route files, each through drivedit_nav.load_route before any
    session file is read: an Everon route or a sealed path without --sealed-eval, and a file that is not a valid
    route, are refused."""
    import drivedit_nav
    routes = {}
    for path in paths:
        try:
            route = drivedit_nav.load_route(path, sealed_eval=sealed_eval)
        except drivedit_nav.SealedMapError as e:
            raise Refused(str(e))
        except (OSError, ValueError) as e:
            raise Refused("--route %s: %s" % (path, e))
        routes[route.route_id] = (route, path, None)
    return routes


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
    ap.add_argument("--thresholds", default=None,
                    help="thresholds file (default: thresholds_dc2.<pin_id>.json of the session's pin)")
    ap.add_argument("--sealed-eval", action="store_true", help="the sealed evaluation path (Everon)")
    ap.add_argument("--checks", default=None, help="comma-separated subset of check ids")
    ap.add_argument("--image-suffix", default=None,
                    help="TBD-4: the suffix of the image files, one of the reader's IMAGE_SUFFIXES (default .png)")
    ap.add_argument("--json", action="store_true", help="print the report JSON instead of the summary")
    ap.add_argument("--legacy", action="store_true", help="dc-1.0 self-test: the session must FAIL")
    return ap.parse_args(argv)


def validate(args):
    """(report dict, exit code) for parsed arguments; raises Refused."""
    if args.profile != "capture":
        raise Refused("profile %s is not implemented yet (DD-5-min ships the capture profile)" % args.profile)
    session_dir = check_read_path(args.session, "session")
    for name in ("engine_logs", "routes_dir", "launch_record", "allow_list", "vehicle_card", "thresholds"):
        value = getattr(args, name)
        if value:
            setattr(args, name, check_read_path(value, name.replace("_", "-")))
    args.route = [check_read_path(r, "route") for r in args.route]
    if not args.no_report:
        check_report_dir(args.report_dir, sealed=args.sealed_eval)
        report_name(session_dir)
    reader = base.load_reader()
    schema = reader.load_schema()
    image_suffix = reader.IMAGE_SUFFIX if args.image_suffix is None else args.image_suffix
    if image_suffix not in reader.IMAGE_SUFFIXES:
        raise Refused("--image-suffix %r is not one of the TBD-4 names %s" % (image_suffix, reader.IMAGE_SUFFIXES))
    only = set(c.strip() for c in args.checks.split(",")) if args.checks else None
    if only and only - set(checks.CHECK_IDS):
        raise Refused("unknown check id(s) %s" % sorted(only - set(checks.CHECK_IDS)))
    routes = load_routes(args.route, args.sealed_eval)
    session = None
    collector = ImageCollector()
    load_error = None
    trial_id = args.trial
    thr_path = args.thresholds
    if args.legacy:
        try:
            reader.check_paths(session_dir, args.sealed_eval)   # heldout/ and SEALED need --sealed-eval here too
        except reader.SealedMapError as e:
            raise Refused(str(e))
        thr_path = thr_path or thresholds_path(LEGACY_THRESHOLDS_PIN)
        thresholds, thr_sha = load_thresholds(thr_path)
        results = run_legacy(session_dir, schema, thresholds)
        session_id = os.path.basename(os.path.normpath(session_dir))
        trial_id = trial_id or "legacy"
    else:
        try:
            session = reader.open_session(session_dir, sealed_eval=args.sealed_eval, strict=False,
                                          vehicle_card=args.vehicle_card, image_visitor=collector.visit,
                                          image_suffix=image_suffix)
        except reader.SealedMapError as e:
            raise Refused(str(e))
        except reader.LegacySessionError as e:
            load_error = "%s (run --legacy for the dc-1.0 self-test)" % e
        except reader.ReaderError as e:
            load_error = str(e)
        session_id = session.session_id if session is not None else os.path.basename(os.path.normpath(session_dir))
        if session is not None and session.trial_id is not None and trial_id is not None:
            if base.tier_of(trial_id)[1] < base.tier_of(session.trial_id)[1]:
                raise Refused("--trial %s is a lower tier than the session's trial %s: its rules would be relaxed"
                              % (trial_id, session.trial_id))
        if trial_id is None:
            trial_id = session.trial_id if session is not None else None
        if session is not None:
            thr_path = thr_path or thresholds_path(session.meta_get("pin.pin_id"))
        if thr_path:
            thresholds, thr_sha = load_thresholds(thr_path)
        else:                   # unreadable session: every check but C20 reports it without thresholds
            thresholds, thr_sha = {}, None
        ctx = base.Context(session, thresholds, trial_id, args, schema.frozen, load_error, collector, routes)
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
              "contract_status": schema.status, "thresholds_file": thr_path and os.path.basename(thr_path),
              "thresholds_sha256": thr_sha, "verdict": verdict, "exit_code": code, "checks": results,
              "stats": stats,
              "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    return report, code


def report_name(session_dir):
    """The report directory name: the session directory's own name (the session id of a valid session), which must
    be a plain name. The session.json session_id is data and never becomes a path."""
    name = os.path.basename(os.path.normpath(session_dir))
    if not REPORT_NAME_RE.match(name):
        raise Refused("session directory name %r is not a plain name; no report path is built from it" % name)
    return name


def write_report(report, report_dir, sealed):
    """Write <report_dir>/<session directory name>/dc_validate.json, re-checking after every path step that the
    target stays under the report root (no symlink out of it) and, for Everon, under a SEALED directory."""
    root = check_report_dir(report_dir, sealed=sealed)
    out_dir = os.path.join(root, report_name(report["session_dir"]))
    if not _under(out_dir, root.rstrip("/") + "/"):
        raise Refused("report directory %s resolves outside %s" % (out_dir, root))
    check_report_dir(out_dir, sealed=sealed)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dc_validate.json")
    if os.path.realpath(path) != os.path.join(os.path.realpath(out_dir), "dc_validate.json") \
            or not _under(path, root.rstrip("/") + "/"):
        raise Refused("report file %s resolves outside %s" % (path, root))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o644)
    with os.fdopen(fd, "wb") as f:
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
