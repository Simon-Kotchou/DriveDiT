#!/usr/bin/env python3
"""E-19 writer failure semantics: analysis of an e19 play and/or an e19kill play (DC_ProbeEngine, DD-4-probe).

    python3 -I -B tools/probes/t0eng/e19_write.py <e19 play_dir> [--kill <e19kill play_dir>]
    python3 -I -B tools/probes/t0eng/e19_write.py --kill <e19kill play_dir>

r2 4.4 criterion (E-19): FileHandle.Write return on a read-only target; taskkill during commit -> "short or failed
writes detected; the commit log stays consistent". Both named experiments are required: without the read-only case
(c09 verdict ok, which needs a second e19 Play after attrib +R) and without an analysed --kill play the result is at
best INCOMPLETE, so TBD-3 cannot be frozen on a first-Play e19 alone. Checked here (e19_cases.csv):
  exact_returns          c01 (17 bytes), c02 (Write(s, 5)) and every c03 size return exactly what was asked, and
                         the c03 file length is the sum
  read_handle_detected   c05: a write through a READ handle does not return the requested count (or the open fails)
  missing_dir_detected   c06: when OpenFile in a missing directory returns null, that is detected; when it creates the
                         directory instead, that is INFO
  dir_as_file_detected   c07: opening a directory as a file fails or the write comes back short
  short_or_failed_writes_detected   all of the above together
  readonly_target_detected  c09: the write to the read-only target was refused or came back short (verdict ok);
                         "prepared" (the target was just created) or "observed" (it was not read-only) is INCOMPLETE
  overwrite_semantics_recorded  c10: whether WRITE over a longer file truncates it (dc-2.0 heartbeat, TBD-3); PASS
                         when the length after the rewrite was read back (truncates or keeps_tail), the value is the
                         result
  append_reopen_recorded c11: open-APPEND-write-close per line (the dc-2.0 commit-log pattern) adds exactly one line
                         per cycle: PASS when ok, FAIL when a return or the final length differs
INFO (TBD-3): c04 WriteLine (position after "abc", 5 = CR LF), c08 two WRITE handles on one file.
With --kill (e19kill play, Workbench killed by "taskkill /F" in the kill window), for both commit logs
(e19_commit.csv kept open, e19_commit_append.csv reopened per line):
  commit_log_consistent, commit_log_append_consistent
                         every committed chunk exists with n_rows data rows and n_bytes bytes, consecutive keys and
                         64-character payloads; a torn last commit line and uncommitted trailing chunks are detected
                         and discarded (INFO); a torn line before the end, a gap or a mismatch is a FAIL
  kill_happened          no run_footer.txt (the kill ended the play); a footer means the window closed without a
                         kill: INCOMPLETE
Without --kill: kill_run INCOMPLETE. Without an e19 play: writer_cases INCOMPLETE.

Exit codes: 0 PASS, 1 FAIL, 3 INCOMPLETE, 2 usage or input error.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import probe_common as pc  # noqa: E402

LABEL = "E-19 writer semantics"
CASE_COLUMNS = ("case_id", "open_ok", "bytes_requested", "bytes_returned", "pos_after", "length_after", "detected",
                "verdict")
COMMIT_COLUMNS = ["chunk_id", "first_key", "last_key", "n_rows", "n_bytes"]
CHUNK_COLUMNS = "key,world_time_ms,payload"
CHUNK_RE = re.compile(r"^chunk_(\d{5})\.csv$")
PAYLOAD_LENGTH = 64


def case_text(row):
    return "open=%s requested=%s returned=%s pos=%s length=%s detected=%s verdict=%s" % (
        row["open_ok"], row["bytes_requested"], row["bytes_returned"], row["pos_after"], row["length_after"],
        row["detected"], row["verdict"])


def analyse_cases(v, play):
    rows = play.csv("e19_cases", CASE_COLUMNS) if play.has("e19_cases.csv") else None
    if rows is None:
        v.check("short_or_failed_writes_detected", None, "e19_cases.csv missing")
        return
    cases = {}
    for r in rows:
        cases.setdefault(r["case_id"], r)

    results = []

    exact = [cid for cid in cases if cid in ("c01_ascii_row", "c02_length_param") or cid.startswith("c03_sizes")]
    needed = ("c01_ascii_row", "c02_length_param", "c03_sizes_total")
    missing = [cid for cid in needed if cid not in cases]
    not_ok = [cid for cid in exact if cases[cid]["verdict"] != "ok"]
    if missing:
        results.append(v.check("exact_returns", None, "case(s) not run: %s" % ", ".join(missing)))
    else:
        detail = "%d cases ok" % (len(exact) - len(not_ok))
        if not_ok:
            detail += "; not ok: " + "; ".join("%s %s" % (cid, case_text(cases[cid])) for cid in sorted(not_ok))
        results.append(v.check("exact_returns", not not_ok, detail))

    c05 = cases.get("c05_read_handle")
    if c05 is None:
        results.append(v.check("read_handle_detected", None, "c05 not run"))
    elif c05["verdict"] == "skipped":
        results.append(v.check("read_handle_detected", None, "c05 skipped: " + case_text(c05)))
    else:
        results.append(v.check("read_handle_detected", c05["detected"] == "1", case_text(c05)))

    c06 = cases.get("c06_missing_dir")
    if c06 is None:
        results.append(v.check("missing_dir_detected", None, "c06 not run"))
    elif c06["open_ok"] == "0":
        results.append(v.check("missing_dir_detected", c06["detected"] == "1", case_text(c06)))
    else:
        v.info("missing_dir_detected", "OpenFile created or accepted the missing directory: " + case_text(c06))

    c07 = cases.get("c07_dir_as_file")
    if c07 is None:
        results.append(v.check("dir_as_file_detected", None, "c07 not run"))
    else:
        results.append(v.check("dir_as_file_detected", c07["detected"] == "1", case_text(c07)))

    if False in results:
        overall = False
    elif None in results:
        overall = None
    else:
        overall = True
    v.check("short_or_failed_writes_detected", overall, "c01-c03 exact, c05 / c06 (null) / c07 detected")

    c04 = cases.get("c04_writeline")
    if c04:
        crlf = "CR LF" if c04["pos_after"] == "5" else "not CR LF"
        v.info("c04_writeline", "position after WriteLine(\"abc\") = %s (%s); never used for data" % (
            c04["pos_after"], crlf))
    for cid in ("c08_second_writer_h1", "c08_second_writer_h2"):
        if cid in cases:
            v.info(cid, case_text(cases[cid]))
    c09 = cases.get("c09_readonly_target")
    if c09 is None:
        v.check("readonly_target_detected", None, "c09 not run")
    else:
        meaning = {"prepared": "target created: attrib +R it and run e19 again in a later Play",
                   "observed": "the target accepted the write (it is not read-only): attrib +R it and run e19 again",
                   "ok": "write to the read-only target detected"}.get(c09["verdict"], c09["verdict"])
        ok = None
        if c09["verdict"] == "ok":
            ok = c09["detected"] == "1"
        v.check("readonly_target_detected", ok, "%s - %s" % (case_text(c09), meaning))

    c10 = cases.get("c10_overwrite_shorter")
    if c10 is None:
        v.check("overwrite_semantics_recorded", None, "c10 not run (heartbeat rewrite pattern unmeasured)")
    elif c10["verdict"] in ("truncates", "keeps_tail"):
        v.check("overwrite_semantics_recorded", True, "WRITE over a 17-byte file with 5 bytes: %s (%s)" % (
            c10["verdict"], case_text(c10)))
    else:
        v.check("overwrite_semantics_recorded", None, "length after the rewrite not read back: " + case_text(c10))

    c11 = cases.get("c11_append_reopen")
    if c11 is None:
        v.check("append_reopen_recorded", None, "c11 not run (commit-log pattern unmeasured)")
    else:
        v.check("append_reopen_recorded", c11["verdict"] == "ok", "open-APPEND-write-close per line: " + case_text(c11))


def read_commit_log(path):
    """-> (committed rows as dicts, torn line text or None, problems). Only the last line may be torn."""
    with open(path, "rb") as f:
        data = f.read()
    text = data.decode("ascii", errors="replace")
    lines = text.split("\n")
    complete = lines[:-1]           # each terminated by LF
    tail = lines[-1]                # text after the last LF: a torn line, or "" for a clean end
    problems = []
    if not complete or complete[0] != ",".join(COMMIT_COLUMNS):
        problems.append("commit log header missing or wrong")
        return [], tail or None, problems
    rows = []
    for no, line in enumerate(complete[1:], 2):
        parts = line.split(",")
        values = [pc.to_int(p) for p in parts]
        if len(parts) != len(COMMIT_COLUMNS) or None in values:
            problems.append("line %d malformed before the end of the log: %r" % (no, line))
            continue
        rows.append(dict(zip(COMMIT_COLUMNS, values)))
    return rows, (tail if tail else None), problems


def check_chunk(chunk_dir, row):
    """Problems of one committed chunk (empty list = consistent)."""
    path = os.path.join(chunk_dir, "chunk_%05d.csv" % row["chunk_id"])
    if not os.path.isfile(path):
        return ["chunk %d committed but missing" % row["chunk_id"]]
    with open(path, "rb") as f:
        data = f.read()
    problems = []
    if len(data) != row["n_bytes"]:
        problems.append("chunk %d has %d bytes, committed %d" % (row["chunk_id"], len(data), row["n_bytes"]))
    text = data.decode("ascii", errors="replace")
    lines = text.split("\n")
    if lines[-1] != "":
        problems.append("chunk %d does not end with LF" % row["chunk_id"])
    lines = lines[:-1]
    if not lines or lines[0] != CHUNK_COLUMNS:
        problems.append("chunk %d header wrong" % row["chunk_id"])
        return problems
    body = lines[1:]
    if len(body) != row["n_rows"]:
        problems.append("chunk %d has %d rows, committed %d" % (row["chunk_id"], len(body), row["n_rows"]))
    keys = []
    for line in body:
        parts = line.split(",")
        if len(parts) != 3 or len(parts[2]) != PAYLOAD_LENGTH:
            problems.append("chunk %d row malformed: %r" % (row["chunk_id"], line[:40]))
            break
        keys.append(pc.to_int(parts[0]))
    if keys and keys != list(range(row["first_key"], row["last_key"] + 1)):
        problems.append("chunk %d keys %s..%s, committed %d..%d" % (
            row["chunk_id"], keys[0], keys[-1], row["first_key"], row["last_key"]))
    return problems


def analyse_kill(v, play):
    for name, check in (("e19_commit.csv", "commit_log_consistent"),
                        ("e19_commit_append.csv", "commit_log_append_consistent")):
        analyse_commit_log(v, play, name, check)
    if play.footer is None:
        v.check("kill_happened", True, "no run_footer.txt: the play was killed during the kill window")
    else:
        v.check("kill_happened", None, "run_footer.txt present (status=%s reason=%s): no kill in the window" % (
            play.footer.get("status"), play.footer.get("reason")))


def analyse_commit_log(v, play, log_name, check):
    """One commit log of an e19kill play against the chunk files."""
    if not play.has(log_name):
        v.check(check, None, "%s missing" % log_name)
    else:
        rows, torn, problems = read_commit_log(play.path(log_name))
        chunk_dir = play.path("chunks")
        files = {}
        if os.path.isdir(chunk_dir):
            for name in os.listdir(chunk_dir):
                m = CHUNK_RE.match(name)
                if m:
                    files[int(m.group(1))] = name
        committed_ids = [r["chunk_id"] for r in rows]
        if committed_ids != list(range(1, len(committed_ids) + 1)):
            problems.append("committed chunk ids are not 1..%d in order" % len(committed_ids))
        previous_last = 0
        bad_chunks = 0
        for r in rows:
            if r["first_key"] != previous_last + 1:
                problems.append("chunk %d starts at key %d, previous ended at %d" % (
                    r["chunk_id"], r["first_key"], previous_last))
            previous_last = r["last_key"]
            chunk_problems = check_chunk(chunk_dir, r)
            if chunk_problems:
                bad_chunks += 1
                problems.extend(chunk_problems)
        max_committed = max(committed_ids) if committed_ids else 0
        uncommitted = sorted(i for i in files if i not in committed_ids)
        inner = [i for i in uncommitted if i < max_committed]
        if inner:
            problems.append("uncommitted chunk(s) before the last commit: %s" % inner)
        detail = "%d of %d committed chunks verified" % (len(rows) - bad_chunks, len(rows))
        if torn is not None:
            v.info("torn_commit_line", "%s: last commit line torn and discarded: %r" % (log_name, torn))
        if uncommitted:
            v.info("uncommitted_chunks", "%s: %d trailing chunk file(s) without a commit line discarded: %s" % (
                log_name, len(uncommitted), ", ".join(files[i] for i in uncommitted)))
        if problems:
            detail += "; problems: " + "; ".join(problems[:5])
        v.check(check, not problems and bool(rows), detail if rows else "no committed chunk")


def main(argv=None):
    ap = argparse.ArgumentParser(description="E-19 writer failure semantics (r2 4.4)")
    ap.add_argument("play_dir", nargs="?", help="an e19 play directory ($profile/DC_Probes/e19/<launch_id>/play_NN)")
    ap.add_argument("--kill", metavar="PLAY_DIR", help="an e19kill play directory (Workbench killed in the window)")
    args = ap.parse_args(argv)
    if not args.play_dir and not args.kill:
        ap.error("give an e19 play directory, --kill <e19kill play directory>, or both")
    targets = [d for d in (args.play_dir, args.kill) if d]
    v = pc.Verdict(LABEL, " + ".join(os.path.realpath(t) for t in targets))
    if args.play_dir:
        play = pc.Play(args.play_dir)
        analyse_cases(v, play)
        v.check_footer(play)
    else:
        v.check("writer_cases", None, "no e19 play given: the writer cases c01-c11 are not analysed")
    if args.kill:
        analyse_kill(v, pc.Play(args.kill))
    else:
        v.check("kill_run", None, "no --kill play given: r2 4.4 needs the taskkill-during-commit run (e19kill)")
    return v.report()


if __name__ == "__main__":
    sys.exit(pc.run_cli(main))
