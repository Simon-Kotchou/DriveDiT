#!/usr/bin/env python3
"""Generic privacy scan for a public repository (stdlib only).

Flags, in every git-tracked text file (default) or in the files named on the command line:

  steam64       Steam64 account ids: 7656119 followed by 10 digits
  email         e-mail addresses, except noreply ones (noreply@anthropic.com, GitHub noreply addresses)
  win-profile   Windows profile paths: C:\\Users\\<name>, C:/Users/<name>, /mnt/c/Users/<name>
  home-path     Linux home paths: /home/<name>/

The patterns are generic on purpose: this file names no real account, host or address. Matches are
printed redacted (kind, file, line, first characters only) so a CI log never repeats what it found.

No file is skipped for looking binary. A file with a byte-order mark is decoded as UTF-8/16/32. A file
that holds NUL bytes without one (binary, or UTF-16/32 without a BOM) is scanned as a text projection:
the runs of at least 6 printable characters in its UTF-8, UTF-16 and UTF-32 readings, one run per
line. Findings there are reported as `path:runN`.

    python3 tools/ci/secret_scan.py               # all tracked files
    python3 tools/ci/secret_scan.py a.txt b.md    # just these files ("-" reads stdin)

Exit status: 0 clean, 1 findings, 2 usage or git error.
"""

from __future__ import annotations

import codecs
import os
import re
import subprocess
import sys

PATTERNS = {
    "steam64": re.compile(r"(?<!\d)7656119\d{10}(?!\d)"),
    "email": re.compile(r"(?<![\w.%+-])[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"),
    "win-profile": re.compile(r"(?i)(?:\b[A-Z]:[\\/]+|/mnt/[a-z]/)Users[\\/]+(?P<name>[^\\/\s\"'`<>|:*?]+)"),
    "home-path": re.compile(r"(?<![\w.~-])/home/(?P<name>[^/\s\"'`<>|:*?]+)/"),
}

# E-mail addresses that are public by design.
ALLOWED_EMAILS = re.compile(
    r"(?i)^(?:noreply@anthropic\.com|noreply@github\.com|(?:\d+\+)?[A-Za-z0-9-]+@users\.noreply\.github\.com)$"
)

# Domains reserved for documentation and tests (RFC 2606, RFC 6761): never a real mailbox.
RESERVED_EMAIL_DOMAIN = re.compile(r"(?i)@(?:[A-Za-z0-9-]+\.)*(?:example\.(?:com|net|org)|example|invalid|test|localhost)$")

# Profile/home "names" that are placeholders or system accounts, never a person.
PLACEHOLDER_NAMES = re.compile(
    r"(?i)^(?:x|y|user|username|name|you|me|someone|example|<[^>]*>|\$\{?\w+\}?|%\w+%|\.\.\.|\*|public|default|all users)$"
)

# Reviewed exceptions, as (tracked path, kind, matched name). Keep this list short and specific.
ALLOWLIST = {
    ("Dockerfile", "home-path", "drivedit"),          # the image's own account (useradd ... drivedit)
    ("docker-compose.yml", "home-path", "drivedit"),  # mounts into that account's home
}

BOMS = (  # UTF-32 first: its little-endian BOM begins with the UTF-16 one
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
    (codecs.BOM_UTF8, "utf-8"),
)
PROJECTION_ENCODINGS = ("utf-8", "utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be")
# Printable runs: no C0 controls (tab allowed), no DEL, no U+FFFD from undecodable bytes.
PRINTABLE_RUN = re.compile(r"[^\x00-\x08\x0a-\x1f\x7f\ufffd]{6,}")


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        print("secret_scan: cannot list tracked files: %s" % e, file=sys.stderr)
        sys.exit(2)
    return [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def text_projection(data: bytes) -> str:
    """The printable runs of `data` read as UTF-8, UTF-16 and UTF-32 (both byte orders), one per line."""
    runs = []
    for enc in PROJECTION_ENCODINGS:
        runs.extend(PRINTABLE_RUN.findall(data.decode(enc, "replace")))
    return "\n".join(runs)


def read_text(path: str) -> tuple[str, str] | None:
    """(text, unit) for a scannable file, unit "" for lines or "run" for a projection; None if not a file."""
    if path == "-":
        return sys.stdin.read(), ""
    if os.path.islink(path) or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    for bom, enc in BOMS:
        if data.startswith(bom):
            return data[len(bom):].decode(enc, "replace"), ""
    if b"\0" in data:
        return text_projection(data), "run"
    return data.decode("utf-8", "replace"), ""


def redact(s: str) -> str:
    return s[:4] + "..." if len(s) > 4 else s[:1] + "..."


def findings(path: str, text: str):
    for lineno, line in enumerate(text.splitlines(), 1):
        for kind, rx in PATTERNS.items():
            for m in rx.finditer(line):
                hit = m.group(0)
                if kind == "email" and (ALLOWED_EMAILS.match(hit) or RESERVED_EMAIL_DOMAIN.search(hit)):
                    continue
                name = m.groupdict().get("name")
                if name is not None and (PLACEHOLDER_NAMES.match(name) or (path, kind, name) in ALLOWLIST):
                    continue
                yield lineno, kind, hit


def main(argv: list[str]) -> int:
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__)
        return 0
    paths = argv or tracked_files()
    n_files = n_hits = 0
    for path in paths:
        read = read_text(path)
        if read is None:
            continue
        text, unit = read
        n_files += 1
        for lineno, kind, hit in findings(os.path.normpath(path), text):
            n_hits += 1
            print("%s:%s%d: %s: %s" % (path, unit, lineno, kind, redact(hit)))
    print("secret_scan: %d file(s) scanned, %d finding(s)" % (n_files, n_hits), file=sys.stderr)
    return 1 if n_hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
