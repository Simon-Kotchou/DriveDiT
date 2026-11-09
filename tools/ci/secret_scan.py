#!/usr/bin/env python3
"""Generic privacy scan for a public repository (stdlib only).

Flags, in every git-tracked text file (default) or in the files named on the command line:

  steam64       Steam64 account ids: 7656119 followed by 10 digits
  email         e-mail addresses, except noreply ones (noreply@anthropic.com, GitHub noreply addresses)
  win-profile   Windows profile paths: X:\\Users\\<name>, X:/Users/<name>, /mnt/x/Users/<name> (any drive letter)
  home-path     Linux home paths: /home/<name>, with or without a child path
  private-ip    RFC 1918 IPv4 addresses (10/8, 172.16/12, 192.168/16). Loopback and the RFC 5737 documentation
                ranges are not RFC 1918 and pass; so does a dotted quad in a version context (`version = "..."`,
                `==`, `>=`, or right after `<name>-` as in a wheel file name), which is a version, not an address
  windows-build Windows release or build strings: Windows 10/11 followed by a release or build (a release like
                21H2, or a five-digit build), and `Version 10.0.<build>` as `ver` and `systeminfo` print it
  wsl-version   WSL versions (WSL followed by a three-part version) and WSL kernel release strings
  machine-path  machine-absolute paths (DD-22): a WSL drive mount /mnt/<letter>, alone or with a child path, and a
                Windows drive path <letter>:\\ or <letter>:/. A machine's paths live in its untracked local config
                (tools/local_config.py), never in the repo. The placeholder drives X, Y and Z pass (the synthetic test
                config and the examples use them, as /home/x is a placeholder home)

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
import ipaddress
import os
import re
import subprocess
import sys

PATTERNS = {
    "steam64": re.compile(r"(?<!\d)7656119\d{10}(?!\d)"),
    "email": re.compile(r"(?<![\w.%+-])[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"),
    "win-profile": re.compile(r"(?i)(?:\b[A-Z]:[\\/]+|/mnt/[a-z]/)Users[\\/]+(?P<name>[^\\/\s\"'`<>|:*?]+)"),
    # The name ends at "/" or at a quote, space, separator or bracket, or at the end of the line (a bare /home/<name>).
    "home-path": re.compile(r"(?<![\w.~-])/home/(?P<name>[^/\s\"'`<>|:*?,;=()\[\]]+)"),
    # A dotted quad of 0-255 octets, not inside a longer dotted number; findings() keeps only RFC 1918 addresses.
    "private-ip": re.compile(r"(?<![\w.])(?P<name>(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d))(?!\w|\.\d)"),
    "windows-build": re.compile(
        r"(?i)\bWindows[\s_-]*1[01]\b\W{0,3}(?:(?:Pro|Home|Enterprise|Education|IoT|LTSC|N)\W{1,3}){0,2}"
        r"(?:version\s+|build\s+)?(?:\d{2}H[12]|\d{5}(?:\.\d+)?)\b"
        r"|\bVersion:?\s+10\.0\.\d{5}\b"
    ),
    "wsl-version": re.compile(
        r"(?i)\bWSL\s*2?\s*(?:version:?\s*)?v?\d+\.\d+\.\d+|\b\d+\.\d+\.\d+(?:\.\d+)?-microsoft-standard\b"
    ),
    # The drive letter is the name: /mnt/<letter> not inside a longer path or word, or <letter>:\ / <letter>:/ not
    # after a word character, %, $ or a backslash (so https://, a %s:\ format, a %VAR:/=\% substitution and a regex
    # such as \d:\d are not paths).
    "machine-path": re.compile(
        r"(?<![\w./~-])/mnt/(?P<name>[A-Za-z])(?![\w-])|(?<![\w%$\\])(?P<drive>[A-Za-z]):[\\/]"
    ),
}

# Drive letters that are placeholders, never a machine's drive (machine-path).
PLACEHOLDER_DRIVES = ("x", "y", "z")

# RFC 1918 private networks: 10/8, 172.16/12, 192.168/16 (as integers, so this file holds no dotted quad). Loopback
# (127/8), link-local and the RFC 5737 documentation ranges are not in them.
PRIVATE_NETWORKS = tuple(ipaddress.IPv4Network(n) for n in ((0x0A000000, 8), (0xAC100000, 12), (0xC0A80000, 16)))

# Text right before a dotted quad that makes it a version: a TOML/YAML version field, a pip specifier, or a
# "<name>-" prefix as in a wheel or sdist file name (nvidia_curand-10.4.0.35-py3-none-...).
VERSION_CONTEXT = re.compile(r"""(?i)(?:\bversion\s*[:=]\s*["']?|[=<>~!]=\s*|\w-)$""")

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

# Reviewed exceptions, as (tracked path, kind, matched name, or the whole match for a kind without a name). Keep
# this list short and specific, with the reason.
ALLOWLIST: set[tuple[str, str, str]] = {
    # A version-shaped pin id in DD-0's pin-id validator test (PinIdTests.test_valid_ids): a version, not an address.
    # Assembled, so this file holds no dotted quad.
    ("tools/pin/tests/test_pin.py", "private-ip", ".".join(("10", "0", "1", "2"))),
}

BOMS = (  # UTF-32 first: its little-endian BOM begins with the UTF-16 one
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
    (codecs.BOM_UTF8, "utf-8"),
)
# (encoding, code-unit width): the projection decodes at every byte alignment of the unit, so wide text that starts
# after an odd-length binary header is read too.
PROJECTION_ENCODINGS = (("utf-8", 1), ("utf-16-le", 2), ("utf-16-be", 2), ("utf-32-le", 4), ("utf-32-be", 4))
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
    """The printable runs of `data` read as UTF-8, UTF-16 and UTF-32 (both byte orders, every alignment), one per
    line."""
    runs = []
    for enc, width in PROJECTION_ENCODINGS:
        for offset in range(width):
            runs.extend(PRINTABLE_RUN.findall(data[offset:].decode(enc, "replace")))
    return "\n".join(runs)


def read_texts(path: str) -> list[tuple[str, str]] | None:
    """The texts to scan for a file as [(text, unit)], unit "" for lines or "run" for projection runs; None if not a
    file. A file with a BOM is decoded by it and is also projected, because a BOM names the encoding of the start
    only (a UTF-8 line appended to a UTF-16 log stays contiguous in the bytes). A file without a BOM that holds a NUL
    byte is projected; any other file is UTF-8."""
    if path == "-":
        return [(sys.stdin.read(), "")]
    if os.path.islink(path) or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    for bom, enc in BOMS:
        if data.startswith(bom):
            return [(data[len(bom):].decode(enc, "replace"), ""), (text_projection(data), "run")]
    if b"\0" in data:
        return [(text_projection(data), "run")]
    return [(data.decode("utf-8", "replace"), "")]


def is_placeholder(name: str) -> bool:
    """A placeholder account name, also when a sentence full stop or a file extension follows it (/home/x.json)."""
    stripped = name.rstrip(".")
    candidates = (name, stripped, re.sub(r"\.\w+$", "", stripped))
    return any(c and PLACEHOLDER_NAMES.match(c) for c in candidates)


def redact(s: str) -> str:
    return s[:4] + "..." if len(s) > 4 else s[:1] + "..."


def is_private_address(text: str, before: str) -> bool:
    """An RFC 1918 address, not a version (VERSION_CONTEXT)."""
    if VERSION_CONTEXT.search(before):
        return False
    addr = ipaddress.ip_address(text)
    return any(addr in net for net in PRIVATE_NETWORKS)


def findings(path: str, text: str):
    for lineno, line in enumerate(text.splitlines(), 1):
        for kind, rx in PATTERNS.items():
            for m in rx.finditer(line):
                hit = m.group(0)
                if kind == "email" and (ALLOWED_EMAILS.match(hit) or RESERVED_EMAIL_DOMAIN.search(hit)):
                    continue
                if kind == "private-ip" and not is_private_address(hit, line[: m.start()]):
                    continue
                name = m.groupdict().get("name")
                if kind == "machine-path":
                    name = name or m.group("drive")
                    if name.lower() in PLACEHOLDER_DRIVES:
                        continue
                elif kind != "private-ip" and name is not None and is_placeholder(name):
                    continue
                if (path, kind, hit if name is None else name) in ALLOWLIST:
                    continue
                yield lineno, kind, hit


def main(argv: list[str]) -> int:
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__)
        return 0
    paths = argv or tracked_files()
    n_files = n_hits = 0
    for path in paths:
        texts = read_texts(path)
        if texts is None:
            continue
        n_files += 1
        seen = set()  # a projection run repeats what the decoded text or another alignment already reported
        for text, unit in texts:
            for lineno, kind, hit in findings(os.path.normpath(path), text):
                if unit == "run" and (kind, hit) in seen:
                    continue
                seen.add((kind, hit))
                n_hits += 1
                print("%s:%s%d: %s: %s" % (path, unit, lineno, kind, redact(hit)))
    print("secret_scan: %d file(s) scanned, %d finding(s)" % (n_files, n_hits), file=sys.stderr)
    return 1 if n_hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
