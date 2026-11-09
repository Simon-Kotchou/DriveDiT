#!/usr/bin/env python3
"""Generic privacy scan for a public repository (stdlib only).

Flags, in every git-tracked text file (default) or in the files named on the command line:

  steam64       Steam64 account ids: 7656119 followed by 10 digits
  email         e-mail addresses, except noreply ones (noreply@anthropic.com, GitHub noreply addresses)
  win-profile   Windows profile paths: <X>:\\Users\\<name>, <X>:/Users/<name>, /mnt/<x>/Users/<name> (any drive
                letter), and the environment forms %SystemDrive%\\Users\\<name>, $env:SystemDrive\\Users\\<name>,
                ${HOMEDRIVE}/Users/<name> with a literal name
  home-path     Linux home paths: /home/<name>, with or without a child path
  private-ip    RFC 1918 IPv4 addresses (10/8, 172.16/12, 192.168/16). Loopback and the RFC 5737 documentation
                ranges are not RFC 1918 and pass; so does a dotted quad in a version context (`version = "..."`,
                `==`, `>=`, or right after `<name>-` as in a wheel file name), which is a version, not an address
  windows-build Windows release or build strings: Windows 10/11 followed by a release or build (a release like
                21H2, or a five-digit build), and `Version 10.0.<build>` as `ver` and `systeminfo` print it
  wsl-version   WSL versions (WSL followed by a three-part version) and WSL kernel release strings
  machine-path  machine-absolute paths (DD-22): a WSL drive mount /mnt/<letter>, alone or with a child path, and a
                Windows drive path <letter>:\\ or <letter>:/. A machine's paths live in its untracked local config
                (tools/local_config.py), never in the repo. The placeholder drives X, Y and Z pass only in the
                synthetic config's own folder (tests/fixtures/local_config/) and in *.example files; anywhere else
                only in that config's synthetic forms: the bare drive (/mnt/x, X:\\), an ellipsis (X:\\...) or a path
                under one of its roots
                (DataRoot, PlanRoot, ...; read from tests/fixtures/local_config/placeholder.local.env). A JavaScript
                object key followed by a regular expression literal ({ e:/value/ }) is code, not a path
  wsl-path      WSL UNC paths \\\\wsl$\\<distro>\\... and \\\\wsl.localhost\\<distro>\\... (also JSON-escaped)
  wsl-distro    WSL distro names: after the -d or --distribution option of wsl.exe, as the value of WSL_DISTRO or
                WSL_DISTRO_NAME,
                and the versioned names of the WSL store distros (<name>-<version>). Placeholders pass (<distro>,
                %VAR%, $VAR, Placeholder-*, No-Such-*), as does everything in the synthetic config's folder

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
    "win-profile": re.compile(r"(?i)(?:\b[A-Z]:[\\/]+|/mnt/[a-z]/|%\w+%[\\/]+|\$env:\w+[\\/]+|\$\{?\w+\}?[\\/]+)"
                              r"Users[\\/]+(?P<name>[^\\/\s\"'`<>|:*?%$]+)"),
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
    # \\wsl$\<distro> or \\wsl.localhost\<distro> (two or more backslashes or slashes, so JSON escapes count too)
    "wsl-path": re.compile(r"(?i)(?:\\{2,}|//)wsl(?:\$|\.localhost)(?:\\+|/)(?P<name>[^\\/\s\"'`<>|:*?]+)"),
    # a distro name where a launcher or a config names one, or a versioned WSL store distro anywhere: the -d or
    # --distribution option of wsl(.exe), also after other options, with a space or =, quoted or not; WSL_DISTRO or
    # WSL_DISTRO_NAME set or used as a key; a JSON/YAML key such as "distro", "wslDistro" or "distribution"
    "wsl-distro": re.compile(
        r"(?:\b[Ww][Ss][Ll](?:\.[Ee][Xx][Ee])?(?:\s+-{1,2}[A-Za-z][\w-]*(?:[= ][^\s-]\S*)?)*?\s+(?:-d|--distribution)"
        r"(?:\s+|=)[\"']?"
        r"|\bWSL_DISTRO(?:_NAME)?(?:\s*[=:]\s*[\"']?|[\"']\s*:\s*[\"'])"
        r"|[\"'](?i:distro|wsl_?distro|distribution|distro_?name|wsl_?distro_?name)[\"']\s*:\s*[\"'])"
        r"(?P<name>[A-Za-z0-9%$<{][A-Za-z0-9._%${}<>-]*)"
        r"|\b(?:Ubuntu|Debian|Kali|kali-linux|openSUSE|SUSE|OracleLinux|AlmaLinux|FedoraLinux|archlinux)"
        r"[-_ ](?:\d+(?:\.\d+)*(?:-LTS)?|Leap-\d+(?:\.\d+)*|Tumbleweed)\b"
    ),
}

# Drive letters that are placeholders, never a machine's drive (machine-path): exempt in the synthetic config's own
# folder and in *.example files; elsewhere only as the bare drive or under one of SYNTHETIC_ROOTS.
PLACEHOLDER_DRIVES = ("x", "y", "z")
PLACEHOLDER_HOMES = ("tests/fixtures/local_config/",)
PLACEHOLDER_SUFFIXES = (".example",)
SYNTHETIC_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "tests", "fixtures", "local_config", "placeholder.local.env")


def synthetic_roots(path: str = SYNTHETIC_CONFIG) -> tuple[str, ...]:
    """The first path component of every placeholder-drive value of the synthetic config (DataRoot, PlanRoot, ...):
    the only places on X, Y and Z a file outside that folder may name. None without the file."""
    roots = set()
    try:
        with open(path, encoding="ascii") as fh:
            for line in fh:
                m = re.match(r"^[A-Z_]+=/mnt/([a-z])/([^/\n]+)", line)
                if m and m.group(1) in PLACEHOLDER_DRIVES:
                    roots.add(m.group(2).rstrip("\r"))
    except OSError:
        pass
    return tuple(sorted(roots))


SYNTHETIC_ROOTS = synthetic_roots()
# What may follow /mnt/<placeholder> or <placeholder>:\ outside the synthetic config's folder: nothing path-like (the
# bare drive) or a synthetic root as a whole component.
_BARE_DRIVE_TAIL = re.compile(r"(?:$|[^\w\\/.~-])|[\\/]+(?:$|[^\w\\/.~(-])")
# A string escape right after a placeholder drive that ends it (the bare drive in source text): \n, \r, \t, \" or
# \' not followed by a path character (a quoted X: with an escaped n after it is the bare drive; with otes after the
# n it is a path).
_ESCAPE_TAIL = re.compile(r"^\\[nrt\"'](?![\w.~(+-])")
# What may follow a synthetic root for it to be a whole path component: the end, a separator, white space, a quote,
# a closing bracket or punctuation, or a full stop that ends a sentence. Anything else (DataRoot+x, DataRoot2) is
# another name.
_COMPONENT_END = re.compile(r"$|[\\/\s\"'`)\]},;:>|]|\.(?:$|[\s\"'`)])")
# A JavaScript object key followed by a regular expression literal: { e:/value/ } or , e:/x/g }
_JS_KEY_BEFORE = re.compile(r"[{,]\s*$")
_JS_REGEX_AFTER = re.compile(r"^/(?:[^/\\\s]|\\.)+/[dgimsuy]*\s*[,};)]")
# Distro names that are placeholders.
PLACEHOLDER_DISTROS = re.compile(r"(?i)^(?:placeholder[-\w.]*|no-such[-\w.]*|example[-\w.]*|distro|<.*|%\w+%.*|"
                                 r"\$\{?\w+\}?.*|x|y|z)$")

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
# Reviewed exceptions for one exact piece of one line, as (tracked path, kind, the exact text): a finding passes only
# when it lies inside an occurrence of that text in that file, never anywhere else in it.
ALLOWED_TEXT: set[tuple[str, str, str]] = {
    # The guard patch's comment shows the WSL-to-Windows rewrite on the placeholder drive X (DD-22 wording); the patch
    # bytes are pinned by their sha256 in tools/enfusion-mcp/README.md, so its example stays as reviewed.
    ("tools/enfusion-mcp/patches/bk-wsl-guard.patch", "machine-path",
     '+ * in WSL form ("/mnt/' + 'x/Data/...") must reach it as drive paths ("X' + ':/Data/..."), e.g. the'),
}

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


def in_placeholder_home(path: str) -> bool:
    """The synthetic config's own folder, or an *.example file: placeholders of every kind pass there."""
    rel = path.replace(os.sep, "/")
    return rel.startswith(PLACEHOLDER_HOMES) or "/" + PLACEHOLDER_HOMES[0] in rel or rel.endswith(PLACEHOLDER_SUFFIXES)


def placeholder_drive_ok(path: str, line: str, m) -> bool:
    """A placeholder drive (X, Y, Z) passes in the synthetic config's folder and in *.example files; elsewhere only as
    the bare drive or under one of SYNTHETIC_ROOTS (a real path on such a drive is still found)."""
    if in_placeholder_home(path):
        return True
    rest = line[m.end():] if m.group("name") else line[m.end() - 1:]
    if _BARE_DRIVE_TAIL.match(rest):
        return True
    if _ESCAPE_TAIL.match(rest):   # a string escape after the drive: "X:\n", "/mnt/z\n"
        return True
    rest = rest.lstrip("\\/")
    if rest.startswith("..."):   # an ellipsis: X:\..., /mnt/x/...
        return True
    low = rest.lower()   # drvfs is case-insensitive: x:/dataroot is X:\DataRoot
    return any(low.startswith(root.lower()) and _COMPONENT_END.match(rest[len(root):]) for root in SYNTHETIC_ROOTS)


def allowed_text(path: str, kind: str, line: str, m) -> bool:
    """The match lies inside an occurrence of an ALLOWED_TEXT entry for this file and kind."""
    rel = path.replace(os.sep, "/")
    for p, k, text in ALLOWED_TEXT:
        if (p, k) != (rel, kind):
            continue
        start = line.find(text)
        while start >= 0:
            if start <= m.start() and m.end() <= start + len(text):
                return True
            start = line.find(text, start + 1)
    return False


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
                    if m.group("drive") and _JS_KEY_BEFORE.search(line[: m.start()]) \
                            and _JS_REGEX_AFTER.match(line[m.end() - 1:]):
                        continue
                    if name.lower() in PLACEHOLDER_DRIVES and placeholder_drive_ok(path, line, m):
                        continue
                elif kind in ("wsl-path", "wsl-distro"):
                    if name is not None and PLACEHOLDER_DISTROS.match(name):
                        continue
                    if in_placeholder_home(path):
                        continue
                elif kind != "private-ip" and name is not None and is_placeholder(name):
                    continue
                if (path, kind, hit if name is None else name) in ALLOWLIST:
                    continue
                if allowed_text(path, kind, line, m):
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
