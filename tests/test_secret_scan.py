"""Unit tests for tools/ci/secret_scan.py (DD-15). Stdlib only.

Every positive sample is assembled at run time, so this file never holds a string the scan would flag.
"""

import contextlib
import importlib.util
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_PATH = os.path.join(REPO, "tools", "ci", "secret_scan.py")

_spec = importlib.util.spec_from_file_location("dd_secret_scan", SCAN_PATH)
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)

AT = "@"


def kinds(text, path="x.txt"):
    return [kind for _, kind, _ in scan.findings(path, text)]


def temp_dir():
    """A new directory under $TMPDIR when that is on /dev/shm/ or /mnt/e/, else under /dev/shm/ (never /tmp)."""
    base = os.path.realpath(os.environ.get("TMPDIR") or "/")
    if not ((base.startswith("/dev/shm/") or base.startswith("/mnt/e/")) and os.path.isdir(base)):
        base = "/dev/shm"
    if not (os.path.isdir(base) and os.access(base, os.W_OK)) or base.startswith("/mnt/c"):
        raise unittest.SkipTest("no usable temporary base under /dev/shm/ or /mnt/e/")
    return tempfile.mkdtemp(prefix="dd_secret_scan_", dir=base)


def scan_files(*paths):
    """Run the scanner's command line on `paths`: (exit status, stdout)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = scan.main(list(paths))
    return rc, out.getvalue()


class SecretScanTest(unittest.TestCase):
    def test_steam64(self):
        self.assertEqual(kinds("id=" + "7656119" + "8" * 10), ["steam64"])
        self.assertEqual(kinds("7656119" + "8" * 9), [])        # too short
        self.assertEqual(kinds("1" + "7656119" + "8" * 10), [])  # embedded in a longer number

    def test_email(self):
        self.assertEqual(kinds("mail jane.doe" + AT + "corp-mail.de"), ["email"])
        for ok in ("noreply" + AT + "anthropic.com", "123+bot" + AT + "users.noreply.github.com",
                   "bot" + AT + "users.noreply.github.com", "a" + AT + "example.com", "t" + AT + "example.invalid"):
            self.assertEqual(kinds(ok), [], ok)
        self.assertEqual(kinds("@pytest.mark.slow  y = a @ b.T"), [])

    def test_windows_profile(self):
        for bad in ("C:\\Users\\" + "Jdoe\\x", "c:/Users/" + "jdoe", "/mnt/c/Users/" + "jdoe/y", "/mnt/d/Users/" + "jdoe"):
            self.assertEqual(kinds(bad), ["win-profile"], bad)
        for ok in ("/mnt/c/Users/x/out.json", "C:\\Users\\<name>", "/mnt/c/Users/$USER/a", "C:\\Users\\Public"):
            self.assertEqual(kinds(ok), [], ok)

    def test_home_path(self):
        self.assertEqual(kinds("/home/" + "jdoe/.cache"), ["home-path"])
        for ok in ("/home/x/play_01", "/home/<name>/", "~/home/", "/homebrew/"):
            self.assertEqual(kinds(ok), [], ok)

    def test_bare_home_path(self):
        # The account name alone, with no child path, ends at a quote, space, separator or the end of the line.
        name = "jd" + "oe"
        for bad in ("/home/" + name, "HOME=/home/" + name, '"/home/' + name + '"', "'/home/" + name + "',",
                    "/home/" + name + ":/bin/bash", "dirs: [/home/" + name + "]", "(/home/" + name + ")",
                    "at /home/" + name + ".", "/home/" + "john.doe"):
            self.assertEqual(kinds(bad), ["home-path"], bad)
        for ok in ("/home/x", "/home/x.json", '"/home/<name>"', "/home/$USER", "/home/${USER}", "/home/",
                   "/homepage", "$ROOT/home/.npmrc", "a/home/" + name):
            self.assertEqual(kinds(ok), [], ok)

    def test_allowlist_is_path_specific(self):
        line = "/home/" + "svc/app"
        with mock.patch.object(scan, "ALLOWLIST", {("deploy/Dockerfile", "home-path", "svc")}):
            self.assertEqual(kinds(line, "deploy/Dockerfile"), [])
            self.assertEqual(kinds(line, "docs/x.md"), ["home-path"])
        self.assertEqual(kinds(line, "deploy/Dockerfile"), ["home-path"])

    def test_redact_never_prints_the_match(self):
        hit = "jane.doe" + AT + "corp-mail.de"
        self.assertNotIn(hit, scan.redact(hit))
        self.assertTrue(scan.redact(hit).endswith("..."))


class SecretScanFileTest(unittest.TestCase):
    """Whole files, as the CI job reads them: text that holds NUL bytes or is UTF-16/32 is scanned, not skipped."""

    def setUp(self):
        self.tmp = temp_dir()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.email = "jane.doe" + AT + "corp-mail.de"

    def write(self, name, data):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def assert_found(self, path, kind):
        rc, out = scan_files(path)
        self.assertEqual(rc, 1, out)
        self.assertIn(": %s: " % kind, out)
        self.assertNotIn(self.email, out)  # still redacted

    def test_plain_text(self):
        self.assert_found(self.write("plain.txt", ("mail " + self.email + "\n").encode()), "email")

    def test_nul_prefixed_text(self):
        self.assert_found(self.write("nul.txt", b"\0" + ("mail " + self.email + "\n").encode()), "email")

    def test_nul_inside_binary_framing(self):
        data = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR" + bytes(range(256)) + ("tEXt " + self.email).encode() + b"\0\0"
        self.assert_found(self.write("blob.png", data), "email")

    def test_utf16_with_bom(self):
        for enc in ("utf-16", "utf-16-be", "utf-32"):
            text = "mail " + self.email + "\n"
            data = text.encode(enc) if enc != "utf-16-be" else b"\xfe\xff" + text.encode(enc)
            self.assert_found(self.write("bom_%s.txt" % enc, data), "email")

    def test_utf16_without_bom(self):
        for enc in ("utf-16-le", "utf-16-be"):
            self.assert_found(self.write("nobom_%s.txt" % enc, ("mail " + self.email + "\n").encode(enc)), "email")

    def test_bare_home_path_in_a_file(self):
        rc, out = scan_files(self.write("env.txt", ("HOME=/home/" + "jd" + "oe\n").encode()))
        self.assertEqual(rc, 1, out)
        self.assertIn(": home-path: ", out)

    def test_binary_without_text_is_clean(self):
        data = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 16 + b"\0" * 64
        rc, out = scan_files(self.write("noise.bin", data))
        self.assertEqual((rc, out), (0, ""))


if __name__ == "__main__":
    unittest.main()
