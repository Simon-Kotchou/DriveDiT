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
    """A new directory under $TMPDIR when that is on /dev/shm/, else under /dev/shm/ (never /tmp)."""
    base = os.path.realpath(os.environ.get("TMPDIR") or "/")
    if not (base.startswith("/dev/shm/") and os.path.isdir(base)):
        base = "/dev/shm"
    if not (os.path.isdir(base) and os.access(base, os.W_OK)):
        raise unittest.SkipTest("no usable temporary base under /dev/shm/")
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
        for bad in ("X:\\Users\\" + "Jdoe\\x", "x:/Users/" + "jdoe", "/mnt/x/Users/" + "jdoe/y", "/mnt/y/Users/" + "jdoe"):
            self.assertEqual(kinds(bad), ["win-profile"], bad)
        for ok in ("/mnt/x/Users/x/out.json", "X:\\Users\\<name>", "/mnt/x/Users/$USER/a", "X:\\Users\\Public"):
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

    def test_private_ip(self):
        def q(*octets):
            return ".".join(str(o) for o in octets)

        for bad in (q(10, 1, 2, 3), "host=" + q(172, 16, 5, 4), q(172, 31, 255, 254), q(192, 168, 0, 1) + ":5775",
                    "http://" + q(192, 168, 1, 10) + "/x", "(" + q(10, 0, 0, 1) + ")", "at " + q(10, 9, 8, 7) + ".",
                    "nameserver " + q(172, 19, 0, 1), "via " + q(192, 168, 50, 1) + " dev eth0"):
            self.assertEqual(kinds(bad), ["private-ip"], bad)
        for ok in (q(192, 0, 2, 1), q(198, 51, 100, 7), q(203, 0, 113, 9),       # RFC 5737 documentation
                   q(127, 0, 0, 1), q(0, 0, 0, 0), q(172, 15, 0, 1), q(172, 32, 0, 1), q(169, 254, 1, 1),
                   "1.9.0.37", q(10, 1, 2, 3, 4), q(999, 1, 1, 1), "v" + q(10, 1, 2, 3), q(10, 1, 2, 3) + "a",
                   'version = "' + q(10, 4, 0, 35) + '"', "pkg==" + q(10, 1, 2, 3), "x >= " + q(10, 1, 2, 3),
                   "nvidia_curand-" + q(10, 4, 0, 35) + "-py3-none-any.whl", "version: " + q(192, 168, 0, 1)):
            self.assertEqual(kinds(ok), [], ok)

    def test_private_ip_allow_entry_is_path_specific(self):
        line = '"1.9", "' + ".".join(("10", "0", "1", "2")) + '", "1.9.0.9-rc2"'
        self.assertEqual(kinds(line, "tools/pin/tests/test_pin.py"), [])
        self.assertEqual(kinds(line, "tools/pin/tests/other.py"), ["private-ip"])
        self.assertEqual(len(scan.ALLOWLIST), 1)  # every allow entry is reviewed; add one only with its reason

    def test_windows_build(self):
        win = "Win" + "dows "
        for bad in (win + "10 " + "22H2", win + "11 23H2", win + "10 (" + "19045)", win + "10 Pro 22H2",
                    win + "10, version 21H2", win + "11 build 22631", "[Version 10.0." + "19045.5131]",
                    "OS Version:   10.0." + "22631 N/A"):
            self.assertEqual(kinds(bad), ["windows-build"], bad)
        for ok in (win + "10", win + "11", win + "10 and 11", win + "2000", "needs a recent " + win + "11",
                   win + "host", "version 10.0.1", "torch 2.15.0"):
            self.assertEqual(kinds(ok), [], ok)

    def test_wsl_version(self):
        for bad in ("WSL " + "2.3.26", "WSL version: " + "2.3.26.0", "wsl v" + "2.0.9", "WSL2 " + "2.4.13",
                    "Linux 5.15.167.4" + "-microsoft-standard-WSL2"):
            self.assertEqual(kinds(bad), ["wsl-version"], bad)
        for ok in ("WSL 2", "WSL2", "WSL in NAT mode", "under WSL; use it", "wslinfo --networking-mode"):
            self.assertEqual(kinds(ok), [], ok)

    def test_machine_path(self):
        # DD-22: drive paths of a machine; the letters are assembled so this file holds none
        mnt, sep = "/mnt/", ":\\"
        for bad in (mnt + "e" + "/Data", mnt + "c", "under " + mnt + "d" + ".", "E" + sep + "Data", "d" + ":/x",
                    "`" + "C" + sep + "Program Files`", '"' + "E" + ":\\\\Data" + '"', "(" + mnt + "e" + "/x)",
                    "cd " + mnt + "e" + " && ls"):
            self.assertEqual(kinds(bad), ["machine-path"], bad)
        for ok in (mnt + "x/DataRoot", "Z" + sep + "Steam", "y" + ":/a", mnt + "<drive>/x", "https://example.com/a",
                   "%s" + sep + "%s", "%DDLC_R:/=\\%", r"\d\d" + ":\\d", mnt + "wsl/x", mnt + "eagle", "a" + mnt + "e/x",
                   "$ROOT" + mnt + "e", "<X>" + sep + "Data", "re:/x", "Q" + ":"):
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

    def test_bom_file_with_an_appended_utf8_line(self):
        # A BOM names the encoding of the start only: a UTF-8 line appended to a UTF-16 log stays contiguous in the bytes.
        for enc in ("utf-16", "utf-32"):
            data = "log start\n".encode(enc) + ("mail " + self.email + "\n").encode()
            self.assert_found(self.write("mixed_%s.log" % enc, data), "email")

    def test_wide_text_after_an_odd_length_header(self):
        for enc, width in (("utf-16-le", 2), ("utf-16-be", 2), ("utf-32-le", 4), ("utf-32-be", 4)):
            for prefix in range(1, width):
                data = bytes(range(1, prefix + 1)) + ("mail " + self.email + "\n").encode(enc)
                self.assert_found(self.write("shifted_%s_%d.bin" % (enc, prefix), data), "email")

    def test_a_bom_file_hit_is_reported_once(self):
        rc, out = scan_files(self.write("once.txt", ("mail " + self.email + "\n").encode("utf-16")))
        self.assertEqual(rc, 1, out)
        self.assertEqual(out.count(": email: "), 1, out)


if __name__ == "__main__":
    unittest.main()
