"""Unit tests for tools/ci/secret_scan.py (DD-15). Stdlib only.

Every positive sample is assembled at run time, so this file never holds a string the scan would flag.
"""

import importlib.util
import os
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_PATH = os.path.join(REPO, "tools", "ci", "secret_scan.py")

_spec = importlib.util.spec_from_file_location("dd_secret_scan", SCAN_PATH)
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)

AT = "@"


def kinds(text, path="x.txt"):
    return [kind for _, kind, _ in scan.findings(path, text)]


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

    def test_allowlist_is_path_specific(self):
        line = "/home/" + "drivedit/app"
        self.assertEqual(kinds(line, "Dockerfile"), [])
        self.assertEqual(kinds(line, "docs/x.md"), ["home-path"])

    def test_redact_never_prints_the_match(self):
        hit = "jane.doe" + AT + "corp-mail.de"
        self.assertNotIn(hit, scan.redact(hit))
        self.assertTrue(scan.redact(hit).endswith("..."))


if __name__ == "__main__":
    unittest.main()
