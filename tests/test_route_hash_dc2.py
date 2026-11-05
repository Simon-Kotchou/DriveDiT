#!/usr/bin/env python3
"""episodes.route_sha256 of long one-line routes (DD-7): DC_Route.Load hashes the exact bytes the contract defines.

    TMPDIR=/dev/shm/<x> python3 -I -B -m pytest --noconftest -p no:cacheprovider -q tests/test_route_hash_dc2.py

dc-2.0 9.6 defines episodes.route_sha256 as the sha256 of the route's canonical JSON (drivedit_nav.Route.sha256).
make_t0_loop.py writes that text and one LF, so DC_Route.Load hashes the file without its final LF. The canonical
form has no whitespace: a route is one line (arland_t0_loop: 39,692 characters). The first production session
(captures/arland/2026-11-05/T0L) logged 6370fc78f426..., the sha256 of the route's first 8191 bytes, and the
validator's C17 failed it against the route file's 77838c1346f2... The cause was not the file read: FileHandle.Read
returned the whole file (the same read gave the source and card digests session.json stamps correctly, and the
parser drove all 1288 points). It was the Substring copy that dropped the final LF before hashing. string.Substring,
string.Trim and string.Format return at most 8191 characters (1.8.0.13 Core/generated/Types/string.c:92, :106,
:119) and cut a longer result without an error.

These tests read the Enforce sources (src/enfusion/datacapture/DC_ExpertHL.c, DC_LLCv1.c) and check:
  - the route-hash statements of DC_Route.Load, run by a small evaluator with the Enforce string semantics (the
    8191-character cap of Substring), give drivedit_nav's Route.sha256 for one-line routes of any length, and for
    the committed route the value the fixed engine logs. The same evaluator on the statements before the fix
    (402843a) reproduces the session's 6370fc78f426... exactly, so the model of the cause is the cause;
  - DC_Sha256.HashText is HashPrefix over the whole text, and HashPrefix reads its prefix in place (ToAscii, the
    bit length of the clamped count), never through Substring;
  - no reader of a file's text (DC_TextFile, DC_JsonScan, DC_VehicleCard, DC_Route) passes that text through
    Substring, Trim or string.Format; the two token cuts left are bounded (IsNull: 4 characters; ReadStrings:
    refused above DC_TextFile.SUBSTRING_MAX_CHARS);
  - DC_TextFile.Read takes FileHandle.Read of GetLength() bytes, and its ReadLine fallback joined by LF can only
    succeed with exactly GetLength() characters: a line cut by any ReadLine limit fails the read, never a digest.
"""

import hashlib
import json
import os
import re
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import drivedit_nav  # noqa: E402

SOURCES = os.path.join(REPO, "src", "enfusion", "datacapture")
EXPERT_SOURCE = os.path.join(SOURCES, "DC_ExpertHL.c")
LLC_SOURCE = os.path.join(SOURCES, "DC_LLCv1.c")
ROUTE_FILE = os.path.join(REPO, "addons", "DataCapture", "Configs", "routes", "arland_t0_loop.json")
SUBSTRING_MAX = 8191        # string.c:92 (Substring), :106 (Trim), :119 (Format) of 1.8.0.13
# episodes.route_sha256 of every episode of the first production session (arland_T0L_20261105T072648Z_wb_i00_66b8)
SESSION_ROUTE_SHA = "6370fc78f4266db37367ce7a3110a2e2b7515aeb90f695382092948d946f0c9d"
# Route.sha256 of the committed arland_t0_loop.json (the value C17 compares with)
CONTRACT_ROUTE_SHA = "77838c1346f2c7a6b18be78570d9f8cb560ee10a0e69718b28eb0f028cc0f66f"
# The hash statements of DC_Route.Load before the fix (402843a), as the engine ran them in that session
OLD_HASH_STATEMENTS = """
		string hashed = text;
		if (text.EndsWith("\\n"))
			hashed = text.Substring(0, text.Length() - 1);

		m_sSha256 = DC_Sha256.HashText(hashed);
"""


def read(path):
    with open(path, "r", encoding="ascii", newline="") as f:
        return f.read()


def code_only(source):
    """The source without its // comments (a statement test must not match a comment)."""
    return "\n".join(line.split("//", 1)[0] for line in source.split("\n"))


def class_text(source, cls):
    """The text of one Enforce class, from its declaration to the next top-level class."""
    found = re.search(r"\nclass %s(?:\s*:\s*\w+)?\s*\n\{" % cls, source)
    rest = source[found.start() + 1:]
    following = re.search(r"\n(?://[^\n]*\n|\[[^\n]*\]\n)*class \w+", rest[1:])
    return rest if following is None else rest[:following.start() + 1]


def method_body(source, cls, method):
    """One method of an Enforce class, comments dropped (lines kept)."""
    body = class_text(source, cls)
    head = re.search(r"\n\t(?:(?:override|protected|static) )*[\w<>]+ %s\(" % method, body)
    return code_only(body[head.start():].split("\n\t}\n", 1)[0])


def collapse(text):
    return " ".join(text.split())


class EnforceString:
    """The Enforce string calls of the hash statements, with the 1.8.0.13 semantics: Substring returns at most
    SUBSTRING_MAX characters (string.c:92); Length, EndsWith and ToAscii see the whole string."""

    @staticmethod
    def substring(text, start, length):
        return text[start:start + length][:SUBSTRING_MAX]


def sha256_text(text):
    """DC_Sha256.HashText: FIPS 180-4 over 7-bit ASCII (test_fit_steer_kappa.py checks the Enforce code against
    hashlib on emulated Enforce ints)."""
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def hash_prefix(text, count):
    """DC_Sha256.HashPrefix: the first count characters (clamped to [0, len]), read in place."""
    return sha256_text(text[:min(max(count, 0), len(text))])


def run_hash_statements(statements, text):
    """Runs the route-hash statements of DC_Route.Load on text and returns m_sSha256. Only the statement shapes these
    statements use are known (declarations of string / int from text or text.Length(), an if on text.EndsWith, a
    Substring assignment, a decrement, the HashText / HashPrefix assignment); anything else raises, so a change of
    the Enforce statements must come with this evaluator."""
    env = {"text": text}

    def value(expr):
        expr = expr.strip()
        m = re.fullmatch(r"(\w+)\.Length\(\)(?:\s*-\s*(\d+))?", expr)
        if m:
            return len(env[m.group(1)]) - int(m.group(2) or 0)
        m = re.fullmatch(r"(\w+)\.Substring\((.+?),\s*(.+)\)", expr)
        if m:
            return EnforceString.substring(env[m.group(1)], value(m.group(2)), value(m.group(3)))
        m = re.fullmatch(r"DC_Sha256\.HashText\((\w+)\)", expr)
        if m:
            return sha256_text(env[m.group(1)])
        m = re.fullmatch(r"DC_Sha256\.HashPrefix\((\w+),\s*(.+)\)", expr)
        if m:
            return hash_prefix(env[m.group(1)], value(m.group(2)))
        if re.fullmatch(r"\d+", expr):
            return int(expr)
        if re.fullmatch(r"\w+", expr):
            return env[expr]
        raise AssertionError("unknown expression %r" % expr)

    def execute(statement):
        m = re.fullmatch(r"(?:string |int )?(\w+)\s*=\s*(.+);", statement)
        if m:
            env[m.group(1)] = value(m.group(2))
            return
        m = re.fullmatch(r"(\w+)--;", statement)
        if m:
            env[m.group(1)] -= 1
            return
        raise AssertionError("unknown statement %r" % statement)

    lines = [line.strip() for line in code_only(statements).split("\n") if line.strip()]
    i = 0
    while i < len(lines):
        m = re.fullmatch(r'if \((\w+)\.EndsWith\("\\n"\)\)', lines[i])
        if m:
            if env[m.group(1)].endswith("\n"):
                execute(lines[i + 1])
            i += 2
            continue
        execute(lines[i])
        i += 1
    return env["m_sSha256"]


def current_hash_statements():
    """The statements of DC_Route.Load between `m_sPath = path;` and the JSON parse (the route hash)."""
    body = method_body(read(EXPERT_SOURCE), "DC_Route", "Load")
    return body.split("m_sPath = path;", 1)[1].split("JsonLoadContext context", 1)[0]


def one_line_route_text(length):
    """A canonical one-line JSON text (drivedit_nav.canonical_json) of exactly length characters."""
    base = drivedit_nav.canonical_json({"pad": "", "schema": "drivedit.route/1"})
    text = drivedit_nav.canonical_json({"pad": "x" * (length - len(base)), "schema": "drivedit.route/1"})
    assert len(text) == length and "\n" not in text
    return text


class RouteHashTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expert = read(EXPERT_SOURCE)
        cls.llc = read(LLC_SOURCE)
        cls.statements = current_hash_statements()
        with open(ROUTE_FILE, "rb") as f:
            cls.raw = f.read()

    def test_the_evaluator_knows_every_statement(self):
        """The current hash statements are exactly the shapes the evaluator runs (no Substring left)."""
        self.assertEqual(collapse(self.statements), collapse(
            'int hashedLength = text.Length(); if (text.EndsWith("\\n")) hashedLength--; '
            'm_sSha256 = DC_Sha256.HashPrefix(text, hashedLength);'))
        self.assertNotIn("Substring", self.statements)

    def test_long_one_line_routes_hash_whole(self):
        """Route.sha256 (the canonical text) for one-line files of every length around and above the cap, with
        and without the final LF; the statements before the fix diverge exactly above 8191 characters."""
        for length in (100, SUBSTRING_MAX - 1, SUBSTRING_MAX, SUBSTRING_MAX + 1, SUBSTRING_MAX + 2,
                       2 * SUBSTRING_MAX + 1, 39692, 100000):
            canonical = one_line_route_text(length)
            want = sha256_text(canonical)
            for file_text in (canonical + "\n", canonical):
                self.assertEqual(run_hash_statements(self.statements, file_text), want, (length, len(file_text)))
            old = run_hash_statements(OLD_HASH_STATEMENTS, canonical + "\n")
            if length < SUBSTRING_MAX:
                self.assertEqual(old, want, length)
            else:
                self.assertEqual(old, sha256_text(canonical[:SUBSTRING_MAX]), length)
                if length > SUBSTRING_MAX:
                    self.assertNotEqual(old, want, length)

    def test_committed_route_logs_the_contract_hash(self):
        """arland_t0_loop.json: one 39,692-character line and an LF. The fixed statements log Route.sha256
        (77838c1346f2..., what C17 compares with); the statements before the fix give the first session's
        6370fc78f426..., the sha256 of the first 8191 bytes."""
        text = self.raw.decode("ascii")
        self.assertEqual(text.count("\n"), 1)
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(len(text) - 1, 39692)
        route = drivedit_nav.load_route(ROUTE_FILE)
        self.assertEqual(route.sha256, CONTRACT_ROUTE_SHA)
        self.assertEqual(run_hash_statements(self.statements, text), route.sha256)
        self.assertEqual(run_hash_statements(OLD_HASH_STATEMENTS, text), SESSION_ROUTE_SHA)
        self.assertEqual(hashlib.sha256(self.raw[:SUBSTRING_MAX]).hexdigest(), SESSION_ROUTE_SHA)

    def test_hash_text_is_hash_prefix_in_place(self):
        """HashText hashes the whole text through HashPrefix; HashPrefix clamps the count, reads each character by
        ToAscii and pads with the bit length of the clamped count; no Substring in either."""
        hash_text = collapse(method_body(self.llc, "DC_Sha256", "HashText"))
        self.assertIn("return HashPrefix(text, text.Length());", hash_text)
        prefix = method_body(self.llc, "DC_Sha256", "HashPrefix")
        flat = collapse(prefix)
        self.assertIn("int length = Math.ClampInt(count, 0, text.Length());", flat)
        self.assertIn("for (int i = 0; i < length; i++) { int code = text.ToAscii(i);", flat)
        self.assertIn("int bitsHigh = Shr(length, 29); int bitsLow = length << 3;", flat)
        self.assertNotIn("Substring", prefix)
        self.assertNotIn("Substring", hash_text)
        self.assertEqual(hash_prefix("abcdef", 3), sha256_text("abc"))
        self.assertEqual(hash_prefix("abc", 10), sha256_text("abc"))
        self.assertEqual(hash_prefix("abc", -1), sha256_text(""))

    def test_no_capped_string_call_on_file_text(self):
        """The readers of a file's text never pass it through Substring, Trim or string.Format: in DC_TextFile,
        DC_JsonScan, DC_VehicleCard.Load and DC_Route.Load the only Substring calls on the text are IsNull's
        4-character token and ReadStrings' item, which is refused above SUBSTRING_MAX_CHARS first."""
        readers = [code_only(class_text(self.llc, cls)) for cls in ("DC_TextFile", "DC_JsonScan")]
        readers.append(method_body(self.llc, "DC_VehicleCard", "Load"))
        readers.append(method_body(self.expert, "DC_Route", "Load"))
        text_calls = []
        for code in readers:
            text_calls += re.findall(r"\btext\.(?:Substring|Trim|TrimInPlace)\([^;]*", code)
            self.assertIsNone(re.search(r"string\.Format\([^;]*\btext\b(?!\.Length\(\))", code))
        self.assertEqual(sorted(text_calls), sorted(["text.Substring(index, 4) == \"null\"",
                                                      "text.Substring(index, close - index)"]))
        self.assertIn("static const int SUBSTRING_MAX_CHARS = %d;" % SUBSTRING_MAX, self.llc)
        strings = collapse(method_body(self.llc, "DC_JsonScan", "ReadStrings"))
        guard = "if (close - index > DC_TextFile.SUBSTRING_MAX_CHARS) {"
        self.assertIn(guard, strings)
        self.assertLess(strings.index(guard), strings.index("item = text.Substring(index, close - index);"))

    def test_read_fails_instead_of_cutting(self):
        """DC_TextFile.Read: FileHandle.Read of GetLength() bytes (not limited for a string, FileHandle.c), the
        ReadLine fallback only after a short read, and the length check after it: a fallback whose lines a ReadLine
        limit cut (8191 is assumed here; 1.8 documents none) gives fewer characters and fails the read."""
        flat = collapse(method_body(self.llc, "DC_TextFile", "Read"))
        for statement in ("int length = file.GetLength(); file.Read(text, length); file.Close();",
                          "if (text.Length() != length) ReadLines(path, length, text);",
                          "if (text.Length() != length) { failure ="):
            self.assertIn(statement, flat)
        self.assertLess(flat.index("ReadLines(path, length, text);"), flat.index("if (text.Length() != length) {"))

        def read_back(content, read_limit, line_limit):
            """Python mirror of DC_TextFile.Read and ReadLines: (ok, text)."""
            length = len(content)
            text = content[:read_limit]
            if len(text) != length:
                lines = content.split("\n")
                if content.endswith("\n"):
                    lines = lines[:-1]
                text = "\n".join(line[:line_limit] for line in lines)
                if len(text) == length - 1:
                    text += "\n"
            return len(text) == length, text

        content = self.raw.decode("ascii")
        self.assertEqual(read_back(content, len(content), SUBSTRING_MAX), (True, content))
        self.assertEqual(read_back(content, 4096, 10 ** 6), (True, content))
        ok, text = read_back(content, 4096, SUBSTRING_MAX)
        self.assertFalse(ok)
        self.assertEqual(len(text), SUBSTRING_MAX)

    def test_the_route_parser_reads_the_whole_text(self):
        """Parsing never used the cut copy: JsonLoadContext.LoadFromString, ReadPolyline and ReadCorridor take
        text, so the first session drove the whole route; only the hash had the copy."""
        flat = collapse(method_body(self.expert, "DC_Route", "Load"))
        for statement in ("context.LoadFromString(text)", "ReadPolyline(text, failure)", "ReadCorridor(text, failure)"):
            self.assertIn(statement, flat)
        obj = json.loads(self.raw.decode("ascii"))
        self.assertEqual(obj["source"]["n_points"], len(obj["polyline"]))
        self.assertGreater(len(drivedit_nav.canonical_json(obj)), SUBSTRING_MAX)


if __name__ == "__main__":
    unittest.main()
