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
  - DC_TextFile.Read takes FileHandle.Read of GetLength() bytes; its ReadLine fallback refuses a line of 8191 or
    more characters and takes the final LF from the file's last byte, so under each ReadLine model (no limit, cut or
    split at 8191; 1.8 documents none) it gives the file's bytes or fails. Before, it inferred the final LF from the
    length, and a one-line 8192-byte file without LF cut to 8191 read back "whole" with a wrong last byte;
  - DC_JsonScan.ReadStrings reads an 8191-character item whole and refuses an 8192-character one;
  - the two values of session data that passed through a capped call (exe.cli.dcInstance by Trim, the collision
    payload's other_prefab by Substring) reach the file whole at any length.
The behavioural tests take what the Enforce code does from its statements (fallback_model, read_strings_guard,
session_models), so removing a guard fails them, not only a statement assertion.
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
SESSION_SOURCE = os.path.join(SOURCES, "DC_Session.c")
CRASH_SOURCE = os.path.join(SOURCES, "DC_CrashWatch.c")
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


# ReadLines before the fix (402843a): the final LF inferred from the length
OLD_READ_LINES = """
class DC_TextFile
{
	protected static void ReadLines(string path, int length, out string text)
	{
		text = string.Empty;
		FileHandle file = FileIO.OpenFile(path, FileMode.READ);
		if (!file)
			return;

		string line;
		bool first = true;
		while (file.ReadLine(line) >= 0)
		{
			if (!first)
				text += "\\n";

			text += line;
			first = false;
		}

		file.Close();
		if (text.Length() == length - 1)
			text += "\\n";
	}
}
"""


def fallback_model(source):
    """What DC_TextFile.ReadLines does, read from its statements: whether it refuses a line of SUBSTRING_MAX_CHARS
    or more, and where the final LF comes from (the file's last byte, or inferred from a length one short)."""
    body = collapse(method_body(source, "DC_TextFile", "ReadLines"))
    refuses = ("if (line.Length() >= SUBSTRING_MAX_CHARS) lineAtLimit = true;" in body
               and "if (lineAtLimit) return;" in body)
    if 'file.Seek(length - 1); file.Read(lastByte, 1);' in body and 'if (lastByte == "\\n") joined += "\\n";' in body:
        final_lf = "last_byte"
    elif 'if (text.Length() == length - 1) text += "\\n";' in body:
        final_lf = "inferred"
    else:
        raise AssertionError("unknown ReadLines: %s" % body)
    return {"refuses_long_lines": refuses, "final_lf": final_lf}


def read_lines_model(content, line_model):
    """The lines ReadLine returns: as in the file (no LF; no empty line after a final LF), cut at SUBSTRING_MAX, or
    split into SUBSTRING_MAX pieces."""
    lines = content.split("\n")
    if content.endswith("\n"):
        lines = lines[:-1]
    if line_model == "cut":
        return [line[:SUBSTRING_MAX] for line in lines]
    if line_model == "split":
        out = []
        for line in lines:
            pieces = [line[i:i + SUBSTRING_MAX] for i in range(0, len(line), SUBSTRING_MAX)] or [""]
            out += pieces
        return out
    return lines


def read_back(content, read_limit, line_model, model):
    """DC_TextFile.Read with a FileHandle.Read that returns read_limit bytes and ReadLines as model says: (ok, text)."""
    length = len(content)
    text = content[:read_limit]
    if len(text) != length:
        lines = read_lines_model(content, line_model)
        joined = "\n".join(lines)
        if model["refuses_long_lines"] and any(len(line) >= SUBSTRING_MAX for line in lines):
            joined = ""
        elif model["final_lf"] == "last_byte":
            if length > 0 and content[-1] == "\n":
                joined += "\n"
        elif len(joined) == length - 1:
            joined += "\n"
        text = joined
    return len(text) == length, text


def read_strings_guard(source):
    """(limit, refuses) of DC_JsonScan.ReadStrings' length guard before its Substring."""
    body = collapse(method_body(source, "DC_JsonScan", "ReadStrings"))
    m = re.search(r"if \(close - index > DC_TextFile\.SUBSTRING_MAX_CHARS\) \{([^}]*)\}", body)
    if m is None:
        return (None, False)
    limit = int(re.search(r"static const int SUBSTRING_MAX_CHARS = (\d+);", source).group(1))
    assert body.index(m.group(0)) < body.index("item = text.Substring(index, close - index);")
    return (limit, "return false;" in m.group(1))


def read_strings(text, key, guard):
    """DC_JsonScan.ReadStrings on a flat array of plain strings: (ok, values); an unguarded item longer than the
    limit comes back cut by Substring."""
    limit, refuses = guard
    rest = text.split('"%s":' % key, 1)[1].strip()
    assert rest.startswith("[")
    values, index = [], 1
    while True:
        index = rest.index('"', index) + 1
        close = rest.index('"', index)
        if limit is not None and close - index > limit and refuses:
            return False, values
        values.append(EnforceString.substring(rest, index, close - index))
        index = close + 1
        tail = rest[index:].lstrip()
        if tail.startswith("]"):
            return True, values
        index = rest.index(",", index) + 1


def session_models(session, crash):
    """How exe.cli.dcInstance and the other_prefab payload are copied, read from the statements."""
    record = collapse(session)
    prefab = collapse(method_body(crash, "DC_CrashWatch", "PrefabPath"))
    bounded = ("if (cliInstance.Length() <= CAPPED_STRING_MAX_CHARS) cliInstance = cliInstance.Trim(); "
               'json.AddString("dcInstance", cliInstance);') in record
    if not bounded:
        assert 'json.AddString("dcInstance", cliInstance.Trim());' in record
    pieces = ("int take = Math.MinInt(path.Length() - at, DC_Session.CAPPED_STRING_MAX_CHARS); "
              "rest += path.Substring(at, take); at += take;") in prefab and "path = rest;" in prefab
    if not pieces:
        assert "path = path.Substring(close + 1, path.Length() - close - 1);" in prefab
    return {"instance_trim_bounded": bounded, "prefab_pieces": pieces}


def enforce_trim(text):
    """string.Trim: at most SUBSTRING_MAX characters (string.c:106)."""
    return text.strip()[:SUBSTRING_MAX]


def record_instance(raw, model):
    if model["instance_trim_bounded"]:
        return enforce_trim(raw) if len(raw) <= SUBSTRING_MAX else raw
    return enforce_trim(raw)


def prefab_rest(path, model):
    close = path.index("}")
    if model["prefab_pieces"]:
        rest, at = "", close + 1
        while at < len(path):
            take = min(len(path) - at, SUBSTRING_MAX)
            rest += EnforceString.substring(path, at, take)
            at += take
        return rest
    return EnforceString.substring(path, close + 1, len(path) - close - 1)


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
        ReadLine fallback only after a short read, and the length check after it. ReadLines (fallback_model, from the
        source) refuses a line of SUBSTRING_MAX_CHARS or more and takes the final LF from the file's last byte, so
        under each ReadLine model (no limit, lines cut at 8191, lines split into 8191-character pieces; 1.8 documents
        no limit) a fallback read either gives the file's bytes or fails. The fallback before this fix inferred the
        final LF from the length: a one-line 8192-byte file without LF, cut to 8191, read back "whole" with its last
        byte replaced by an LF (Codex review of #17)."""
        flat = collapse(method_body(self.llc, "DC_TextFile", "Read"))
        for statement in ("int length = file.GetLength(); file.Read(text, length); file.Close();",
                          "if (text.Length() != length) ReadLines(path, length, text);",
                          "if (text.Length() != length) { failure ="):
            self.assertIn(statement, flat)
        self.assertLess(flat.index("ReadLines(path, length, text);"), flat.index("if (text.Length() != length) {"))
        model = fallback_model(self.llc)
        self.assertEqual(model, {"refuses_long_lines": True, "final_lf": "last_byte"})
        cases = [self.raw.decode("ascii"),                          # the route: one 39,692-character line and an LF
                 "{\"x\":1}" + " " * 8185,                            # 8192 bytes, one line, no LF (Codex's case)
                 "a" * 9000 + "\n",                                    # one line split at 8191 joins to its length
                 "a" * 8190 + "\n" + "b" * 10,                        # lines below the limit, no final LF
                 "short\nlines\nonly\n", "short\nlines\nonly", "\n\n", "x"]
        for content in cases:
            for read_limit in (len(content), 0, 4096):
                for line_model in ("none", "cut", "split"):
                    ok, text = read_back(content, read_limit, line_model, model)
                    self.assertTrue(not ok or text == content, (len(content), read_limit, line_model))
        # the reads that succeed: the whole-file read always; the fallback for files of short lines
        self.assertEqual(read_back(cases[0], len(cases[0]), "cut", model), (True, cases[0]))
        self.assertFalse(read_back(cases[0], 4096, "none", model)[0])      # a route never via the fallback
        self.assertEqual(read_back(cases[4], 3, "cut", model), (True, cases[4]))
        self.assertEqual(read_back(cases[5], 3, "split", model), (True, cases[5]))
        # the fallback before the fix (402843a) under the same models
        old = {"refuses_long_lines": False, "final_lf": "inferred"}
        self.assertEqual(fallback_model(OLD_READ_LINES), old)
        ok, text = read_back(cases[1], 0, "cut", old)
        self.assertTrue(ok)
        self.assertNotEqual(text, cases[1])
        self.assertEqual(text, cases[1][:SUBSTRING_MAX] + "\n")
        ok, text = read_back(cases[2], 0, "split", old)
        self.assertTrue(ok and text != cases[2])

    def test_read_strings_refuses_an_item_above_the_limit(self):
        """DC_JsonScan.ReadStrings (read_strings, its guard taken from the source): an item of 8191 characters is
        read whole, one of 8192 is refused (Substring would return its first 8191 characters)."""
        guard = read_strings_guard(self.llc)
        self.assertEqual(guard, (SUBSTRING_MAX, True))
        for length, want in ((SUBSTRING_MAX - 1, True), (SUBSTRING_MAX, True), (SUBSTRING_MAX + 1, False),
                             (3 * SUBSTRING_MAX, False)):
            item = "s" * length
            ok, values = read_strings('{"ids": ["a", "%s", "b"]}' % item, "ids", guard)
            self.assertEqual(ok, want, length)
            if ok:
                self.assertEqual(values, ["a", item, "b"])
        # without the refusal the 8192-character item would come back cut
        ok, values = read_strings('{"ids": ["%s"]}' % ("s" * (SUBSTRING_MAX + 1)), "ids", (SUBSTRING_MAX, False))
        self.assertEqual((ok, len(values[0])), (True, SUBSTRING_MAX))

    def test_session_data_never_takes_a_capped_copy(self):
        """Two values written into session data passed through a capped call: session.json exe.cli.dcInstance (Trim
        of the raw CLI value) and the collision payload's other_prefab (Substring of the prefab path). Now the CLI
        value is trimmed only when Trim can return it whole, and the prefab path is copied in pieces."""
        session, crash = read(SESSION_SOURCE), read(CRASH_SOURCE)
        self.assertIn("static const int CAPPED_STRING_MAX_CHARS = %d;" % SUBSTRING_MAX, session)
        model = session_models(session, crash)
        self.assertEqual(model, {"instance_trim_bounded": True, "prefab_pieces": True})
        for raw in ("01", " 01 ", "x" * SUBSTRING_MAX + " ", " " + "x" * (SUBSTRING_MAX + 5) + " "):
            recorded = record_instance(raw, model)
            self.assertTrue(recorded in (raw, raw.strip()), len(raw))
            if len(raw) <= SUBSTRING_MAX:
                self.assertEqual(recorded, raw.strip())
        for rest in (20, SUBSTRING_MAX, SUBSTRING_MAX + 1, 3 * SUBSTRING_MAX + 7):
            path = "{0123456789ABCDEF}" + ("p" * rest)
            self.assertEqual(prefab_rest(path, model), "p" * rest, rest)
        # the code before: both cut silently
        old = {"instance_trim_bounded": False, "prefab_pieces": False}
        self.assertEqual(len(record_instance("x" * (SUBSTRING_MAX + 5), old)), SUBSTRING_MAX)
        self.assertEqual(len(prefab_rest("{G}" + "p" * (SUBSTRING_MAX + 9), old)), SUBSTRING_MAX)

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
