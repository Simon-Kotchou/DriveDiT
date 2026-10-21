#!/usr/bin/env python3
"""Static check of an Enfusion text world file (.layer / .ent) before Workbench loads it.

    python3 -I -B tools/check_layer.py addons/DataCapture/DataDrive_Layers/default.layer [more files]

Reads only. Reports, per file:
  * quoted numeric vectors such as  m_vHoodOffset "2.2 0.95 0"  (vanilla 1.8 never quotes vectors; one of them
    made 1.8 stop parsing DataDrive's layer at byte 1468 and silently drop every later entity);
  * brace balance and unterminated strings, counted outside string literals;
  * whether an inline userScript is present;
  * the entity list (class, name, inherited prefab, coords), including the children of $grp blocks;
  * every {GUID}path resource reference (check them against the game's resourceDatabase.rdb separately).
Exit status 1 if any file has a quoted numeric vector, unbalanced braces or an unterminated string, else 0.
"""

import hashlib
import re
import sys

NUM = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
QUOTED_TUPLE = re.compile(r'^\s*(?:[A-Za-z_]\w*|"[^"]+")\s+"(%s(?:\s+%s){1,3})"\s*$' % (NUM, NUM))
HEADER = re.compile(r'^(\$grp\s+)?([A-Za-z_]\w*)(?:\s+([A-Za-z_]\w*))?(?:\s*:\s*"([^"]*)")?\s*\{\s*$')
CHILD = re.compile(r'^\s*([A-Za-z_]\w*)\s*\{\s*$')
COORDS = re.compile(r'^\s*coords\s+(%s)\s+(%s)\s+(%s)\s*$' % (NUM, NUM, NUM))
RESREF = re.compile(r'\{([0-9A-Fa-f]{16})\}([^"\s\\]+)')


def scan(text):
    """Yield (line_no, depth_at_line_start, starts_in_string, line); return final (depth, in_string, errors)."""
    depth, in_str, esc = 0, False, False
    errors = []
    rows = []
    for no, line in enumerate(text.split("\n"), 1):
        rows.append((no, depth, in_str, line))
        for ch in line:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    errors.append("line %d: closing brace without an opening one" % no)
                    depth = 0
        if in_str:
            errors.append("line %d: string literal not closed on its line" % no)
            in_str, esc = False, False
    if depth:
        errors.append("end of file: %d unclosed brace(s)" % depth)
    return rows, errors


def check(path):
    data = open(path, "rb").read()
    text = data.decode("utf-8", "replace")
    print("== %s  (%d bytes, sha256 %s%s)" % (path, len(data), hashlib.sha256(data).hexdigest(),
                                             ", CRLF" if b"\r\n" in data else ""))
    rows, errors = scan(text)
    quoted = []
    entities = []
    current = None
    grp = None
    user_script = None
    for no, depth, in_str, line in rows:
        if in_str:
            continue
        m = QUOTED_TUPLE.match(line)
        if m:
            quoted.append("line %d: %s" % (no, line.strip()))
        if depth == 0:
            h = HEADER.match(line)
            if h:
                current = {"line": no, "cls": h.group(2), "name": h.group(3) or "-", "prefab": h.group(4) or "-",
                           "coords": "-"}
                grp = current if h.group(1) else None
                if not grp:
                    entities.append(current)
        elif depth == 1 and grp is not None:
            c = CHILD.match(line)
            if c:
                current = {"line": no, "cls": grp["cls"], "name": c.group(1), "prefab": grp["prefab"],
                           "coords": "-"}
                entities.append(current)
        if line.startswith(" userScript ") and depth == 1:
            user_script = no
        m = COORDS.match(line)
        if m and current is not None and depth == (2 if grp is not None else 1):
            current["coords"] = " ".join(m.groups())
    refs = {}
    for m in RESREF.finditer(text):
        key = "{%s}%s" % (m.group(1).upper(), m.group(2))
        refs[key] = refs.get(key, 0) + 1

    print("braces/strings: %s" % ("OK" if not errors else "; ".join(errors)))
    print("quoted numeric vectors: %d" % len(quoted))
    for q in quoted:
        print("  " + q)
    print("userScript: %s" % ("present, line %d" % user_script if user_script else "absent"))
    print("entities: %d" % len(entities))
    for e in entities:
        print("  L%-4d %-20s %-22s %-70s %s" % (e["line"], e["cls"], e["name"], e["prefab"], e["coords"]))
    print("resource refs: %d unique" % len(refs))
    for key in sorted(refs, key=lambda k: k.split("}", 1)[1]):
        print("  %s%s" % (key, "  x%d" % refs[key] if refs[key] > 1 else ""))
    return not errors and not quoted


def main(argv):
    if not argv:
        print(__doc__.strip())
        return 2
    ok = True
    for path in argv:
        ok = check(path) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
