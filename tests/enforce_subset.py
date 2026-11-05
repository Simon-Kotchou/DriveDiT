"""A translator of a small subset of Enforce Script into Python, so tests can run capture code itself, not a mirror.

    classes = translate(source, ["DC_NoiseRng", "DC_Noise"], externals={"DC_NoiseConfig": ConfigStub})

It covers what DC_Noise.c's DC_NoiseRng and DC_Noise use, and refuses anything else (TranslateError), so a change of
those classes that leaves the subset fails the tests that run them instead of being skipped silently:
  - members: [protected|private|static|const|ref] <type> name [= literal | {list} | new Class()];
  - methods: [protected|static|override] <type> Name(params) { ... }; a method with an out parameter is not translated
    (calling it raises);
  - statements: blocks, declarations, = += -= *= /= ++ --, if / else if / else, while, for (init; cond; step),
    foreach (type name : array), return, break, continue, expression statements (calls);
  - expressions with C precedence (fully parenthesised in the output): ! - unary, * / %, + -, << >>, < <= > >=,
    == !=, &, ^, |, &&, ||; calls, member access, indexing, new Class(), string literals, int (decimal or hex) and
    float literals, true / false / null, string.Empty, string.Format, Math.*, Print, LogLevel.*, x.ToString(...).
Enforce semantics kept: int is 32-bit two's complement (every store to an int is wrapped, << wraps, >> is arithmetic),
int / int truncates toward zero, a float stored in an int truncates toward zero, Math.Round rounds half away from
zero, arrays are array<T> (Count, Insert, Clear, Find, Copy, IsEmpty, Remove, indexing). Floats are Python doubles
(Enforce's are 32-bit), so values agree to about 1e-7 relative, not bit for bit.
"""

import math
import re


class TranslateError(ValueError):
    pass


def i32(v):
    v = int(v) & 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def to_int(v):
    """A value stored in an Enforce int: floats truncate toward zero, then 32-bit wrap."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            raise ValueError("non-finite float stored in an int")
        v = math.trunc(v)
    return i32(v)


def to_float(v):
    return float(v)


def div(a, b):
    """Enforce /: int / int truncates toward zero, anything else is a float division."""
    if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool) and not isinstance(b, bool):
        if b == 0:
            raise ZeroDivisionError("int division by zero")
        q = abs(a) // abs(b)
        return i32(q if (a >= 0) == (b >= 0) else -q)
    return float(a) / float(b)


def mod(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return i32(int(math.fmod(a, b)))
    return math.fmod(a, b)


def shl(a, b):
    return i32(int(a) << int(b))


class EArray(list):
    """array<T>."""

    def Count(self):
        return len(self)

    def Insert(self, v):
        self.append(v)
        return len(self) - 1

    def Clear(self):
        del self[:]

    def Find(self, v):
        try:
            return self.index(v)
        except ValueError:
            return -1

    def Copy(self, other):
        self[:] = list(other)
        return len(self)

    def IsEmpty(self):
        return len(self) == 0

    def Remove(self, i):
        del self[i]


class _Math:
    PI = math.pi
    PI2 = 2.0 * math.pi
    PI_HALF = math.pi / 2.0
    E = math.e
    DEG2RAD = math.pi / 180.0
    RAD2DEG = 180.0 / math.pi

    @staticmethod
    def Round(f):
        return float(math.floor(abs(f) + 0.5) * (1 if f >= 0 else -1))

    @staticmethod
    def Floor(f):
        return float(math.floor(f))

    @staticmethod
    def Min(a, b):
        return float(min(a, b))

    @staticmethod
    def Max(a, b):
        return float(max(a, b))

    @staticmethod
    def MinInt(a, b):
        return to_int(min(a, b))

    @staticmethod
    def MaxInt(a, b):
        return to_int(max(a, b))

    @staticmethod
    def Clamp(v, lo, hi):
        return float(min(max(v, lo), hi))

    @staticmethod
    def ClampInt(v, lo, hi):
        return to_int(min(max(v, lo), hi))

    @staticmethod
    def AbsFloat(f):
        return float(abs(f))

    @staticmethod
    def AbsInt(i):
        return to_int(abs(i))

    @staticmethod
    def Sqrt(f):
        return math.sqrt(f)

    @staticmethod
    def Log(f):
        return math.log(f)

    @staticmethod
    def Sin(f):
        return math.sin(f)

    @staticmethod
    def Cos(f):
        return math.cos(f)

    @staticmethod
    def Pow(v, p):
        return math.pow(v, p)


def to_string(v, *args):
    """x.ToString(...) of an int, float or bool."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    decimals = args[1] if len(args) > 1 else 6
    return "%.*f" % (decimals, v)


def format_string(fmt, *args):
    out = fmt
    for i in range(len(args), 0, -1):
        out = out.replace("%%%d" % i, to_string(args[i - 1]) if not isinstance(args[i - 1], str) else args[i - 1])
    return out


RUNTIME = {"i32": i32, "_int": to_int, "_float": to_float, "_div": div, "_mod": mod, "_shl": shl, "EArray": EArray,
           "Math": _Math, "_tostr": to_string, "_format": format_string, "math": math}

TOKEN_RE = re.compile(r'\s*(?:(?P<str>"(?:\\.|[^"\\])*")|(?P<num>0x[0-9A-Fa-f]+|\d+\.\d*|\.\d+|\d+)|'
                      r'(?P<id>[A-Za-z_]\w*)|(?P<op><<=|>>=|<<|>>|<=|>=|==|!=|&&|\|\||\+\+|--|\+=|-=|\*=|/=|'
                      r'[-+*/%<>=!&|^~(){}\[\];,.:?]))')
PRIMITIVES = {"int": "int", "float": "float", "bool": "bool", "string": "string", "void": "void"}
MODIFIERS = {"protected", "private", "static", "const", "ref", "override", "notnull", "autoptr", "owned"}
BINARY = [("||",), ("&&",), ("|",), ("^",), ("&",), ("==", "!="), ("<", "<=", ">", ">="), ("<<", ">>"), ("+", "-"),
          ("*", "/", "%")]


def strip_comments(text):
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif text[i] == '"':
            m = re.compile(r'"(?:\\.|[^"\\])*"').match(text, i)
            out.append(m.group(0))
            i = m.end()
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def tokenize(text):
    tokens, pos = [], 0
    text = text.rstrip()
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            raise TranslateError("cannot tokenize at %r" % text[pos:pos + 30])
        kind = m.lastgroup
        tokens.append((kind, m.group(kind)))
        pos = m.end()
    return tokens


class Scope:
    def __init__(self, fields, methods, classes, statics):
        self.fields, self.methods, self.classes, self.statics = fields, methods, classes, statics
        self.locals = [{}]

    def push(self):
        self.locals.append({})

    def pop(self):
        self.locals.pop()

    def declare(self, name, typ):
        self.locals[-1][name] = typ

    def local_type(self, name):
        for frame in reversed(self.locals):
            if name in frame:
                return frame[name]
        return None


class MethodTranslator:
    def __init__(self, tokens, scope, field_types, return_type):
        self.t, self.i, self.scope, self.field_types, self.ret = tokens, 0, scope, field_types, return_type
        self.lines = []

    # ---------------------------------------------------------------- tokens
    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else (None, None)

    def take(self, value=None):
        tok = self.peek()
        if value is not None and tok[1] != value:
            raise TranslateError("expected %r, got %r near %s" % (value, tok[1], self.context()))
        self.i += 1
        return tok

    def context(self):
        return " ".join(v for _, v in self.t[max(0, self.i - 6):self.i + 6])

    # ---------------------------------------------------------------- types
    def at_type(self):
        """A declaration starts here: <type> <name> with type a primitive, a class or array<...>."""
        k, v = self.peek()
        j = 0
        while self.peek(j)[1] in MODIFIERS:
            j += 1
        k, v = self.peek(j)
        if k != "id":
            return False
        if v == "array" and self.peek(j + 1)[1] == "<":
            return True
        if v in PRIMITIVES or v in self.scope.classes:
            return self.peek(j + 1)[0] == "id"
        return False

    def read_type(self):
        while self.peek()[1] in MODIFIERS:
            self.take()
        _, v = self.take()
        if v == "array":
            self.take("<")
            _, inner = self.take()
            self.take(">")
            return "array<%s>" % inner
        return v

    # ---------------------------------------------------------------- statements
    def emit(self, depth, text):
        self.lines.append("    " * depth + text)

    def block(self, depth):
        self.take("{")
        self.scope.push()
        start = len(self.lines)
        while self.peek()[1] != "}":
            self.statement(depth)
        self.take("}")
        self.scope.pop()
        if len(self.lines) == start:
            self.emit(depth, "pass")

    def body(self, depth):
        """The body of a control statement: a block or one statement."""
        start = len(self.lines)
        if self.peek()[1] == "{":
            self.block(depth)
        else:
            self.scope.push()
            self.statement(depth)
            self.scope.pop()
        if len(self.lines) == start:
            self.emit(depth, "pass")

    def statement(self, depth):
        k, v = self.peek()
        if v == "{":
            self.block(depth)
        elif v == "if":
            self.if_chain(depth, "if")
        elif v == "while":
            self.take()
            self.take("(")
            cond = self.expression(")")
            self.take(")")
            self.emit(depth, "while %s:" % cond)
            self.body(depth + 1)
        elif v == "for":
            self.for_loop(depth)
        elif v == "foreach":
            self.take()
            self.take("(")
            typ = self.read_type()
            _, name = self.take()
            self.take(":")
            src = self.expression(")")
            self.take(")")
            self.scope.push()
            self.scope.declare(name, typ)
            self.emit(depth, "for %s in list(%s):" % (name, src))
            self.body(depth + 1)
            self.scope.pop()
        elif v == "return":
            self.take()
            if self.peek()[1] == ";":
                self.emit(depth, "return")
            else:
                value = self.expression(";")
                self.emit(depth, "return %s" % self.cast(self.ret, value))
            self.take(";")
        elif v in ("break", "continue"):
            self.take()
            self.take(";")
            self.emit(depth, v)
        elif self.at_type():
            self.declaration(depth)
            self.take(";")
        else:
            self.emit(depth, self.simple(";"))
            self.take(";")

    def if_chain(self, depth, word):
        self.take("if")
        self.take("(")
        cond = self.expression(")")
        self.take(")")
        self.emit(depth, "%s %s:" % (word, cond))
        self.body(depth + 1)
        if self.peek()[1] == "else":
            self.take()
            if self.peek()[1] == "if":
                self.if_chain(depth, "elif")
            else:
                self.emit(depth, "else:")
                self.body(depth + 1)

    def for_loop(self, depth):
        self.take("for")
        self.take("(")
        self.scope.push()
        if self.at_type():
            self.declaration(depth)
        elif self.peek()[1] != ";":
            self.emit(depth, self.simple(";"))
        self.take(";")
        cond = self.expression(";") if self.peek()[1] != ";" else "True"
        self.take(";")
        step_start = self.i
        depth_paren = 0
        while not (self.peek()[1] == ")" and depth_paren == 0):
            if self.peek()[1] == "(":
                depth_paren += 1
            elif self.peek()[1] == ")":
                depth_paren -= 1
            self.take()
        step_tokens = self.t[step_start:self.i]
        self.take(")")
        self.emit(depth, "while %s:" % cond)
        start = len(self.lines)
        self.body(depth + 1)
        if any(line.strip() == "continue" for line in self.lines[start:]):
            raise TranslateError("continue inside a for loop is not translated")
        if step_tokens:
            sub = MethodTranslator(step_tokens + [("op", ";")], self.scope, self.field_types, self.ret)
            self.emit(depth + 1, sub.simple(";"))
        self.scope.pop()

    def declaration(self, depth):
        typ = self.read_type()
        while True:
            _, name = self.take()
            self.scope.declare(name, typ)
            if self.peek()[1] == "=":
                self.take()
                if self.peek()[1] == "{":
                    value = self.list_literal()
                else:
                    value = self.cast(typ, self.expression(";", ","))
            else:
                value = self.default(typ)
            self.emit(depth, "%s = %s" % (name, value))
            if self.peek()[1] != ",":
                break
            self.take(",")

    def list_literal(self):
        self.take("{")
        items = []
        while self.peek()[1] != "}":
            items.append(self.expression("}", ","))
            if self.peek()[1] == ",":
                self.take()
        self.take("}")
        return "EArray([%s])" % ", ".join(items)

    @staticmethod
    def default(typ):
        if typ == "int":
            return "0"
        if typ == "float":
            return "0.0"
        if typ == "bool":
            return "False"
        if typ == "string":
            return '""'
        if typ.startswith("array<"):
            return "None"
        return "None"

    @staticmethod
    def cast(typ, value):
        if typ == "int":
            return "_int(%s)" % value
        if typ == "float":
            return "_float(%s)" % value
        return value

    def target_type(self, target_tokens):
        """The declared type of an assignment target: a local, a field, an element of an int / float array."""
        names = [v for k, v in target_tokens if k == "id"]
        if len(target_tokens) == 1:
            name = names[0]
            return self.scope.local_type(name) or self.field_types.get(name)
        if target_tokens and target_tokens[-1][1] == "]" and names:
            typ = self.scope.local_type(names[0]) or self.field_types.get(names[0])
            if typ and typ.startswith("array<"):
                return typ[6:-1]
        return None

    def simple(self, *stops):
        """An expression statement: an assignment (= += -= *= /=), ++ / --, or a call."""
        start = self.i
        depth = 0
        while True:
            k, v = self.peek()
            if v is None:
                raise TranslateError("unterminated statement near %s" % self.context())
            if depth == 0 and v in stops:
                break
            if v in ("(", "["):
                depth += 1
            elif v in (")", "]"):
                depth -= 1
            if depth == 0 and v in ("=", "+=", "-=", "*=", "/=", "++", "--"):
                target = self.t[start:self.i]
                op = v
                self.take()
                tgt = self.sub_expression(target)
                typ = self.target_type(target)
                if op in ("++", "--"):
                    value = "%s %s 1" % (tgt, "+" if op == "++" else "-")
                else:
                    rhs = self.expression(*stops)
                    if op == "=":
                        value = rhs
                    elif op == "/=":
                        value = "_div(%s, %s)" % (tgt, rhs)
                    else:
                        value = "(%s %s (%s))" % (tgt, op[0], rhs)
                return "%s = %s" % (tgt, self.cast(typ, value) if typ in ("int", "float") else value)
            self.take()
        self.i = start
        return self.expression(*stops)

    def sub_expression(self, tokens):
        sub = MethodTranslator(tokens + [("op", ";")], self.scope, self.field_types, self.ret)
        out = sub.expression(";")
        if sub.peek()[1] != ";":
            raise TranslateError("unparsed target %s" % tokens)
        return out

    # ---------------------------------------------------------------- expressions (C precedence)
    def expression(self, *stops):
        return self.binary(0, stops)

    def binary(self, level, stops):
        if level == len(BINARY):
            return self.unary(stops)
        left = self.binary(level + 1, stops)
        while self.peek()[1] in BINARY[level] and self.peek()[1] not in stops:
            _, op = self.take()
            right = self.binary(level + 1, stops)
            if op == "&&":
                left = "(%s and %s)" % (left, right)
            elif op == "||":
                left = "(%s or %s)" % (left, right)
            elif op == "/":
                left = "_div(%s, %s)" % (left, right)
            elif op == "%":
                left = "_mod(%s, %s)" % (left, right)
            elif op == "<<":
                left = "_shl(%s, %s)" % (left, right)
            else:
                left = "(%s %s %s)" % (left, op, right)
        return left

    def unary(self, stops):
        v = self.peek()[1]
        if v == "!":
            self.take()
            return "(not %s)" % self.unary(stops)
        if v == "-":
            self.take()
            return "(-%s)" % self.unary(stops)
        if v == "+":
            self.take()
            return self.unary(stops)
        if v == "~":
            self.take()
            return "(~%s)" % self.unary(stops)
        return self.postfix(stops)

    def postfix(self, stops):
        expr, kind = self.primary(stops)
        while True:
            v = self.peek()[1]
            if v == ".":
                self.take()
                _, name = self.take()
                if self.peek()[1] == "(":
                    args = self.arguments()
                    if name == "ToString":
                        expr = "_tostr(%s)" % ", ".join([expr] + args)
                    else:
                        expr = "%s.%s(%s)" % (expr, name, ", ".join(args))
                else:
                    expr = "%s.%s" % (expr, name)
                kind = "value"
            elif v == "(" and kind == "callable":
                expr = "%s(%s)" % (expr, ", ".join(self.arguments()))
                kind = "value"
            elif v == "[":
                self.take()
                index = self.expression("]")
                self.take("]")
                expr = "%s[%s]" % (expr, index)
            else:
                return expr

    def arguments(self):
        self.take("(")
        args = []
        while self.peek()[1] != ")":
            args.append(self.expression(")", ","))
            if self.peek()[1] == ",":
                self.take()
        self.take(")")
        return args

    def primary(self, stops):
        k, v = self.take()
        if v == "(":
            inner = self.expression(")")
            self.take(")")
            return "(%s)" % inner, "value"
        if k == "num":
            return (str(int(v, 16)) if v.startswith("0x") else v), "value"
        if k == "str":
            return v, "value"
        if k != "id":
            raise TranslateError("unexpected %r near %s" % (v, self.context()))
        if v in ("true", "false"):
            return ("True" if v == "true" else "False"), "value"
        if v == "null":
            return "None", "value"
        if v == "new":
            _, cls = self.take()
            args = self.arguments()
            return "%s(%s)" % (cls, ", ".join(args)), "value"
        if v == "string" and self.peek()[1] == ".":
            self.take(".")
            _, name = self.take()
            if name == "Empty":
                return '""', "value"
            if name == "Format":
                return "_format", "callable"
            raise TranslateError("string.%s" % name)
        if v == "Print":
            return "(lambda *a: None)", "callable"
        if v == "LogLevel":
            self.take(".")
            self.take()
            return "None", "value"
        if v == "Math":
            return "Math", "value"
        if self.scope.local_type(v) is not None:
            return v, "value"
        if v in self.scope.methods:
            return "self.%s" % v, "callable"
        if v in self.scope.fields or v in self.scope.statics:
            return "self.%s" % v, "value"
        if v in self.scope.classes:
            return v, "value"
        raise TranslateError("unknown name %r near %s" % (v, self.context()))


def split_members(tokens):
    """(fields [(name, type, init tokens, static)], methods [(name, return type, params, body tokens)])."""
    fields, methods, i = [], [], 0
    while i < len(tokens):
        j = i
        mods = []
        while tokens[j][1] in MODIFIERS:
            mods.append(tokens[j][1])
            j += 1
        if tokens[j][1] == "array":
            typ = "array<%s>" % tokens[j + 2][1]
            j += 4
        else:
            typ = tokens[j][1]
            j += 1
        name = tokens[j][1]
        j += 1
        if tokens[j][1] == "(":
            depth, k = 0, j
            while True:
                if tokens[k][1] == "(":
                    depth += 1
                elif tokens[k][1] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            params = tokens[j + 1:k]
            k += 1
            if tokens[k][1] != "{":
                raise TranslateError("method %s without a body" % name)
            depth, m = 0, k
            while True:
                if tokens[m][1] == "{":
                    depth += 1
                elif tokens[m][1] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                m += 1
            methods.append((name, typ, params, tokens[k:m + 1], "static" in mods))
            i = m + 1
        else:
            k = j
            depth = 0
            while not (tokens[k][1] == ";" and depth == 0):
                if tokens[k][1] in ("{", "("):
                    depth += 1
                elif tokens[k][1] in ("}", ")"):
                    depth -= 1
                k += 1
            init = tokens[j + 1:k] if tokens[j][1] == "=" else []
            fields.append((name, typ, init, "static" in mods))
            i = k + 1
    return fields, methods


def parse_params(tokens):
    """[(name, type)]; None when a parameter is out or inout (not translated)."""
    params, cur = [], []
    for tok in tokens + [("op", ",")]:
        if tok[1] == ",":
            if cur:
                words = [v for _, v in cur if v not in MODIFIERS]
                if "out" in words or "inout" in words:
                    return None
                name = words[-1]
                typ = "".join(words[:-1])
                params.append((name, typ.replace("array<", "array<")))
            cur = []
        else:
            cur.append(tok)
    return params


def translate(source, class_names, externals=None):
    """{class name: Python class} of the given Enforce classes of source (one module, in order)."""
    externals = dict(externals or {})
    code = strip_comments(source)
    texts = {}
    for cls in class_names:
        m = re.search(r"\nclass %s(?:\s*:\s*\w+)?\s*\{" % cls, "\n" + code)
        if not m:
            raise TranslateError("no class %s" % cls)
        start = m.end()
        depth, j = 1, start
        body = "\n" + code
        while depth:
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
            elif body[j] == '"':
                j = re.compile(r'"(?:\\.|[^"\\])*"').match(body, j).end() - 1
            j += 1
        texts[cls] = body[start:j - 1]
    known_classes = set(class_names) | set(externals) | {"DC_HLCommand", "DC_NoiseInput"}
    out = []
    for cls in class_names:
        tokens = tokenize(texts[cls])
        fields, methods = split_members(tokens)
        field_types = {name: typ for name, typ, _, _ in fields}
        scope = Scope(set(field_types), set(m[0] for m in methods), known_classes,
                      set(n for n, _, _, s in fields if s))
        lines = ["class %s(object):" % cls]
        for name, typ, init, _ in fields:
            if not init:
                value = MethodTranslator.default(typ)
                if typ.startswith("array<"):
                    value = "None"
            else:
                mt = MethodTranslator(init + [("op", ";")], scope, field_types, typ)
                value = mt.list_literal() if init[0][1] == "{" else MethodTranslator.cast(typ, mt.expression(";"))
            lines.append("    _init_%s = staticmethod(lambda self: %s)" % (name, value))
        lines.append("    def __init__(self):")
        for name, _, _, _ in fields:
            lines.append("        self.%s = %s._init_%s(self)" % (name, cls, name))
        lines.append("        pass")
        for name, ret, params, body, static in methods:
            plist = parse_params(params)
            if plist is None:
                lines.append("    def %s(self, *a, **k):" % name)
                lines.append("        raise NotImplementedError('%s.%s has an out parameter')" % (cls, name))
                continue
            mt = MethodTranslator(body, scope, field_types, ret)
            scope.locals = [{}]
            for pname, ptype in plist:
                scope.declare(pname, ptype)
            mt.block(2)
            if mt.i != len(body):
                raise TranslateError("%s.%s: unparsed tokens" % (cls, name))
            args = ", ".join(["self"] + [p for p, _ in plist])
            lines.append("    def %s(%s):" % (name, args))
            for pname, ptype in plist:
                if ptype in ("int", "float"):
                    lines.append("        %s = %s" % (pname, MethodTranslator.cast(ptype, pname)))
            lines += mt.lines
        out.append("\n".join(lines))
    namespace = dict(RUNTIME)
    namespace.update(externals)
    program = "\n\n".join(out)
    try:
        exec(compile(program, "<enforce %s>" % ", ".join(class_names), "exec"), namespace)
    except SyntaxError as e:
        raise TranslateError("translated Python does not compile: %s\n%s" % (e, program))
    result = {cls: namespace[cls] for cls in class_names}
    result["__program__"] = program
    return result
