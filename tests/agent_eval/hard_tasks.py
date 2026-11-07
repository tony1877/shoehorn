"""Hard tier of the agent eval (see gen_tasks.py): tasks the current agent should fail about half the time.

  big_repo      a 30-module package; a wrong total comes from a cache decorator that ignores keyword arguments,
                with look-alike helpers as decoys. Needs searching, not reading everything.
  tz_bucket     events are counted per day in local time instead of UTC; the arithmetic helper that looks guilty
                is correct (two hypotheses)
  long_file     three named handlers out of 30 near-identical ones in a ~900-line file need a fix; the rest must
                stay byte-identical
  order_dep     tests pass one at a time and fail together: a mutable default argument shares state
  cp1252        a CLI prints a check mark; it works in a console and crashes when its output is piped (tests run it
                through subprocess)
  semver_range  implement a version-range matcher from a spec full of edge cases
  mini_interp   implement a small expression language (tokenizer, parser, evaluator) from SPEC.md: long horizon
"""
import json

STANDARD_RULES = ("Run the tests with `python -m unittest` from the workspace root. Do not change files under tests/ "
                  "unless the task says so. When you finish, report what you changed and the test result.")


def big_repo(t):
    names = ["orders", "invoices", "customers", "catalog", "shipping", "returns", "coupons", "audit", "ledger", "tax",
             "fx", "inventory", "warehouse", "suppliers", "reports", "exports", "imports", "notify", "billing",
             "refunds", "loyalty", "pricing_rules", "regions", "calendar", "limits", "quotas", "metrics", "search",
             "sessions", "webhooks"]
    t.w("shop/__init__.py", "")
    t.w("shop/util/__init__.py", "")
    t.w("shop/util/cache.py", '''
        """Memoization for pure lookups (rates, rules). Values are cached per argument tuple."""
        import functools


        def memo(fn):
            store = {}

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                key = (fn.__name__,) + args
                if key not in store:
                    store[key] = fn(*args, **kwargs)
                return store[key]

            wrapper.cache_clear = store.clear
            return wrapper


        def memo_kw(fn):
            """Same as memo, keyed on keyword arguments too. Used by the reporting module."""
            store = {}

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                key = (fn.__name__,) + args + tuple(sorted(kwargs.items()))
                if key not in store:
                    store[key] = fn(*args, **kwargs)
                return store[key]

            return wrapper
        ''')
    t.w("shop/util/money.py", '''
        def cents(x):
            return int(round(x * 100))


        def fmt(c):
            return f"{c // 100}.{c % 100:02d}"
        ''')
    t.w("shop/fx.py", '''
        from .util.cache import memo

        RATES = {"EUR": 1.0, "USD": 0.92, "GBP": 1.17}


        @memo
        def rate(currency, *, inverse=False):
            r = RATES[currency]
            return 1 / r if inverse else r
        ''')
    t.w("shop/orders.py", '''
        from .fx import rate
        from .util.money import cents


        def total_eur(lines):
            """lines: (amount, currency) pairs -> total in EUR cents."""
            return sum(cents(a * rate(c)) for a, c in lines)


        def total_in(lines, currency):
            """Total converted from EUR into `currency`, in cents."""
            return cents(total_eur(lines) / 100 * rate(currency, inverse=True))
        ''')
    for i, n in enumerate(names):
        if n in ("orders", "fx"):
            continue
        t.w(f"shop/{n}.py", f'''
            """{n.replace('_', ' ').title()} helpers."""
            from .util.cache import memo
            from .util.money import cents


            @memo
            def {n}_factor(kind):
                return {{"std": 1.0, "promo": 0.{9 - i % 5}, "bulk": 0.8{i % 10}}}.get(kind, 1.0)


            def {n}_apply(amount, kind="std"):
                return cents(amount * {n}_factor(kind)) / 100
            ''')
    t.w("tests/__init__.py", "")
    t.w("tests/test_orders.py", '''
        import unittest
        from shop.orders import total_eur, total_in


        class OrdersTest(unittest.TestCase):
            def test_eur(self):
                self.assertEqual(total_eur([(10, "EUR"), (10, "USD")]), 1920)

            def test_usd_after_eur(self):
                total_eur([(1, "USD")])
                self.assertEqual(total_in([(92, "EUR")], "USD"), 10000)
        ''')
    t.hidden("test_hidden.py", '''
        import unittest
        from shop import fx
        from shop.orders import total_eur, total_in


        class Hidden(unittest.TestCase):
            def test_rate_kw(self):
                self.assertEqual(fx.rate("GBP"), 1.17)
                self.assertAlmostEqual(fx.rate("GBP", inverse=True), 1 / 1.17)
                self.assertEqual(fx.rate("GBP"), 1.17)

            def test_totals(self):
                self.assertEqual(total_in([(117, "EUR")], "GBP"), 10000)
                self.assertEqual(total_eur([(100, "GBP")]), 11700)

            def test_still_cached(self):
                calls = []
                orig = fx.RATES
                class Spy(dict):
                    def __getitem__(self, k):
                        calls.append(k)
                        return dict.__getitem__(self, k)
                fx.RATES = Spy(orig)
                try:
                    if hasattr(fx.rate, "cache_clear"):
                        fx.rate.cache_clear()
                    fx.rate("USD"); fx.rate("USD")
                    self.assertEqual(calls.count("USD"), 1, "rate() must stay memoized")
                finally:
                    fx.RATES = orig
        ''')
    t.prompt = (f"Converting an order total to USD gives a wrong amount (tests/test_orders.py shows it). Find the root "
                f"cause and fix it there, keeping lookups memoized. {STANDARD_RULES}")


def tz_bucket(t):
    t.w("events/__init__.py", "")
    t.w("events/daily.py", '''
        """Daily event counts. Timestamps are Unix seconds; days are UTC calendar days."""
        from datetime import datetime

        DAY = 86400


        def day_start(ts):
            """Start of the UTC day containing ts (pure arithmetic: Unix time has no leap seconds)."""
            return ts - ts % DAY


        def day_label(ts):
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


        def counts(timestamps):
            out = {}
            for ts in timestamps:
                label = day_label(ts)
                out[label] = out.get(label, 0) + 1
            return dict(sorted(out.items()))
        ''')
    t.w("tests/__init__.py", "")
    t.w("tests/test_daily.py", '''
        import unittest
        from events.daily import counts

        # 2026-03-01 00:00:00 UTC
        MAR1 = 1772323200


        class DailyTest(unittest.TestCase):
            def test_counts(self):
                self.assertEqual(counts([MAR1 + 10, MAR1 + 86390, MAR1 + 86410]),
                                 {"2026-03-01": 2, "2026-03-02": 1})
        ''')
    t.hidden("test_hidden.py", '''
        import os
        import time
        import unittest
        from events.daily import counts, day_start

        MAR1 = 1772323200


        class Hidden(unittest.TestCase):
            def test_day_start(self):
                self.assertEqual(day_start(MAR1 + 5), MAR1)

            def test_any_tz(self):
                if not hasattr(time, "tzset"):
                    self.assertEqual(counts([MAR1 - 1, MAR1]), {"2026-02-28": 1, "2026-03-01": 1})
                    return
                for tz in ("UTC", "America/New_York", "Asia/Tokyo"):
                    os.environ["TZ"] = tz
                    time.tzset()
                    self.assertEqual(counts([MAR1 - 1, MAR1]), {"2026-02-28": 1, "2026-03-01": 1})

            def test_end_of_day(self):
                self.assertEqual(counts([MAR1 + 86399]), {"2026-03-01": 1})
        ''')
    t.prompt = f"The daily event counts are wrong on this machine. Find out why and fix it. {STANDARD_RULES}"


def long_file(t):
    kinds = ["user", "order", "invoice", "refund", "coupon", "address", "payment", "shipment", "review", "ticket",
             "vendor", "product", "price", "stock", "batch", "report", "export", "import", "session", "token",
             "webhook", "event", "metric", "alert", "audit", "role", "team", "project", "task", "comment"]
    targets = t.rnd.sample(kinds[3:], 3)
    blocks = []
    for k in kinds:
        cls = k.capitalize() + "Handler"
        blocks.append(f'''

class {cls}(BaseHandler):
    """CRUD handler for {k} records."""

    kind = "{k}"
    required = ("id", "name")

    def validate(self, record):
        missing = [f for f in self.required if f not in record]
        if missing:
            raise ValueError(f"{k}: missing {{', '.join(missing)}}")
        if len(record["name"]) > 64:
            raise ValueError("{k}: name too long")
        return True

    def create(self, record):
        self.validate(record)
        self.rows[record["id"]] = dict(record)
        return self.rows[record["id"]]

    def update(self, record_id, changes):
        row = dict(self.rows[record_id], **changes)
        self.validate(row)
        self.rows[record_id] = row
        return row
''')
    body = ('"""Record handlers. Each kind has its own class on purpose: they diverge in production."""\n\n\n'
            'class BaseHandler:\n    def __init__(self):\n        self.rows = {}\n' + "".join(blocks))
    t.w("handlers.py", body)
    expected = body
    for k in targets:
        old = f'''        if len(record["name"]) > 64:
            raise ValueError("{k}: name too long")
        return True'''
        new = f'''        if len(record["name"]) > 64:
            raise ValueError("{k}: name too long")
        if not record["name"].strip():
            raise ValueError("{k}: name is blank")
        return True'''
        assert expected.count(old) == 1
        expected = expected.replace(old, new)
    with open(f"{t.key}/expected_handlers.py", "w", encoding="utf-8", newline="\n") as f:
        f.write(expected)
    t.w("tests/__init__.py", "")
    t.w("tests/test_handlers.py", '''
        import unittest
        import handlers


        class HandlerTest(unittest.TestCase):
            def test_create(self):
                h = handlers.UserHandler()
                self.assertEqual(h.create({"id": 1, "name": "a"})["name"], "a")
        ''')
    t.grade = {"hidden_tests": False, "exact_file": {"path": "handlers.py", "expected": "expected_handlers.py"}}
    names = ", ".join(k.capitalize() + "Handler" for k in targets)
    t.prompt = (f"In handlers.py, make validate() of exactly these three classes reject a name that is empty or only "
                f"whitespace with ValueError(\"<kind>: name is blank\"), checked right after the length check: "
                f"{names}. Use the same two-line if/raise style as the length check. No other class may change, "
                f"and nothing else in the file may change. {STANDARD_RULES}")


def order_dep(t):
    t.w("tags/__init__.py", "")
    t.w("tags/core.py", '''
        def add_tags(item, new=(), existing=[]):
            """Return the item's tag list with `new` appended, without duplicates, keeping order."""
            for tag in list(item.get("tags", existing)) + list(new):
                if tag not in existing:
                    existing.append(tag)
            item["tags"] = existing
            return existing


        def tag_counts(items):
            counts = {}
            for item in items:
                for tag in item.get("tags", []):
                    counts[tag] = counts.get(tag, 0) + 1
            return counts
        ''')
    t.w("tests/__init__.py", "")
    t.w("tests/test_a_add.py", '''
        import unittest
        from tags.core import add_tags


        class AddTest(unittest.TestCase):
            def test_add(self):
                self.assertEqual(add_tags({}, ["x", "y", "x"]), ["x", "y"])
        ''')
    t.w("tests/test_b_counts.py", '''
        import unittest
        from tags.core import add_tags, tag_counts


        class CountTest(unittest.TestCase):
            def test_counts(self):
                a = {}
                add_tags(a, ["p"])
                self.assertEqual(tag_counts([a]), {"p": 1})
        ''')
    t.hidden("test_hidden.py", '''
        import unittest
        from tags.core import add_tags, tag_counts


        class Hidden(unittest.TestCase):
            def test_independent(self):
                a, b = {}, {}
                add_tags(a, ["p"]); add_tags(b, ["q"])
                self.assertEqual((a["tags"], b["tags"]), (["p"], ["q"]))

            def test_existing_kept(self):
                item = {"tags": ["old"]}
                self.assertEqual(add_tags(item, ["new", "old"]), ["old", "new"])

            def test_input_not_aliased(self):
                src = ["k"]
                item = {"tags": src}
                add_tags(item, ["z"])
                self.assertEqual(src, ["k"])

            def test_counts(self):
                a, b = {}, {}
                add_tags(a, ["p", "q"]); add_tags(b, ["q"])
                self.assertEqual(tag_counts([a, b]), {"p": 1, "q": 2})
        ''')
    t.prompt = (f"Each test file passes when run on its own, but `python -m unittest` fails. Fix the code. "
                f"{STANDARD_RULES}")


def cp1252(t):
    t.w("checker/__init__.py", "")
    t.w("checker/__main__.py", '''
        """python -m checker FILE...: report which files end with a newline."""
        import sys


        def main(paths):
            bad = 0
            for p in paths:
                with open(p, "rb") as f:
                    data = f.read()
                ok = data.endswith(b"\\n")
                bad += not ok
                print(f"{'✓' if ok else '✗'} {p}")
            print(f"{len(paths) - bad}/{len(paths)} files end with a newline")
            return 1 if bad else 0


        if __name__ == "__main__":
            sys.exit(main(sys.argv[1:]))
        ''')
    t.w("tests/__init__.py", "")
    t.w("tests/test_cli.py", '''
        import os
        import subprocess
        import sys
        import tempfile
        import unittest


        def run(*paths):
            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
            return subprocess.run([sys.executable, "-m", "checker", *paths], capture_output=True, env=env)


        class CliTest(unittest.TestCase):
            def test_piped_output(self):
                with tempfile.TemporaryDirectory() as d:
                    good = os.path.join(d, "a.txt")
                    open(good, "w").write("x\\n")
                    p = run(good)
                    self.assertEqual(p.returncode, 0, p.stderr.decode(errors="replace"))
                    self.assertIn("1/1 files", p.stdout.decode("utf-8"))
        ''')
    t.hidden("test_hidden.py", '''
        import os
        import subprocess
        import sys
        import tempfile
        import unittest


        def run(*paths):
            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
            return subprocess.run([sys.executable, "-m", "checker", *paths], capture_output=True, env=env)


        class Hidden(unittest.TestCase):
            def test_marks_utf8(self):
                with tempfile.TemporaryDirectory() as d:
                    a, b = os.path.join(d, "a"), os.path.join(d, "b")
                    open(a, "w").write("x\\n"); open(b, "w").write("x")
                    p = run(a, b)
                    out = p.stdout.decode("utf-8")
                    self.assertEqual(p.returncode, 1)
                    self.assertIn("\\u2713 " + a, out)
                    self.assertIn("\\u2717 " + b, out)
                    self.assertIn("1/2 files", out)
        ''')
    t.prompt = (f"`python -m checker` works when I run it in my terminal, but the test that runs it through "
                f"subprocess fails. Fix the program (keep the check marks, output must be UTF-8). {STANDARD_RULES}")


def semver_range(t):
    t.w("semver/__init__.py", "")
    t.w("semver/match.py", '''
        """Version ranges. Implement satisfies(version, rng) to this spec:

        - A version is MAJOR.MINOR.PATCH with an optional pre-release: 1.2.3-beta.2. Pre-release identifiers are
          compared left to right: numeric ones numerically, others as ASCII strings, numeric < non-numeric, and a
          shorter list is lower when all shared identifiers are equal. Any pre-release is lower than its release.
        - A range is one or more comparator sets joined by "||"; it matches if any set matches.
        - A set is comparators separated by whitespace, all of which must match. Comparators: =V, V (same as =V),
          >V, >=V, <V, <=V, ^V and ~V.
        - ^1.2.3 means >=1.2.3 <2.0.0; ^0.2.3 means >=0.2.3 <0.3.0; ^0.0.3 means >=0.0.3 <0.0.4.
        - ~1.2.3 means >=1.2.3 <1.3.0.
        - "*" or an empty set matches every release version.
        - A pre-release version only matches a set if some comparator in that set has a pre-release with the same
          MAJOR.MINOR.PATCH (so 1.3.0-beta does not satisfy >=1.2.0, but 1.2.0-beta.3 satisfies >=1.2.0-beta.2).
        - Whitespace around operators is allowed: ">= 1.2.3" equals ">=1.2.3".
        - Invalid input raises ValueError.
        """


        def satisfies(version, rng):
            raise NotImplementedError
        ''')
    t.w("tests/__init__.py", "")
    t.w("tests/test_match.py", '''
        import unittest
        from semver.match import satisfies


        class MatchTest(unittest.TestCase):
            def test_caret(self):
                self.assertTrue(satisfies("1.4.0", "^1.2.3"))
                self.assertFalse(satisfies("2.0.0", "^1.2.3"))
        ''')
    t.hidden("test_hidden.py", '''
        import unittest
        from semver.match import satisfies as s


        class Hidden(unittest.TestCase):
            def test_basic(self):
                self.assertTrue(s("1.2.3", "1.2.3"))
                self.assertTrue(s("1.2.3", "=1.2.3"))
                self.assertFalse(s("1.2.4", "1.2.3"))
                self.assertTrue(s("1.5.0", ">=1.2.0 <2.0.0"))
                self.assertFalse(s("2.0.0", ">=1.2.0 <2.0.0"))
                self.assertTrue(s("3.0.0", "<1.0.0 || >=3.0.0"))

            def test_caret_zero(self):
                self.assertTrue(s("0.2.9", "^0.2.3"))
                self.assertFalse(s("0.3.0", "^0.2.3"))
                self.assertFalse(s("0.0.4", "^0.0.3"))
                self.assertTrue(s("0.0.3", "^0.0.3"))

            def test_tilde(self):
                self.assertTrue(s("1.2.9", "~1.2.3"))
                self.assertFalse(s("1.3.0", "~1.2.3"))

            def test_star(self):
                self.assertTrue(s("9.9.9", "*"))
                self.assertFalse(s("9.9.9-rc.1", "*"))

            def test_prerelease_order(self):
                self.assertTrue(s("1.2.0-beta.11", ">1.2.0-beta.2"))
                self.assertTrue(s("1.2.0-beta", "<1.2.0-beta.1"))
                self.assertTrue(s("1.2.0-alpha.9", "<1.2.0-alpha.beta"))
                self.assertTrue(s("1.2.0-rc.1", "<1.2.0 >=1.2.0-rc.1"))

            def test_prerelease_gate(self):
                self.assertFalse(s("1.3.0-beta", ">=1.2.0"))
                self.assertTrue(s("1.2.0-beta.3", ">=1.2.0-beta.2"))
                self.assertFalse(s("1.2.1-beta.3", ">=1.2.0-beta.2"))

            def test_spaces(self):
                self.assertTrue(s("1.2.3", ">= 1.2.3"))
                self.assertTrue(s("1.2.3", "  >=1.0.0   <2.0.0 "))

            def test_invalid(self):
                for bad in [("1.2", "*"), ("1.2.3", ">>1.0.0"), ("a.b.c", "1.0.0"), ("1.2.3", "^x")]:
                    with self.assertRaises(ValueError):
                        s(*bad)
        ''')
    t.prompt = f"Implement semver.match.satisfies exactly to the spec in its docstring. {STANDARD_RULES}"


def mini_interp(t):
    t.w("SPEC.md", '''
        # calc: a tiny expression language

        A program is a sequence of statements separated by newlines or `;`. Blank lines and `#` comments (to end of
        line) are ignored.

        - `let NAME = EXPR` binds a new variable in the current scope. Rebinding a name in the same scope is an error.
        - `set NAME = EXPR` assigns an existing variable (searching outward through scopes). Unknown name: error.
        - `print EXPR` appends the value's text to the output list.
        - Any other statement is an expression; its value is discarded.

        Expressions, lowest to highest precedence:

        1. `a or b`, `a and b` (short-circuit; `or` is lower than `and`); `not a`
        2. comparisons `== != < <= > >=` (non-associative: `a < b < c` is an error)
        3. `+ -` (left-assoc), 4. `* / %` (left-assoc; `/` is integer division truncating toward zero; division or
           modulo by zero is an error), 5. unary `-`, 6. `^` power (right-assoc, binds tighter than unary minus:
           `-2^2` is -4), 7. primaries: integers, `true`, `false`, names, `( EXPR )`, `if C then A else B` (an
           expression), and blocks `{ stmt; stmt; ... }` whose value is the last statement's value (`nil` if none)
           and which open a new scope.

        Values are integers, booleans and nil. Text form: integers in decimal, `true`/`false`, `nil`. Arithmetic on
        booleans or nil is an error; `==`/`!=` compare any values (different types are unequal); ordering comparisons
        need two integers. `and`/`or` return booleans and need boolean operands (the right one only when evaluated).

        `run(source)` returns the list of printed strings. Every error raises `CalcError` with a message.
        ''')
    t.w("calc/__init__.py", "from .interp import CalcError, run\n")
    t.w("calc/interp.py", '''
        """Implement the language in SPEC.md."""


        class CalcError(Exception):
            pass


        def run(source):
            raise NotImplementedError
        ''')
    t.w("tests/__init__.py", "")
    t.w("tests/test_calc.py", '''
        import unittest
        from calc import run


        class CalcTest(unittest.TestCase):
            def test_print(self):
                self.assertEqual(run("print 1 + 2 * 3"), ["7"])
        ''')
    t.hidden("test_hidden.py", '''
        import unittest
        from calc import CalcError, run


        class Hidden(unittest.TestCase):
            def eq(self, src, out):
                self.assertEqual(run(src), out)

            def err(self, src):
                with self.assertRaises(CalcError):
                    run(src)

            def test_arith(self):
                self.eq("print 7 / 2; print -7 / 2; print 7 % 3", ["3", "-3", "1"])
                self.eq("print 2 ^ 3 ^ 2", ["512"])
                self.eq("print -2 ^ 2", ["-4"])
                self.eq("print (1 + 2) * 3 - 4", ["5"])

            def test_errors(self):
                self.err("print 1 / 0")
                self.err("print 1 < 2 < 3")
                self.err("print true + 1")
                self.err("print x")
                self.err("let a = 1; let a = 2")
                self.err("set q = 1")
                self.err("print (1 + 2")

            def test_logic(self):
                self.eq("print true and not false or false", ["true"])
                self.eq("print false and (1 / 0 == 1)", ["false"])
                self.eq("print 1 == true; print nil == nil", ["false", "true"])
                self.err("print 1 and true")

            def test_scopes(self):
                self.eq("let a = 1\\n{ let a = 2; print a }\\nprint a", ["2", "1"])
                self.eq("let a = 1; { set a = 5 }; print a", ["5"])
                self.eq("print { }", ["nil"])
                self.eq("let x = { let y = 3; y * 2 }; print x", ["6"])

            def test_if(self):
                self.eq("let n = 4\\nprint if n % 2 == 0 then n / 2 else 3 * n + 1", ["2"])
                self.eq("# comment\\n\\nprint 1 # trailing\\n", ["1"])
        ''')
    t.prompt = (f"Implement the language described in SPEC.md in calc/interp.py: run(source) and CalcError. Write "
                f"your own tests as you go. {STANDARD_RULES}")


# cp1252 is kept for machines without a UTF-8 system code page; on this one piped output is already UTF-8.
FAMILIES = {f.__name__: f for f in (big_repo, tz_bucket, long_file, order_dep, semver_range, mini_interp)}
