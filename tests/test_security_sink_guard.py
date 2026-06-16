"""Security sink guard — ratcheting AST check for taint-into-sink patterns.

CodeQL re-flags the same three shapes whenever new code skips the project's
sanitizer helpers. This guard catches them at PR time so the alert backlog can
only shrink. It uses a frozen baseline (``KNOWN_VIOLATIONS``): the test fails
only when a *new* violation appears outside that set. Fixing a flagged site and
removing its entry is the supported way to shrink the baseline; the set may
never grow.

Patterns detected (all AST-based, so exception-logs are not false-flagged):

1. ``str(exc)`` / f-string-of-exc reaching a ``jsonify`` value — stack-trace
   exposure. The fix is ``web.security.error_response``.
2. A request-derived value passed un-wrapped to a ``logger.*`` call — log
   injection. The fix is ``safe_log_value``. A plain ``except E as e: logger...
   {e}`` is NOT flagged (a caught exception is not a remote taint source).

SQL injection is deliberately NOT checked here: judging it needs source-to-sink
taint analysis (an f-string interpolating ``placeholders="?,?"`` is safe, one
interpolating ``request.args['x']`` is not), which a syntactic guard cannot do.
CodeQL owns that class; the bound-parameter convention is its local mitigation.
"""

import ast
from pathlib import Path

import pytest

# Frozen baseline of pre-existing violations: "<relpath>:<rule>" with a count.
# Regenerate intentionally with `python tests/test_security_sink_guard.py --baseline`.
KNOWN_VIOLATIONS: dict[str, int] = {}


def _project_root() -> Path:
    return Path(__file__).parent.parent


def _scanned_files() -> list[Path]:
    root = _project_root()
    out: list[Path] = []
    for sub in ("web", "core", "utils", "camera", "detectors", "analytics", "ingest"):
        d = root / sub
        if d.exists():
            out.extend(p for p in d.rglob("*.py") if "__pycache__" not in p.parts)
    return out


def _is_request_derived(node: ast.AST) -> bool:
    """True if the expression roots in flask ``request.*`` (remote taint)."""
    cur = node
    while isinstance(cur, (ast.Attribute, ast.Subscript, ast.Call)):
        if isinstance(cur, ast.Call):
            cur = cur.func
        elif isinstance(cur, ast.Subscript):
            cur = cur.value
        else:
            cur = cur.value
    return isinstance(cur, ast.Name) and cur.id == "request"


_SANITIZERS = {"safe_log_value", "_safe_log_value", "_slv", "safe_validation_message"}


def _unsanitized_request_use(node: ast.AST) -> bool:
    """True if a ``request.*`` value is used outside any sanitizer call.

    A sanitizer call (e.g. ``_safe_log_value(request.remote_addr)``) is a
    boundary: its arguments are considered neutralized, so the walk does not
    descend into them.
    """
    if isinstance(node, ast.Call):
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name in _SANITIZERS:
            return False
    if isinstance(node, (ast.Attribute, ast.Subscript)) and _is_request_derived(node):
        return True
    return any(_unsanitized_request_use(child) for child in ast.iter_child_nodes(node))


def _refs_exception(node: ast.AST, exc_names: set[str]) -> bool:
    """True if a bound exception name reaches this expression unsanitized.

    A sanitizer call (e.g. ``safe_validation_message(exc, ...)``) is a boundary
    whose result is bounded by a constant allow-list, so the walk stops there.
    """
    if isinstance(node, ast.Call):
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name in _SANITIZERS:
            return False
    if isinstance(node, ast.Name) and node.id in exc_names:
        return True
    return any(
        _refs_exception(child, exc_names) for child in ast.iter_child_nodes(node)
    )


class _Scanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []
        self._exc_stack: list[set[str]] = []

    # Track exception names bound by `except E as e:` so jsonify can use them.
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._exc_stack.append({node.name} if node.name else set())
        self.generic_visit(node)
        self._exc_stack.pop()

    def _exc_names(self) -> set[str]:
        names: set[str] = set()
        for s in self._exc_stack:
            names |= s
        return names

    def visit_Call(self, node: ast.Call) -> None:
        self._check_jsonify_stacktrace(node)
        self._check_logger_injection(node)
        self.generic_visit(node)

    def _check_jsonify_stacktrace(self, node: ast.Call) -> None:
        if not (isinstance(node.func, ast.Name) and node.func.id == "jsonify"):
            return
        exc_names = self._exc_names()
        if not exc_names:
            return
        for arg in node.args:
            if isinstance(arg, ast.Dict):
                for v in arg.values:
                    if v is not None and _refs_exception(v, exc_names):
                        self.violations.append(f"{node.lineno}:stack-trace-exposure")

    def _check_logger_injection(self, node: ast.Call) -> None:
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"logger", "logging", "log"}
        ):
            return
        for arg in node.args:
            if _unsanitized_request_use(arg):
                self.violations.append(f"{node.lineno}:log-injection")
                return


def _scan_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []
    scanner = _Scanner()
    scanner.visit(tree)
    return scanner.violations


def _scan_repo() -> dict[str, int]:
    root = _project_root()
    counts: dict[str, int] = {}
    for path in _scanned_files():
        rel = str(path.relative_to(root))
        for v in _scan_file(path):
            counts[f"{rel}:{v.split(':', 1)[1]}"] = (
                counts.get(f"{rel}:{v.split(':', 1)[1]}", 0) + 1
            )
    return counts


@pytest.mark.arch_hard
def test_no_new_security_sink_violations() -> None:
    """New taint-into-sink patterns must not appear beyond the frozen baseline."""
    new = _new_violations()
    assert not new, (
        "New security-sink violation(s) detected. Route through the project "
        "sanitizers (web.security.error_response / safe_log_value, bound SQL "
        "params) instead of widening the baseline:\n  " + "\n  ".join(new)
    )


def _new_violations() -> list[str]:
    current = _scan_repo()
    new = []
    for key, count in current.items():
        if count > KNOWN_VIOLATIONS.get(key, 0):
            new.append(
                f"{key} (found {count}, baseline {KNOWN_VIOLATIONS.get(key, 0)})"
            )
    return sorted(new)


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        # Fork-sparing entry point for the pre-push hook (no pytest).
        violations = _new_violations()
        if violations:
            print("New security-sink violation(s):\n  " + "\n  ".join(violations))
            sys.exit(1)
        sys.exit(0)

    found = _scan_repo()
    if "--baseline" in sys.argv:
        print("KNOWN_VIOLATIONS: dict[str, int] = {")
        for key in sorted(found):
            print(f'    "{key}": {found[key]},')
        print("}")
    else:
        for key in sorted(found):
            print(f"{key}: {found[key]}")
        print(f"\nTotal: {sum(found.values())} across {len(found)} keys")
