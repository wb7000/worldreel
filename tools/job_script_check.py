"""Compile-check the step scripts embedded in Kaggle jobs.

Kaggle jobs write their real program as a triple-quoted string passed to
textwrap.dedent (or to a helper that calls dedent) and execute it on the GPU.
py_compile on the job file only proves the *outer* file parses -- a nested
triple-quote inside the embedded script silently truncates the string, so the
job dies on Kaggle only after the queue wait. This happened once: a docstring
at line 280 broke character_consistency_v2.py.

Extraction is AST-based rather than a search for the next triple-quote, because
jobs pass their scripts in several shapes (shown with the quotes spelled out to
keep this docstring parseable):

    step.write_text(textwrap.dedent(<triple-quoted source>))   # direct
    run(tts_py, "tts", <triple-quoted source>, args)          # literal argument
    run(py, name, code, args)                                 # code is a param

so a naive "find the next triple-quote after dedent(" lands on an unrelated
docstring and reports a bogus failure.

A job with no embedded script is skipped, not failed. Note: this file's own
docstring must not contain three consecutive double-quote characters anywhere,
or it would fail the very check it implements.
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

DEDENT_NAMES = {"dedent", "textwrap.dedent"}
# Call sites that consume source text rather than display text.
CODE_CALL_NAMES = {"run", "exec", "exec_code", "write_text", "compile"}


def _name(node: ast.expr) -> str:
    """dotted name of a call target, e.g. textwrap.dedent"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _const_str(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _resolve(node: ast.expr, bindings: dict[str, str]) -> str | None:
    """Follow a Name back to the string it was assigned, if any."""
    direct = _const_str(node)
    if direct is not None:
        return direct
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    return None


def collect(tree: ast.AST) -> list[str]:
    """Every string literal in the file that is used as source code."""
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = _const_str(node.value)
            if value is None:
                continue
            for lhs in node.targets:
                if isinstance(lhs, ast.Name):
                    bindings[lhs.id] = value

    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _name(node.func)
        short = callee.rsplit(".", 1)[-1]

        if callee in DEDENT_NAMES or short == "dedent" or short in CODE_CALL_NAMES:
            candidates = list(node.args)
        else:
            continue

        for arg in candidates:
            value = _resolve(arg, bindings)
            if value is None:
                continue
            # Multi-line means *several* lines, not one line with an escaped
            # newline: marker.write_text("...\n") is data, not a script, and
            # reporting it as a syntax error would be a bogus failure.
            lines = [ln for ln in value.splitlines() if ln.strip()]
            if len(lines) >= 2:
                found.append(value)
    return found


def check(paths: list[Path]) -> list[str]:
    """Return one human-readable failure per broken job; empty means all good."""
    failures: list[str] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            failures.append(f"{path}: job file itself does not parse: {exc.msg}")
            continue

        for index, code in enumerate(collect(tree)):
            try:
                ast.parse(textwrap.dedent(code))
            except SyntaxError as exc:
                failures.append(
                    f"{path.name}: embedded script #{index} line {exc.lineno}: {exc.msg}"
                )
    return failures


def main() -> int:
    jobs = [Path(j) for j in sys.argv[1:]] or sorted(Path("training/kaggle/jobs").glob("*.py"))
    failures = check(jobs)
    for failure in failures:
        print(f"FAIL {failure}")
    if not failures:
        print(f"all {len(jobs)} job files: embedded scripts parse")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
