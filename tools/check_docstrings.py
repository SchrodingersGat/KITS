"""Require a docstring on every function and class, and every argument documented.

Ruff's pydocstyle rules only require docstrings on *public* objects, and D417
only checks arguments when a docstring already has an ``Args:`` section. This
checker closes both gaps for Google-style docstrings:

* every module, class, function and method (including private ones and nested
  functions) has a docstring;
* every argument except ``self``/``cls`` is listed in the ``Args:`` section.

Usage: ``python tools/check_docstrings.py [paths...]`` (defaults to ``src``).
Exits non-zero and prints ``path:line: message`` for each problem.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterator
from pathlib import Path

_SECTION_RE = re.compile(r"^(\s*)(Args|Arguments|Parameters):\s*$")
_ENTRY_RE = re.compile(r"^(\s*)(\*{0,2}\w+)\s*(\([^)]*\))?\s*:")
_IMPLICIT_ARGS = {"self", "cls"}


def documented_args(docstring: str) -> set[str]:
    """Return the argument names listed in a Google-style ``Args:`` section.

    Args:
        docstring: The docstring text to parse.

    Returns:
        Argument names, without any leading ``*``/``**``.
    """
    names: set[str] = set()
    lines = docstring.expandtabs().splitlines()
    i = 0
    while i < len(lines):
        section = _SECTION_RE.match(lines[i])
        i += 1
        if not section:
            continue
        section_indent = len(section.group(1))
        entry_indent = None
        while i < len(lines):
            line = lines[i]
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent <= section_indent:
                    break  # next section or end of docstring
                if entry_indent is None:
                    entry_indent = indent
                if indent == entry_indent and (entry := _ENTRY_RE.match(line)):
                    names.add(entry.group(2).lstrip("*"))
            i += 1
    return names


def function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return the names of a function's arguments that must be documented.

    Args:
        node: The function definition.

    Returns:
        Argument names in signature order, excluding ``self``/``cls``.
    """
    a = node.args
    names = [arg.arg for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
    names += [arg.arg for arg in (a.vararg, a.kwarg) if arg is not None]
    return [n for n in names if n not in _IMPLICIT_ARGS]


def check_source(source: str, filename: str) -> Iterator[str]:
    """Yield a ``filename:line: message`` string for each docstring problem.

    Args:
        source: Python source code to check.
        filename: Name used in the reported messages.

    Yields:
        One message per missing docstring or undocumented argument.
    """
    tree = ast.parse(source, filename)
    if ast.get_docstring(tree) is None and source.strip():
        yield f"{filename}:1: module has no docstring"

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if ast.get_docstring(node) is None:
                yield f"{filename}:{node.lineno}: class {node.name!r} has no docstring"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            if docstring is None:
                yield f"{filename}:{node.lineno}: function {node.name!r} has no docstring"
                continue
            missing = [n for n in function_args(node) if n not in documented_args(docstring)]
            if missing:
                yield (
                    f"{filename}:{node.lineno}: function {node.name!r} does not document "
                    f"argument(s) {', '.join(missing)} in an Args: section"
                )


def iter_python_files(paths: list[str]) -> Iterator[Path]:
    """Yield the ``.py`` files named by, or found under, ``paths``.

    Args:
        paths: Files and/or directories to search.

    Yields:
        Python source file paths.
    """
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(p.rglob("*.py"))
        elif p.suffix == ".py":
            yield p


def main(argv: list[str] | None = None) -> int:
    """Check the given paths and report problems.

    Args:
        argv: Paths to check; defaults to ``src``.

    Returns:
        Process exit code: 0 if everything is documented, 1 otherwise.
    """
    paths = argv if argv else ["src"]
    problems = [
        msg
        for path in iter_python_files(paths)
        for msg in check_source(path.read_text(encoding="utf-8"), str(path))
    ]
    for msg in problems:
        print(msg)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
