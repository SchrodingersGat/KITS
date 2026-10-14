"""Minimal, tolerant S-expression reader for KiCad files.

KiCad files are trees of ``(keyword arg arg (child ...) ...)`` nodes. This
module turns the text into nested Python lists: each node is a ``list`` whose
first element is the keyword. Bare atoms (keywords, numbers, enum values) are
plain ``str``; quoted strings are :class:`Quoted` so callers can tell
``(net 3)`` (a net number) apart from ``(net "3")`` (a net name).

Unknown tokens are kept as-is rather than rejected, so new KiCad file-format
additions don't break ingest — the board loader only picks out what it needs.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

Node = list  # list[str | Quoted | Node], first element is the keyword

_TOKEN_RE = re.compile(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+')
_ESCAPE_RE = re.compile(r"\\(.)")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}


class SExprError(ValueError):
    """Raised when the input is not a well-formed S-expression."""


class Quoted(str):
    """A string atom that was quoted in the source file."""

    __slots__ = ()


def _unescape(body: str) -> str:
    if "\\" not in body:
        return body
    return _ESCAPE_RE.sub(lambda m: _ESCAPES.get(m.group(1), m.group(1)), body)


def parse(text: str) -> Node:
    """Parse ``text`` containing a single top-level S-expression."""
    stack: list[Node] = []
    root: Node | None = None

    for tok in _TOKEN_RE.findall(text):
        if tok == "(":
            node: Node = []
            if stack:
                stack[-1].append(node)
            elif root is not None:
                raise SExprError("multiple top-level expressions")
            else:
                root = node
            stack.append(node)
        elif tok == ")":
            if not stack:
                raise SExprError("unbalanced ')'")
            stack.pop()
        elif not stack:
            raise SExprError(f"atom {tok!r} outside of any expression")
        elif tok[0] == '"':
            stack[-1].append(Quoted(_unescape(tok[1:-1])))
        else:
            stack[-1].append(tok)

    if stack:
        raise SExprError("unexpected end of input (unclosed '(')")
    if root is None:
        raise SExprError("no expression found")
    return root


def children(node: Node, keyword: str | None = None) -> Iterator[Node]:
    """Yield child nodes, optionally only those with the given keyword."""
    for item in node[1:]:
        if isinstance(item, list) and item and (keyword is None or item[0] == keyword):
            yield item


def find(node: Node, keyword: str) -> Node | None:
    """Return the first child node with ``keyword``, or None."""
    return next(children(node, keyword), None)


def value(node: Node, keyword: str, default=None):
    """Return the first argument of child ``keyword``, or ``default``."""
    child = find(node, keyword)
    if child is None or len(child) < 2:
        return default
    return child[1]


def has_flag(node: Node, flag: str) -> bool:
    """True if ``flag`` appears as a bare atom argument of ``node``.

    KiCad writes some booleans as bare atoms, e.g. ``(via blind (at ...))``.
    """
    return any(item == flag and not isinstance(item, (list, Quoted)) for item in node[1:])
