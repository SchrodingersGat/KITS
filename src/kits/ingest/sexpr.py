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
    """Resolve backslash escapes in the body of a quoted string.

    Backslash-n, -t and -r become newline, tab and carriage return; any other
    escaped character is taken literally (e.g. an escaped quote becomes ``"``).

    Args:
        body: Quoted-string contents with the surrounding quotes removed.

    Returns:
        The unescaped string.
    """
    if "\\" not in body:
        return body
    return _ESCAPE_RE.sub(lambda m: _ESCAPES.get(m.group(1), m.group(1)), body)


def parse(text: str) -> Node:
    """Parse ``text`` containing a single top-level S-expression.

    Args:
        text: Full source text, e.g. the contents of a ``.kicad_pcb`` file.

    Returns:
        The root node as nested lists; quoted atoms are :class:`Quoted`.

    Raises:
        SExprError: If parentheses are unbalanced, there is more than one
            top-level expression, an atom lies outside any expression, or the
            input contains no expression.
    """
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
    """Yield child nodes, optionally only those with the given keyword.

    Args:
        node: Parent node; its keyword (first element) is skipped.
        keyword: If given, only yield children whose keyword equals this.

    Yields:
        Non-empty child nodes in file order. Atom arguments are skipped.
    """
    for item in node[1:]:
        if isinstance(item, list) and item and (keyword is None or item[0] == keyword):
            yield item


def find(node: Node, keyword: str) -> Node | None:
    """Return the first child node with ``keyword``, or None.

    Args:
        node: Parent node to search (direct children only).
        keyword: Keyword of the child to find.

    Returns:
        The first matching child node, or None if there is none.
    """
    return next(children(node, keyword), None)


def value(node: Node, keyword: str, default=None):
    """Return the first argument of child ``keyword``, or ``default``.

    For example ``value(node, "layer")`` on ``(... (layer "F.Cu"))`` returns
    ``"F.Cu"``.

    Args:
        node: Parent node to search (direct children only).
        keyword: Keyword of the child whose first argument is wanted.
        default: Value returned if the child is missing or has no arguments.

    Returns:
        The child's first argument (an atom string, :class:`Quoted` or node),
        or ``default``.
    """
    child = find(node, keyword)
    if child is None or len(child) < 2:
        return default
    return child[1]


def has_flag(node: Node, flag: str) -> bool:
    """True if ``flag`` appears as a bare atom argument of ``node``.

    KiCad writes some booleans as bare atoms, e.g. ``(via blind (at ...))``.
    Quoted strings and child nodes never match.

    Args:
        node: Node whose arguments are checked.
        flag: Bare atom to look for, e.g. ``"blind"``.

    Returns:
        True if the flag is present.
    """
    return any(item == flag and not isinstance(item, (list, Quoted)) for item in node[1:])
