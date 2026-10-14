import pytest

from kits.ingest.sexpr import Quoted, SExprError, find, has_flag, parse, value


def test_parse_nested_and_quoted():
    node = parse(r'(root (a 1 2) (name "hello \"world\"") flag (b (c x)))')
    assert node[0] == "root"
    assert find(node, "a") == ["a", "1", "2"]
    name = value(node, "name")
    assert name == 'hello "world"'
    assert isinstance(name, Quoted)
    assert find(find(node, "b"), "c") == ["c", "x"]
    assert has_flag(node, "flag")
    assert not has_flag(node, "missing")


def test_bare_atoms_are_not_quoted():
    node = parse('(net 3 "3")')
    assert not isinstance(node[1], Quoted)
    assert isinstance(node[2], Quoted)


def test_quoted_string_with_parens_and_whitespace():
    node = parse('(x "a (b) c\nd")')
    assert node[1] == "a (b) c\nd"


def test_quoted_flag_is_not_a_flag():
    assert not has_flag(parse('(via "blind")'), "blind")


@pytest.mark.parametrize("text", ["", "(a", "(a))", "(a) (b)", "atom"])
def test_malformed(text):
    with pytest.raises(SExprError):
        parse(text)


def test_deep_nesting_does_not_recurse():
    depth = 5000
    node = parse("(" * depth + ")" * depth)
    for _ in range(depth - 1):
        node = node[0]
    assert node == []
