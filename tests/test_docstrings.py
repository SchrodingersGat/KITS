import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent.parent

_spec = importlib.util.spec_from_file_location("check_docstrings", ROOT / "tools" / "check_docstrings.py")
check_docstrings = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_docstrings)


def _problems(source: str) -> list[str]:
    return list(check_docstrings.check_source(source, "x.py"))


def test_documented_args_parses_google_sections():
    doc = """Summary.

    Args:
        a: First.
        b (int): Second, with a type.
            Continuation: not an argument.
        *args: Extra positionals.
        **kwargs: Extra keywords.

    Returns:
        not_an_arg: Something.
    """
    assert check_docstrings.documented_args(doc) == {"a", "b", "args", "kwargs"}


def test_missing_docstrings_are_reported_including_private_and_nested():
    source = '''"""Module."""


class A:
    def _private(self):
        pass


def outer():
    """Outer."""

    def inner():
        pass
'''
    problems = _problems(source)
    assert any("class 'A'" in p for p in problems)
    assert any("'_private'" in p for p in problems)
    assert any("'inner'" in p for p in problems)
    assert not any("'outer'" in p for p in problems)


def test_undocumented_arguments_are_reported():
    source = '''"""Module."""


def f(self, a, b, *rest, c=1, **opts):
    """Summary.

    Args:
        a: Documented.
        rest: Documented.
    """
'''
    (problem,) = _problems(source)
    assert "b, c, opts" in problem


def test_fully_documented_source_passes():
    source = '''"""Module."""


class A:
    """A class."""

    def method(self, x):
        """Do it.

        Args:
            x: The thing.
        """

    @property
    def value(self):
        """The value."""
'''
    assert _problems(source) == []


def test_codebase_is_fully_documented():
    assert check_docstrings.main([str(ROOT / "src"), str(ROOT / "tools")]) == 0
