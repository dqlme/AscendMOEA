"""Tests for third-party component registration."""

import pytest

from ascendmoea.registry import ComponentRegistry


def test_registry_supports_decorator_and_factory() -> None:
    """Register and construct a component through the public protocol."""
    registry = ComponentRegistry("example")

    @registry.register(name="Demo")
    class Demo:
        """Minimal registry test component."""

        def __init__(self, value: int) -> None:
            """Store the supplied value."""
            self.value = value

    assert registry.names() == ("Demo",)
    assert registry.create("Demo", value=3).value == 3


def test_registry_rejects_accidental_replacement() -> None:
    """Prevent an extension from silently shadowing an existing name."""
    registry = ComponentRegistry("example")

    class First:
        """First test component."""

    class Second:
        """Second test component."""

    registry.register(First, name="same")
    with pytest.raises(KeyError):
        registry.register(Second, name="same")
