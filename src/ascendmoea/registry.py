"""Typed registries for algorithms, problems, and user extensions."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import Generic, TypeVar, overload

ComponentT = TypeVar("ComponentT")


class ComponentRegistry(MutableMapping[str, type[ComponentT]], Generic[ComponentT]):
    """Store constructible component classes under stable public names.

    The registry implements ``MutableMapping`` so existing dictionary-style
    lookup remains available. The :meth:`register` method can also be used as
    a decorator for third-party extensions.

    Args:
        kind: Human-readable component kind used in validation errors.
        components: Optional component classes to register immediately.
    """

    def __init__(
        self,
        kind: str,
        components: Iterator[type[ComponentT]] | tuple[type[ComponentT], ...] = (),
    ) -> None:
        """Create a registry and register the supplied component classes."""
        self.kind = str(kind)
        self._components: dict[str, type[ComponentT]] = {}
        for component in components:
            self.register(component)

    def __getitem__(self, name: str) -> type[ComponentT]:
        """Return the component registered under ``name``."""
        try:
            return self._components[name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<empty>"
            raise KeyError(
                f"Unknown {self.kind} {name!r}. Available {self.kind}s: {available}"
            ) from exc

    def __setitem__(self, name: str, component: type[ComponentT]) -> None:
        """Register ``component`` under an explicit mapping key."""
        self._validate_component(component)
        self._components[str(name)] = component

    def __delitem__(self, name: str) -> None:
        """Remove a registered component."""
        del self._components[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate over component names in insertion order."""
        return iter(self._components)

    def __len__(self) -> int:
        """Return the number of registered components."""
        return len(self._components)

    @overload
    def register(
        self,
        component: type[ComponentT],
        *,
        name: str | None = None,
        replace: bool = False,
    ) -> type[ComponentT]: ...

    @overload
    def register(
        self,
        component: None = None,
        *,
        name: str | None = None,
        replace: bool = False,
    ): ...

    def register(
        self,
        component: type[ComponentT] | None = None,
        *,
        name: str | None = None,
        replace: bool = False,
    ):
        """Register a component class or return a registration decorator.

        Args:
            component: Class to register. Omit it when using decorator syntax.
            name: Public registry name. Defaults to ``component.name`` and then
                to the class name.
            replace: Allow replacement of an existing entry with the same name.

        Returns:
            The registered class, or a decorator that registers a class.
        """

        def decorator(candidate: type[ComponentT]) -> type[ComponentT]:
            """Validate and store one decorated component class."""
            self._validate_component(candidate)
            public_name = str(name or getattr(candidate, "name", candidate.__name__))
            if not public_name:
                raise ValueError(f"A {self.kind} registry name cannot be empty")
            current = self._components.get(public_name)
            if current is not None and current is not candidate and not replace:
                raise KeyError(f"{self.kind.capitalize()} {public_name!r} is already registered")
            self._components[public_name] = candidate
            return candidate

        if component is None:
            return decorator
        return decorator(component)

    def create(self, name: str, /, **kwargs) -> ComponentT:
        """Instantiate a registered component by name.

        Args:
            name: Registry key.
            **kwargs: Keyword arguments forwarded to the component constructor.

        Returns:
            A new component instance.
        """
        return self[name](**kwargs)

    def names(self) -> tuple[str, ...]:
        """Return registered names sorted for deterministic presentation."""
        return tuple(sorted(self._components))

    def _validate_component(self, component: type[ComponentT]) -> None:
        """Reject registry values that are not classes."""
        if not isinstance(component, type):
            raise TypeError(
                f"A registered {self.kind} must be a class, got {type(component).__name__}"
            )
