"""Call a child with the call-time keywords its signature accepts.

A composite whose children differ in kind (a generator with dropout, one without) passes a
call-time keyword such as ``rngs`` only to the children that declare it. A layer without
randomness declares no ``rngs``, as flax's own layers do, and is called without one.

This is the rule ``flax.nnx.Sequential`` applies to its layers (flax commit 7a24e6a7, released in
0.12.11: "filtering ``kwargs`` per layer so every layer receives the keyword arguments its
signature accepts"). flax implements it in the private ``flax.nnx.helpers._filter_kwargs`` and
exports no public equivalent, so this module implements the same rule for artifex's composites.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


_POSITIONAL = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
_KEYWORD = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _filled_by_positionals(parameters: list[inspect.Parameter], num_positional: int) -> set[str]:
    """Positional-or-keyword parameters the positional arguments fill.

    A positional-only parameter is filled too, but its name stays free as a ``**kwargs`` key.
    """
    slots = [p for p in parameters if p.kind in _POSITIONAL][:num_positional]
    return {p.name for p in slots if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD}


def _takes_variadic_keywords(parameters: list[inspect.Parameter]) -> bool:
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)


def _keyword_names(parameters: list[inspect.Parameter]) -> set[str]:
    return {p.name for p in parameters if p.kind in _KEYWORD}


def accepted_keywords(
    fn: Callable[..., Any], num_positional: int, keywords: dict[str, Any]
) -> dict[str, Any]:
    """The keywords ``fn`` accepts when called with ``num_positional`` positional arguments.

    A keyword is accepted when ``fn`` declares it as keyword-only, or as positional-or-keyword
    beyond the parameters the positional arguments fill, or when ``fn`` takes ``**kwargs``. A
    parameter a positional argument already fills is never passed again.

    Args:
        fn: The callable, a module or a function.
        num_positional: How many positional arguments the call passes.
        keywords: The call-time keywords on offer.

    Returns:
        The subset of ``keywords`` to pass.
    """
    parameters = list(inspect.signature(fn).parameters.values())
    filled = _filled_by_positionals(parameters, num_positional)
    if _takes_variadic_keywords(parameters):
        return {name: value for name, value in keywords.items() if name not in filled}
    declared = _keyword_names(parameters) - filled
    return {name: value for name, value in keywords.items() if name in declared}


def call_with_accepted_keywords(fn: Callable[..., Any], *args: Any, **keywords: Any) -> Any:
    """Call ``fn`` with ``args`` and the subset of ``keywords`` its signature accepts.

    Args:
        fn: The child to call.
        *args: Positional arguments, passed as given.
        **keywords: Call-time keywords on offer, such as ``rngs``.

    Returns:
        What ``fn`` returns.
    """
    return fn(*args, **accepted_keywords(fn, len(args), keywords))
