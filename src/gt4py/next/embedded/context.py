# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Generator
from typing import Any, TypeVar, overload

import gt4py.eve as eve
import gt4py.next.common as common
import gt4py.next.embedded as gtx_embedded
import gt4py.next.errors.exceptions as exceptions


_closure_column_range: contextvars.ContextVar[common.NamedRange] = contextvars.ContextVar(
    "_column_range"
)
_offset_provider: contextvars.ContextVar[common.OffsetProvider] = contextvars.ContextVar(
    "_offset_provider"
)
_scan_strategy: contextvars.ContextVar[str | None] = contextvars.ContextVar("_scan_strategy")


_T = TypeVar("_T")


_NO_DEFAULT_SENTINEL: Any = object()


@overload
def get_closure_column_range() -> common.NamedRange: ...


@overload
def get_closure_column_range(default: _T) -> common.NamedRange | _T: ...


def get_closure_column_range(default: _T = _NO_DEFAULT_SENTINEL) -> common.NamedRange | _T:
    """Column range used in 'column mode' in the current embedded iterator closure execution context."""
    result = _closure_column_range.get(default)
    if result is _NO_DEFAULT_SENTINEL:
        raise exceptions.EmbeddedExecutionError(
            "No column range set in the current embedded iterator closure execution context."
        )
    return result


@overload
def get_offset_provider() -> common.OffsetProvider: ...


@overload
def get_offset_provider(default: _T) -> common.OffsetProvider | _T: ...


def get_offset_provider(default: _T = _NO_DEFAULT_SENTINEL) -> common.OffsetProvider | _T:
    """Offset provider used in the current embedded iterator closure execution context."""
    result = _offset_provider.get(default)
    if result is _NO_DEFAULT_SENTINEL:
        raise exceptions.EmbeddedExecutionError(
            "No offset provider set in the current embedded iterator closure execution context."
        )
    return result


def get_scan_strategy(default: _T = None) -> str | None | _T:
    """Embedded scan-execution strategy selected in the current context, if any."""
    return _scan_strategy.get(default)


@contextlib.contextmanager
def update(
    *,
    closure_column_range: common.NamedRange | eve.NothingType = eve.NOTHING,
    offset_provider: common.OffsetProvider | eve.NothingType = eve.NOTHING,
    scan_strategy: str | None | eve.NothingType = eve.NOTHING,
) -> Generator[None, None, None]:
    """Context handler updating the current embedded context with the provided values."""

    closure_token, offset_provider_token, scan_strategy_token = None, None, None
    if closure_column_range is not eve.NOTHING:
        assert not isinstance(closure_column_range, eve.NothingType)
        closure_token = gtx_embedded.context._closure_column_range.set(closure_column_range)
    if offset_provider is not eve.NOTHING:
        assert not isinstance(offset_provider, eve.NothingType)
        offset_provider_token = gtx_embedded.context._offset_provider.set(offset_provider)
    if scan_strategy is not eve.NOTHING:
        assert not isinstance(scan_strategy, eve.NothingType)
        scan_strategy_token = gtx_embedded.context._scan_strategy.set(scan_strategy)

    try:
        yield None
    finally:
        if closure_column_range is not eve.NOTHING:
            assert closure_token is not None
            gtx_embedded.context._closure_column_range.reset(closure_token)
        if offset_provider is not eve.NOTHING:
            assert offset_provider_token is not None
            gtx_embedded.context._offset_provider.reset(offset_provider_token)
        if scan_strategy is not eve.NOTHING:
            assert scan_strategy_token is not None
            gtx_embedded.context._scan_strategy.reset(scan_strategy_token)


@contextlib.contextmanager
def scan_strategy(name: str | None) -> Generator[None, None, None]:
    """Select the embedded scan-execution strategy for calls inside the block.

    Overrides any ``strategy=`` given at ``@scan_operator`` decoration
    (deprecated). Only embedded execution consults this; compiled backends
    (gtfn, dace, ...) ignore it.
    """
    with update(scan_strategy=name):
        yield None


# Module-level sentinel: allocating it per call (`object()`) breaks
# TorchDynamo tracing of the embedded machinery under torch.compile
# (bare `object` is an untraceable builtin, forcing a graph break).
_WITHIN_VALID_CONTEXT_SENTINEL: Any = object()


def within_valid_context() -> bool:
    return _offset_provider.get(_WITHIN_VALID_CONTEXT_SENTINEL) is not _WITHIN_VALID_CONTEXT_SENTINEL
