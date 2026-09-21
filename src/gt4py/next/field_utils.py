# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
from types import ModuleType

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.eve.extended_typing import NestedTuple
from gt4py.next import common, named_collections, utils
from gt4py.next.type_system import type_specifications as ts, type_translation


try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None


def _xp_dtype(xp: ModuleType, np_scalar_type: type):
    """Map a numpy scalar dtype to the matching dtype of the array namespace ``xp``.

    Needed because ``torch.dtype`` is not a constructor like ``np.dtype`` /
    ``jnp.dtype`` / ``cp.dtype`` and ``torch.float64(...)`` does not exist.
    """
    if torch is not None and xp is torch:
        # Map numpy scalar dtype to torch dtype via the canonical name.
        # np.float64 -> "float64" -> torch.float64.
        return getattr(torch, np.dtype(np_scalar_type).name)
    return xp.dtype(np_scalar_type)


@utils.tree_map
def asnumpy(field: common.Field | np.ndarray) -> np.ndarray:
    return field.asnumpy() if isinstance(field, common.Field) else field


def field_from_typespec(
    type_: ts.CollectionTypeSpec | ts.ScalarType, domain: common.Domain, xp: ModuleType
) -> common.MutableField | tuple[common.MutableField | tuple, ...]:
    """
    Allocate a field or (arbitrarily nested) tuple(s) of fields.

    The tuple structure and dtype is taken from a type_specifications.DataType,
    which is either ScalarType or a CollectionTypeSpec of ScalarType (possibly nested).

    >>> field_from_typespec(
    ...     ts.ScalarType(kind=ts.ScalarKind.INT32), common.domain({common.Dimension("I"): 1}), np
    ... )  # doctest: +ELLIPSIS
    NumPyArrayField(... dtype=int32...)
    >>> field_from_typespec(
    ...     ts.TupleType(
    ...         types=[
    ...             ts.ScalarType(kind=ts.ScalarKind.INT32),
    ...             ts.ScalarType(kind=ts.ScalarKind.FLOAT32),
    ...         ]
    ...     ),
    ...     common.domain({common.Dimension("I"): 1}),
    ...     np,
    ... )  # doctest: +ELLIPSIS
    (NumPyArrayField(... dtype=int32...), NumPyArrayField(... dtype=float32...))
    """

    def _constructor(
        type_: ts.CollectionTypeSpec,
        elems: NestedTuple[common.MutableField],
    ) -> named_collections.NamedCollection:
        if isinstance(type_, ts.NamedCollectionType):
            return named_collections.make_named_collection_constructor_from_type_spec(type_)(elems)
        return tuple(elems)

    @utils.tree_map(
        collection_type=ts.COLLECTION_TYPE_SPECS,
        result_collection_constructor=_constructor,
    )
    def impl(type_: ts.ScalarType) -> common.MutableField:
        np_scalar = type_translation.as_dtype(type_).scalar_type
        res = common._field(
            xp.empty(domain.shape, dtype=_xp_dtype(xp, np_scalar)),
            domain=domain,
        )
        assert isinstance(res, common.MutableField)
        return res

    return impl(type_)


def get_array_ns(
    *args: core_defs.Scalar | common.Field | tuple[core_defs.Scalar | common.Field | tuple, ...],
) -> ModuleType:
    for arg in utils.flatten_nested_tuple(args):
        if hasattr(arg, "array_ns"):
            return arg.array_ns
    return np


def device_context(
    *args: core_defs.Scalar | common.Field | tuple[core_defs.Scalar | common.Field | tuple, ...],
) -> contextlib.AbstractContextManager:
    """Context under which fresh arrays are allocated on the device of ``args``.

    ``torch.empty`` & co. allocate on torch's *default* device (CPU), not
    on the device of the tensors an operator receives, so embedded execution
    on GPU-resident torch inputs would mix CUDA and CPU tensors (e.g. the
    scan accumulator from :func:`field_from_typespec`). Entering the
    ``torch.device`` of the first torch-backed field makes all factory
    calls follow the inputs. Other array namespaces (numpy, cupy, jax)
    already allocate where their inputs live; for those (and if no torch
    field is present) this is a no-op.
    """
    if torch is not None:
        for arg in utils.flatten_nested_tuple(args):
            data = getattr(arg, "ndarray", None)
            if isinstance(data, torch.Tensor):
                return torch.device(data.device)
    return contextlib.nullcontext()


def verify_device_field_type(field: common.Field, device: core_defs.DeviceType) -> bool:
    """Check if `field` is suitable for `device`."""
    if not (array_ns := getattr(field, "array_ns", False)):
        return False  # not a NDArrayField
    if device in [core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM]:
        assert core_defs.CUPY_DEVICE_TYPE is not None
        # TODO(havogt): generalize to other array libraries
        return device == core_defs.CUPY_DEVICE_TYPE and array_ns == cp
    else:
        return array_ns == np
