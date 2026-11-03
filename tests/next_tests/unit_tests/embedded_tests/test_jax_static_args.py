# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.embedded.operators import _is_jax_static_arg
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront.fbuiltins import int32


jnp = pytest.importorskip("jax.numpy")

KDim = gtx.Dimension("KDim", kind=gtx.DimensionKind.VERTICAL)
KField = gtx.Field[gtx.Dims[KDim], float]


@gtx.field_operator
def _top_row(a: KField, lz: int32) -> KField:
    return concat_where(KDim == lz - 1, a(KDim - 1), a)


@pytest.mark.parametrize(
    "arg, static",
    [
        (3, True),
        (True, True),
        (np.int32(3), True),
        ((1, (2, 3)), True),
        ((), False),
        (1.5, False),
        (np.float64(1.5), False),
        ("x", False),
    ],
)
def test_is_jax_static_arg(arg, static):
    assert _is_jax_static_arg(arg) is static


def test_integer_argument_stays_static_under_jit():
    """An int compared with a dimension index must not become a tracer when gt4py jits."""
    data = np.arange(6.0)
    a = gtx.as_field([KDim], jnp.asarray(data))
    res = _top_row(a, 6, offset_provider={})
    expected = data.copy()
    expected[5] = data[4]
    np.testing.assert_array_equal(np.asarray(res.ndarray), expected[:6])
    # a different value is a different jit specialization, not a stale one
    res = _top_row(a, 4, offset_provider={})
    expected = data.copy()
    expected[3] = data[2]
    np.testing.assert_array_equal(np.asarray(res.ndarray), expected[: res.ndarray.shape[0]])
