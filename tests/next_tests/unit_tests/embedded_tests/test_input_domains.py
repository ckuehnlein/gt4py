# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import gt4py.next as gtx
from gt4py.next.embedded.input_domains import accessed_input_domains, restrict_to_bounds
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront.fbuiltins import int32


IDim = gtx.Dimension("IDim")
KDim = gtx.Dimension("KDim", kind=gtx.DimensionKind.VERTICAL)
IKField = gtx.Field[gtx.Dims[IDim, KDim], float]


@gtx.field_operator
def _centred_difference(a: IKField, b: IKField) -> IKField:
    return a(IDim + 1) - a(IDim - 1) + b


@gtx.field_operator
def _boundary_row(a: IKField, lz: int32) -> IKField:
    return concat_where(KDim == lz - 1, a(KDim - 2), a)


@gtx.field_operator
def _nested(a: IKField, b: IKField, c: float) -> IKField:
    return _centred_difference(a, b) * c


def test_shifted_input_is_widened():
    domains = accessed_input_domains(_centred_difference, gtx.domain({IDim: (2, 10), KDim: (0, 5)}))
    assert domains == {
        "a": {IDim: (1, 11), KDim: (0, 5)},
        "b": {IDim: (2, 10), KDim: (0, 5)},
    }


def test_scalar_decides_concat_where_domain():
    domains = accessed_input_domains(
        _boundary_row, gtx.domain({IDim: (0, 4), KDim: (0, 8)}), scalar_args={"lz": 8}
    )
    # row 7 reads a(KDim - 2) -> rows 5; rows 0..6 read a itself
    assert domains == {"a": {IDim: (0, 4), KDim: (0, 7)}}


def test_callee_operators_are_inlined():
    domains = accessed_input_domains(_nested, gtx.domain({IDim: (2, 10), KDim: (0, 5)}))
    assert domains["a"] == {IDim: (1, 11), KDim: (0, 5)}
    assert domains["b"] == {IDim: (2, 10), KDim: (0, 5)}
    assert "c" in domains  # a float scalar is kept, with no field bounds


def test_restrict_to_bounds_clips_to_the_field():
    field = gtx.as_field([IDim, KDim], np.arange(20.0).reshape(4, 5))
    restricted = restrict_to_bounds(field, {IDim: (-3, 2), KDim: (1, None)})
    assert restricted.domain == gtx.domain({IDim: (0, 2), KDim: (1, 5)})
    np.testing.assert_array_equal(restricted.asnumpy(), field.asnumpy()[0:2, 1:5])
    assert restrict_to_bounds(field, {IDim: (None, None), KDim: (0, 5)}) is field
