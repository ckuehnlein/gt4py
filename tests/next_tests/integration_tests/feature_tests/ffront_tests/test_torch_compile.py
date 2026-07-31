# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PyTorch ``torch.compile`` backend.

Parallels ``test_jax_jit.py`` on the JAX side: field_operator-level smoke
of basic arithmetic, Cartesian shifts, and call-time caching. The
torch.func autodiff path is exercised by the cloudsc2 dwarf rather than
here.

Note: We use the torch.compile backend in its ``out=`` writeback mode
(via ``cases.verify``). The inline (no-``out=``) eager path is the
default for autodiff and lives in ``embedded.operators.field_operator_call``
— Phase C of the pytorch-embedded design.
"""

import numpy as np
import pytest

import gt4py.next as gtx

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, simple_cartesian_grid


try:
    from gt4py.next.program_processors.runners.torch_compile import torch_compile
    import torch
except ImportError:
    torch_compile = None
    torch = None


pytestmark = pytest.mark.skipif(torch_compile is None, reason="PyTorch is not installed")


@pytest.fixture
def torch_cartesian_case():
    return cases.Case.from_cartesian_grid_descriptor(
        simple_cartesian_grid(),
        backend=torch_compile,
        allocator=torch,
    )


# ---------------------------------------------------------------------------
# Basic arithmetic
# ---------------------------------------------------------------------------


def test_copy(torch_cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKField) -> cases.IJKField:
        return a

    cases.verify_with_default_data(torch_cartesian_case, testee, ref=lambda a: a)


def test_addition(torch_cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKField, b: cases.IJKField) -> cases.IJKField:
        return a + b

    cases.verify_with_default_data(torch_cartesian_case, testee, ref=lambda a, b: a + b)


def test_arithmetic(torch_cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKFloatField, b: cases.IJKFloatField) -> cases.IJKFloatField:
        return a * b - b / a

    cases.verify_with_default_data(torch_cartesian_case, testee, ref=lambda a, b: a * b - b / a)


# ---------------------------------------------------------------------------
# Cartesian shifts
# ---------------------------------------------------------------------------


def test_cartesian_shift(torch_cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKField) -> cases.IJKField:
        return a(IDim + 1)

    a = cases.allocate(torch_cartesian_case, testee, "a").extend({IDim: (0, 1)})()
    out = cases.allocate(torch_cartesian_case, testee, cases.RETURN)()

    cases.verify(torch_cartesian_case, testee, a, out=out, ref=a.asnumpy()[1:])


# ---------------------------------------------------------------------------
# Compile caching — same compiled function reused on second call
# ---------------------------------------------------------------------------


def test_compile_caching(torch_cartesian_case):
    """Verify that repeated calls reuse the cached compiled function."""

    @gtx.field_operator
    def testee(a: cases.IJKField) -> cases.IJKField:
        return a + a

    a = cases.allocate(torch_cartesian_case, testee, "a")()
    out = cases.allocate(torch_cartesian_case, testee, cases.RETURN)()

    cases.run(torch_cartesian_case, testee, a, out=out)
    first_result = out.asnumpy().copy()

    # Second call should hit the cache.
    cases.run(torch_cartesian_case, testee, a, out=out)
    second_result = out.asnumpy()

    assert np.allclose(first_result, second_result)
