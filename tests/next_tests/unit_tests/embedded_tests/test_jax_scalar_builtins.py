# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Scalar math builtins evaluated on JAX tracers, as happens for scalar arguments under jit."""

import numpy as np
import pytest

from gt4py.next.ffront import fbuiltins

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.mark.parametrize(
    "name, reference",
    [
        ("cos", np.cos),
        ("sqrt", np.sqrt),
        ("exp", np.exp),
        ("floor", np.floor),
        ("isfinite", np.isfinite),
    ],
)
def test_math_builtin_on_traced_scalar(name, reference):
    builtin = getattr(fbuiltins, name)

    @jax.jit
    def f(x):
        return builtin(x)

    x = np.float32(0.75)
    result = f(x)
    assert isinstance(result, jax.Array)
    np.testing.assert_allclose(np.asarray(result), reference(x), rtol=1e-6)


def test_math_builtin_on_numpy_scalar_unchanged():
    result = fbuiltins.cos(np.float32(0.75))
    assert isinstance(result, np.float32)
    np.testing.assert_allclose(result, np.cos(np.float32(0.75)))


def test_number_builtins_on_traced_scalar():
    @jax.jit
    def f(x):
        return fbuiltins.abs(x) + fbuiltins.neg(x)

    assert float(f(np.float32(-2.0))) == 4.0
