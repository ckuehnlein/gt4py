# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import types

from gt4py.next import backend, common, constructors, utils
from gt4py.next.embedded import operators as embedded_operators
from gt4py.next.ffront import stages as ffront_stages
from gt4py.next.otf import arguments, definitions, stages


try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None


if jax:

    def _make_callable(
        definition: types.FunctionType, offset_provider: common.OffsetProvider
    ) -> types.FunctionType:
        @jax.jit
        def _pure_callable(input_args, out):
            embedded_operators.field_operator_call(
                embedded_operators.EmbeddedOperator(definition),
                input_args,
                {"out": out, "offset_provider": offset_provider},
            )
            return out

        def _callable(*args, **kwargs):
            input_args = args[:-1]
            out = args[-1]
            result_out = _pure_callable(input_args, out)

            # Copy the concrete result arrays back into the *original* output
            # fields so the caller sees the updated data.
            flat_orig = utils.flatten_nested_tuple((out,)) if isinstance(out, tuple) else (out,)
            flat_result = (
                utils.flatten_nested_tuple((result_out,))
                if isinstance(result_out, tuple)
                else (result_out,)
            )
            for orig, result in zip(flat_orig, flat_result):
                if hasattr(orig, "_ndarray"):
                    object.__setattr__(orig, "_ndarray", result.ndarray)

        return _callable

    @dataclasses.dataclass(frozen=True)
    class JaxJitBackend(backend.Backend):
        executor: types.FunctionType = dataclasses.field(init=False)
        allocator: constructors.Allocator = dataclasses.field(init=False)
        transforms: backend.Transforms = dataclasses.field(init=False)

        def __post_init__(self):
            object.__setattr__(self, "executor", lambda inp: None)
            object.__setattr__(self, "allocator", jnp)
            object.__setattr__(self, "transforms", backend.DEFAULT_TRANSFORMS)

        def compile(
            self, program: definitions.IRDefinitionT, compile_time_args: arguments.CompileTimeArgs
        ) -> stages.ExecutableProgram:
            if not isinstance(program, ffront_stages.DSLFieldOperatorDef):
                raise NotImplementedError(
                    f"JaxJitBackend can only be used from DSLFieldOperatorDef, got {type(program)}"
                )
            return _make_callable(program.definition, compile_time_args.offset_provider)


jax_jit = JaxJitBackend("jax.jit")
