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
    import torch
except ImportError:
    torch = None


if torch:

    def _make_callable(
        definition: types.FunctionType, offset_provider: common.OffsetProvider
    ) -> types.FunctionType:
        @torch.compile
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
            # The embedded machinery is not fully Dynamo-traceable: reads of
            # contextvars (embedded_context) force graph breaks, and resuming
            # a frame that holds gt4py's frozen metadata dataclasses (Domain,
            # UnitRange) as locals fails with "can't reconstruct arbitrary
            # frozen dataclass instances". With suppress_errors, Dynamo falls
            # back to eager for exactly those frames and compiles the tensor
            # compute in the rest — the intended semantics for this backend.
            # Scoped via config.patch so the process-global default of user
            # code outside this backend is untouched.
            with torch._dynamo.config.patch(suppress_errors=True):
                result_out = _pure_callable(input_args, out)

            # Copy the concrete result tensors back into the *original* output
            # fields so the caller sees the updated data. Mirrors the JAX
            # JaxJitBackend pattern — TorchTensorField is a frozen dataclass,
            # so ``object.__setattr__`` is required.
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

    # --- CUDA graphs for whole calls --------------------------------------------
    #
    # torch has no counterpart of ``jax.jit`` for gt4py's embedded execution:
    # TorchDynamo cannot trace the Field machinery and drops forward-mode
    # tangents. What removes the per-call Python and kernel-launch overhead on
    # a GPU is CUDA graph capture of a whole call: the kernels of one call
    # (a field operator, a TL/AD wrapper, a model time step) are recorded once
    # per input signature and replayed as one launch. ``graphed(fn)`` wraps any
    # callable on tensors / gt4py Fields (pytrees) that way, with the mode
    # chosen per call: plain, forward AD (``forward_ad`` dual inputs, static
    # primal + tangent buffers) or autograd (``requires_grad`` inputs, via
    # ``torch.cuda.make_graphed_callables``). The application chooses the
    # capture boundary, which should be the largest region with static
    # shapes and no host synchronisation. Calls with non-CUDA or
    # functorch-wrapped inputs, and nested graphed calls, run ``fn`` directly.

    _GRAPHED_ACTIVE = False

    class _CapturedCall:
        def __init__(self, fn, spec, flat, tensor_idx, mode, tensors):
            import torch.autograd.forward_ad as fwAD
            import torch.utils._pytree as pytree

            self.spec, self.flat, self.tensor_idx, self.mode = spec, list(flat), tensor_idx, mode
            self.out_spec = None

            def run(inputs):
                fl = list(self.flat)
                for i, t in zip(self.tensor_idx, inputs):
                    fl[i] = t
                a, kw = pytree.tree_unflatten(fl, self.spec)
                out, self.out_spec = pytree.tree_flatten(fn(*a, **kw))
                return list(out)

            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            if mode == "grad":
                sample = tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in tensors)
                self.graphed = torch.cuda.make_graphed_callables(
                    lambda *ts: tuple(run(list(ts))), sample
                )
                return
            if mode == "dual":
                duals = [fwAD.unpack_dual(t) for t in tensors]
                self.static_p = [d.primal.detach().clone() for d in duals]
                self.static_t = [
                    torch.zeros_like(d.primal) if d.tangent is None else d.tangent.detach().clone()
                    for d in duals
                ]

                def capture():
                    outs = run([fwAD.make_dual(p, t) for p, t in zip(self.static_p, self.static_t)])
                    un = [fwAD.unpack_dual(o) for o in outs]
                    return (
                        [u.primal for u in un],
                        [torch.zeros_like(u.primal) if u.tangent is None else u.tangent for u in un],
                    )

            else:
                self.static_p = [t.detach().clone() for t in tensors]

                def capture():
                    return run(self.static_p), None

            with torch.cuda.stream(side):
                for _ in range(2):  # warm-up (lazy kernel compilation, allocator) before capture
                    capture()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out_p, self.static_out_t = capture()

        def __call__(self, tensors):
            import torch.autograd.forward_ad as fwAD
            import torch.utils._pytree as pytree

            if self.mode == "grad":
                outs = list(self.graphed(*tensors))
            else:
                if self.mode == "dual":
                    for bp, bt, t in zip(self.static_p, self.static_t, tensors):
                        d = fwAD.unpack_dual(t)
                        bp.copy_(d.primal)
                        bt.copy_(torch.zeros_like(bp) if d.tangent is None else d.tangent)
                else:
                    for b, t in zip(self.static_p, tensors):
                        b.copy_(t)
                self.graph.replay()
                outs = [o.clone() for o in self.static_out_p]  # replay overwrites the static outputs
                if self.mode == "dual":
                    outs = [fwAD.make_dual(o, t.clone()) for o, t in zip(outs, self.static_out_t)]
            return pytree.tree_unflatten(outs, self.out_spec)

    class _Graphed:
        def __init__(self, fn):
            self.fn = fn
            self.cache: dict = {}

        def __call__(self, *args, **kwargs):
            global _GRAPHED_ACTIVE
            import torch.autograd.forward_ad as fwAD
            import torch.utils._pytree as pytree

            if (
                _GRAPHED_ACTIVE
                or not torch.cuda.is_available()
                or torch.cuda.is_current_stream_capturing()
            ):
                return self.fn(*args, **kwargs)
            flat, spec = pytree.tree_flatten((args, kwargs))
            tensor_idx = [i for i, x in enumerate(flat) if isinstance(x, torch.Tensor)]
            tensors = [flat[i] for i in tensor_idx]
            if not tensors or not all(t.is_cuda for t in tensors):
                return self.fn(*args, **kwargs)
            if any(torch._C._functorch.is_functorch_wrapped_tensor(t) for t in tensors):
                return self.fn(*args, **kwargs)
            has_dual = fwAD._current_level >= 0 and any(
                fwAD.unpack_dual(t).tangent is not None for t in tensors
            )
            needs_grad = torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
            mode = "dual" if has_dual else ("grad" if needs_grad else "plain")
            consts = tuple(
                (i, flat[i]) if _hashable(flat[i]) else (i, repr(flat[i]))
                for i in range(len(flat))
                if i not in tensor_idx
            )
            key = (
                mode,
                repr(spec),
                tuple(tensor_idx),
                tuple((tuple(t.shape), t.dtype, str(t.device)) for t in tensors),
                consts,
            )
            captured = self.cache.get(key)
            _GRAPHED_ACTIVE = True
            try:
                if captured is None:
                    captured = self.cache[key] = _CapturedCall(
                        self.fn, spec, flat, tensor_idx, mode, tensors
                    )
                return captured(tensors)
            finally:
                _GRAPHED_ACTIVE = False

    def _hashable(x) -> bool:
        try:
            hash(x)
        except TypeError:
            return False
        return True

    def graphed(fn):
        """Replay ``fn`` as a CUDA graph per input signature (see the note above).

        ``fn`` takes and returns tensors, gt4py Fields or pytrees of them;
        other arguments are treated as constants of the captured graph.
        Shapes, dtypes, devices, dual / requires_grad state and the constants
        form the cache key. On CPU inputs ``fn`` is simply called.
        """
        return _Graphed(fn)

    @dataclasses.dataclass(frozen=True)
    class TorchCompileBackend(backend.Backend):
        executor: types.FunctionType = dataclasses.field(init=False)
        allocator: constructors.Allocator = dataclasses.field(init=False)
        transforms: backend.Transforms = dataclasses.field(init=False)

        def __post_init__(self):
            object.__setattr__(self, "executor", lambda inp: None)
            object.__setattr__(self, "allocator", torch)
            object.__setattr__(self, "transforms", backend.DEFAULT_TRANSFORMS)

        def compile(
            self, program: definitions.IRDefinitionT, compile_time_args: arguments.CompileTimeArgs
        ) -> stages.ExecutableProgram:
            if not isinstance(program, ffront_stages.DSLFieldOperatorDef):
                raise NotImplementedError(
                    f"TorchCompileBackend can only be used from DSLFieldOperatorDef, "
                    f"got {type(program)}"
                )
            return _make_callable(program.definition, compile_time_args.offset_provider)


torch_compile = TorchCompileBackend("torch.compile")

if torch is None:

    def graphed(fn):  # type: ignore[misc]
        return fn
