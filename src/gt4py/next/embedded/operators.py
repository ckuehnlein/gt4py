# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from types import ModuleType
from typing import Any, Callable, Generic, Optional, ParamSpec, Sequence, TypeVar

import numpy as np

from gt4py import eve
from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.next import common, errors, field_utils, named_collections, utils
from gt4py.next.embedded import common as embedded_common, context as embedded_context
from gt4py.next.field_utils import get_array_ns
from gt4py.next.ffront import fbuiltins
from gt4py.next.otf import arguments
from gt4py.next.type_system import type_specifications as ts, type_translation

try:
    import jax
except ImportError:
    jax: Optional[ModuleType] = None  # type: ignore[no-redef]

try:
    import torch
except ImportError:
    torch: Optional[ModuleType] = None  # type: ignore[no-redef]


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _field_float_kind(args: Sequence[Any]) -> Optional[ts.ScalarKind]:
    """Float precision carried by the field arguments of a scan call.

    Returns FLOAT64 if any field holds 64-bit floats, FLOAT32 if the float
    fields are all 32-bit, None if no float field is present. Nested tuple
    arguments are walked recursively.
    """
    found32 = False

    def visit(arg: Any) -> Optional[ts.ScalarKind]:
        nonlocal found32
        if isinstance(arg, tuple):
            for a in arg:
                if (kind := visit(a)) is not None:
                    return kind
            return None
        if isinstance(arg, common.Field):
            ndarray = arg.ndarray
            if getattr(ndarray, "weak_type", False):
                # JAX weak-typed arrays (built from Python scalars) don't pin
                # the precision — they follow whatever they combine with.
                return None
            name = str(getattr(ndarray, "dtype", ""))
            if name.endswith("float64"):
                return ts.ScalarKind.FLOAT64
            if name.endswith("float32"):
                found32 = True
        return None

    for arg in args:
        if (kind := visit(arg)) is not None:
            return kind
    return ts.ScalarKind.FLOAT32 if found32 else None


def _weak_init_type(
    init: xtyping.MaybeNestedInTuple[core_defs.ScalarT], args: Sequence[Any]
) -> ts.TypeSpec:
    """Type of the scan carry, with Python-float inits treated as weakly typed.

    ``type_translation.from_value`` maps a plain Python float to float64, which
    would silently upcast an otherwise float32 scan (the float64 carry wins
    every promotion). In embedded execution the carry precision should follow
    the data, so float64 scalars in the deduced init type are narrowed to
    float32 when the field arguments are uniformly 32-bit.
    """
    init_type = type_translation.from_value(init)
    if _field_float_kind(args) != ts.ScalarKind.FLOAT32:
        return init_type

    def narrow(t: ts.TypeSpec) -> ts.TypeSpec:
        if isinstance(t, ts.ScalarType) and t.kind == ts.ScalarKind.FLOAT64:
            return ts.ScalarType(kind=ts.ScalarKind.FLOAT32)
        if isinstance(t, ts.TupleType):
            return ts.TupleType(types=[narrow(x) for x in t.types])
        return t

    return narrow(init_type)


@dataclasses.dataclass(frozen=True)
class EmbeddedOperator(Generic[_R, _P]):
    fun: Callable[_P, _R]

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return self.fun(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class ScanOperator(EmbeddedOperator[xtyping.MaybeNestedInTuple[core_defs.ScalarT], _P]):
    forward: bool
    init: xtyping.MaybeNestedInTuple[core_defs.ScalarT]
    axis: common.Dimension

    def __call__(  # type: ignore[override]
        self,
        *args: common.Field | core_defs.Scalar,
        **kwargs: common.Field | core_defs.Scalar,
    ) -> (
        common.Field[Any, core_defs.ScalarT]
        | tuple[common.Field[Any, core_defs.ScalarT] | tuple, ...]
    ):
        scan_range = embedded_context.get_closure_column_range()
        assert self.axis == scan_range.dim
        scan_axis = scan_range.dim
        all_args = [*args, *kwargs.values()]
        domain_intersection = _intersect_scan_args(*all_args)
        non_scan_domain = common.Domain(*[nr for nr in domain_intersection if nr.dim != scan_axis])

        out_domain = common.Domain(
            *[scan_range if nr.dim == scan_axis else nr for nr in domain_intersection]
        )
        if scan_axis not in out_domain.dims:
            # even if the scan dimension is not in the input, we can scan over it
            out_domain = common.Domain(*out_domain, (scan_range))

        xp = get_array_ns(*(arguments.extract(arg) for arg in all_args))
        init_type = _weak_init_type(self.init, all_args)
        assert isinstance(init_type, ts.TupleType | ts.ScalarType | ts.NamedCollectionType)
        res = field_utils.field_from_typespec(init_type, out_domain, xp)

        def scan_loop(hpos: Sequence[common.NamedIndex]) -> None:
            acc: xtyping.MaybeNestedInTuple[core_defs.ScalarT] = self.init
            for k in scan_range.unit_range if self.forward else reversed(scan_range.unit_range):
                pos = (*hpos, common.NamedIndex(scan_axis, k))
                new_args = [_tuple_at(pos, arg) for arg in args]
                new_kwargs = {k: _tuple_at(pos, v) for k, v in kwargs.items()}
                acc = self.fun(acc, *new_args, **new_kwargs)  # type: ignore[arg-type] # need to express that the first argument is the same type as the return
                # convert custom NamedCollections to plain tuples for assignment
                acc_extracted = arguments.extract(acc)
                res_extracted = arguments.extract(res)
                assert xtyping.is_maybe_nested_in_tuple_of(acc_extracted, core_defs.Scalar)  # type: ignore[arg-type]  # Scalar is a Union
                assert xtyping.is_maybe_nested_in_tuple_of(res_extracted, common.MutableField)  # type: ignore[type-abstract]  # MutableField is abstract/generic
                _tuple_assign_value(pos, res_extracted, acc_extracted)

        if len(non_scan_domain) == 0:
            # if we don't have any dimension orthogonal to scan_axis, we need to do one scan_loop
            scan_loop(())
        else:
            for hpos in embedded_common.iterate_domain(non_scan_domain):
                scan_loop(hpos)

        return res


@dataclasses.dataclass(frozen=True)
class ScanOperatorVectorized(
    EmbeddedOperator[xtyping.MaybeNestedInTuple[core_defs.ScalarT], _P]
):
    """Scan operator whose body runs once per K-level on a horizontal slice.

    Where ``ScanOperator`` iterates the scan body once per (horizontal, K)
    position, this variant calls the body once per K with all horizontal
    points in parallel. The accumulator is a Field over the non-scan dims,
    not a scalar.
    """

    forward: bool
    init: xtyping.MaybeNestedInTuple[core_defs.ScalarT]
    axis: common.Dimension

    def __call__(  # type: ignore[override]
        self,
        *args: common.Field | core_defs.Scalar,
        **kwargs: common.Field | core_defs.Scalar,
    ) -> (
        common.Field[Any, core_defs.ScalarT]
        | tuple[common.Field[Any, core_defs.ScalarT] | tuple, ...]
    ):
        scan_range = embedded_context.get_closure_column_range()
        assert self.axis == scan_range.dim
        scan_axis = scan_range.dim
        all_args = [*args, *kwargs.values()]
        domain_intersection = _intersect_scan_args(*all_args)
        non_scan_domain = common.Domain(*[nr for nr in domain_intersection if nr.dim != scan_axis])

        out_domain = common.Domain(
            *[scan_range if nr.dim == scan_axis else nr for nr in domain_intersection]
        )
        if scan_axis not in out_domain.dims:
            # even if the scan dimension is not in the input, we can scan over it
            out_domain = common.Domain(*out_domain, (scan_range))

        xp = get_array_ns(*(arguments.extract(arg) for arg in all_args))
        init_type = _weak_init_type(self.init, all_args)
        assert isinstance(init_type, ts.TupleType | ts.ScalarType | ts.NamedCollectionType)
        # Allocate result / accumulator on the device of the inputs (torch).
        device_ctx = field_utils.device_context(*all_args)
        with device_ctx:
            res = field_utils.field_from_typespec(init_type, out_domain, xp)

        def scan_loop() -> None:
            acc: common.MutableField | tuple[common.MutableField | tuple, ...] = (
                field_utils.field_from_typespec(init_type, non_scan_domain, xp)
            )
            _tuple_assign_field(target=acc, source=self.init, domain=non_scan_domain)
            for k in scan_range.unit_range if self.forward else reversed(scan_range.unit_range):
                new_args = [
                    arg if core_defs.is_scalar_type(arg) else arg[common.NamedIndex(scan_axis, k)]
                    for arg in args
                ]
                new_kwargs = {kk: v[common.NamedIndex(scan_axis, k)] for kk, v in kwargs.items()}
                acc = self.fun(acc, *new_args, **new_kwargs)  # type: ignore[arg-type]

                k_slice = common.Domain(
                    *non_scan_domain,
                    common.named_range((scan_axis, (k, k + 1))),
                )
                broadcasted_acc = _tuple_broadcast_field(acc, (*non_scan_domain.dims, scan_axis))
                _tuple_assign_field(res, broadcasted_acc, k_slice)

        with device_ctx:
            scan_loop()

        return res


def _to_jax_field(field):
    """Materialise a Field on the JAX array namespace; pass through non-Fields."""
    from gt4py.next.embedded import nd_array_field

    if not isinstance(field, common.Field) or isinstance(field, nd_array_field.JaxArrayField):
        return field
    assert isinstance(field, nd_array_field.NumPyArrayField), (
        f"_to_jax_field: unsupported field type {type(field).__name__}"
    )
    return common._field(jax.numpy.asarray(field.ndarray), domain=field.domain)


def _to_numpy_field(field):
    """Materialise a Field on the NumPy array namespace; pass through non-Fields.

    No-op when the underlying array is a JAX tracer (i.e. we're inside an
    outer ``jax.jit`` trace): conversion to numpy is forbidden there and the
    caller will receive the still-JAX field, which is what they need.
    """
    if not isinstance(field, common.Field):
        return field
    if jax is not None and isinstance(field.ndarray, jax.core.Tracer):
        return field
    return common._field(np.asarray(field.ndarray), domain=field.domain)


def _transpose(f, dims):
    # Bypass common._field's singledispatch: under jax.jit tracing the
    # transposed array is a JAX tracer which the dispatch is not registered
    # for. Reconstruct via the concrete field class.
    @utils.tree_map
    def impl(f):
        xp = get_array_ns(f)
        arr = xp.transpose(f.ndarray, axes=[f.domain.dim_index(dim) for dim in dims])
        domain = common.Domain(*(f.domain[dim] for dim in dims))
        return type(f)(domain, arr)

    return impl(f)


def _broadcast_to(f, domain):
    # See note in _transpose about the singledispatch bypass.
    xp = get_array_ns(f)
    return type(f)(domain, xp.broadcast_to(f.ndarray, domain.shape))


def _jax_scan_unroll() -> int | bool:
    import os

    value = os.environ.get("GT4PY_JAX_SCAN_UNROLL", "1")
    return True if value == "all" else int(value)


@dataclasses.dataclass(frozen=True)
class ScanOperatorJax(EmbeddedOperator[xtyping.MaybeNestedInTuple[core_defs.ScalarT], _P]):
    """Scan operator emitting a single ``jax.lax.scan``.

    The scan body is traced once and compiled into a rolled XLA loop, rather
    than unrolled across the K range. Compile-time cost is independent of
    NLEV; runtime is one fused kernel.
    """

    forward: bool
    init: xtyping.MaybeNestedInTuple[core_defs.ScalarT]
    axis: common.Dimension

    def __call__(  # type: ignore[override]
        self,
        *args: common.Field | core_defs.Scalar,
        **kwargs: common.Field | core_defs.Scalar,
    ) -> (
        common.Field[Any, core_defs.ScalarT]
        | tuple[common.Field[Any, core_defs.ScalarT] | tuple, ...]
    ):
        if jax is None:
            raise RuntimeError("ScanOperatorJax requires jax to be installed.")
        from jax import lax, numpy as jnp

        scan_range = embedded_context.get_closure_column_range()
        assert self.axis == scan_range.dim
        scan_axis = scan_range.dim

        args = [_to_jax_field(arg) for arg in args]
        kwargs = {k: _to_jax_field(v) for k, v in kwargs.items()}

        all_args = [*args, *kwargs.values()]
        domain_intersection = _intersect_scan_args(*all_args)
        non_scan_domain = common.Domain(*[nr for nr in domain_intersection if nr.dim != scan_axis])

        out_domain = common.Domain(
            *[scan_range if nr.dim == scan_axis else nr for nr in domain_intersection]
        )
        if scan_axis not in out_domain.dims:
            # even if the scan dimension is not in the input, we can scan over it
            out_domain = common.Domain(*out_domain, (scan_range))

        init_type = _weak_init_type(self.init, all_args)
        assert isinstance(init_type, ts.TupleType | ts.ScalarType | ts.NamedCollectionType)

        from gt4py.next.embedded import nd_array_field

        def jax_fun(carry, x):
            # lax.scan presents one slice per step with the scan axis dropped.
            # The pytree unflatten reattaches the *full* domain to the sliced
            # ndarray; we rebuild the field with the trailing domain. Use the
            # direct constructor since common._field's singledispatch is not
            # registered for JAX tracers.
            x = utils.tree_map(
                lambda f: nd_array_field.JaxArrayField(f.domain[1:], f.ndarray)
            )(x)
            res = self.fun(carry, *x)
            return (res, res)

        def make_field(f):
            return nd_array_field.JaxArrayField(common.domain(out_domain), jnp.full(out_domain.shape, f))

        def scan_loop():
            new_dims = (scan_axis, *non_scan_domain.dims)
            new_args_seq = tuple(
                make_field(arg) if not isinstance(arg, common.Field) else arg for arg in args
            )
            new_args_seq = tuple(
                _transpose(
                    _broadcast_to(
                        fbuiltins.broadcast(arg[scan_range], (*non_scan_domain.dims, scan_axis)),
                        out_domain,
                    ),
                    new_dims,
                )
                for arg in new_args_seq
            )
            assert len(kwargs) == 0, "ScanOperatorJax does not yet support kwargs"

            init = field_utils.field_from_typespec(init_type, non_scan_domain, jnp)
            _tuple_assign_field(target=init, source=self.init, domain=non_scan_domain)
            # GT4PY_JAX_SCAN_UNROLL=<n|all>: unroll the XLA loop body n times
            # (or fully) so XLA can fuse across levels; the rolled loop is
            # launch-bound on GPUs for small horizontal sizes. Default 1.
            res = lax.scan(
                jax_fun, init, new_args_seq, reverse=not self.forward, unroll=_jax_scan_unroll()
            )
            res = res[1]
            res = utils.tree_map(
                lambda f: nd_array_field.JaxArrayField(
                    common.Domain(scan_range, *f.domain), f.ndarray
                )
            )(res)
            res = _transpose(res, out_domain.dims)
            return res

        res = scan_loop()
        return utils.tree_map(lambda a: _to_numpy_field(a))(res)


# PyTorch eager scan: the body iterates the K range with a Python for-loop,
# horizontal slice per K. Functionally identical to ScanOperatorVectorized;
# the alias exists so ``strategy="torch"`` reads naturally and we have a
# documented hook point for future torch-specific variants (e.g. wrapping
# the loop in ``torch.compile``).
#
# torch.autograd / torch.func.{jvp,vjp} traverse the Python loop natively;
# no torch.lax.scan equivalent is needed. ``torch._higher_order_ops.scan``
# does exist (PyTorch >= 2.5) but is forward-only — it fails under
# torch.func.{jvp,vjp} with a TorchDynamo functorch-unwrap error. See
# docs/pytorch-embedded-plan.md for details.
ScanOperatorTorch = ScanOperatorVectorized
# ``FieldOperator.__call__`` constructs a new ``EmbeddedOperator`` for every
# call, so a plain ``jax.jit(op)`` in ``field_operator_call`` would start
# from an empty dispatch cache each time: the operator was re-traced and
# re-compiled on every invocation (measured: 7 XLA compilations per call of
# the CLOUDSC2 stencil). ``EmbeddedOperator`` is a frozen dataclass, so
# operators built from the same definition compare and hash equal and can
# key one persistent jitted callable. Shapes / dtypes / domains still take
# part in JAX's own cache key, so a shape change re-traces as usual.
_JAX_JIT_CACHE: dict[EmbeddedOperator, Callable] = {}


def _jax_jitted(op: EmbeddedOperator) -> Callable:
    try:
        return _JAX_JIT_CACHE[op]
    except KeyError:
        assert jax is not None
        return _JAX_JIT_CACHE.setdefault(op, jax.jit(op))


def _get_out_domain(out: xtyping.MaybeNestedInTuple[common.MutableField]) -> common.Domain:
    return embedded_common.domain_intersection(
        *[f.domain for f in utils.flatten_nested_tuple((out,))]
    )


def field_operator_call(op: EmbeddedOperator[_R, _P], args: Any, kwargs: Any) -> Optional[_R]:
    if "out" in kwargs:
        # called from program or direct field_operator as program
        new_context_kwargs = {}
        if embedded_context.within_valid_context():
            # called from program
            assert "offset_provider" not in kwargs
        else:
            # field_operator as program
            if "offset_provider" not in kwargs:
                raise errors.MissingArgumentError(None, "offset_provider", True)
            offset_provider = kwargs.pop("offset_provider", None)

            new_context_kwargs["offset_provider"] = offset_provider

        out = kwargs.pop("out")

        domain = kwargs.pop("domain", None)

        # TODO(havogt): To do the assignment of the resulting fields we extract containers and act on plain tuples.
        # We currently apply the extract on both the rhs (`res`) computed by the operator and the lhs (`out`, provided by the user)
        # without checking if the types are consistent. However, these errors are caught in linting if enabled.
        container_extracted_out = arguments.extract(out)
        assert xtyping.is_maybe_nested_in_tuple_of(container_extracted_out, common.MutableField)  # type: ignore[type-abstract]  # MutableField is abstract/generic
        out_domain = (
            utils.tree_map(common.domain)(domain)
            if domain is not None
            else _get_out_domain(container_extracted_out)
        )

        new_context_kwargs["closure_column_range"] = _get_vertical_range(out_domain)

        with embedded_context.update(**new_context_kwargs):
            res = op(*args, **kwargs)
        container_extracted_res = arguments.extract(res)  # type: ignore[arg-type] # TODO(havogt): see notes above
        _tuple_assign_field(container_extracted_out, container_extracted_res, domain=out_domain)  # type: ignore[arg-type]
        return None
    else:
        # called from other field_operator or missing `out` argument
        domain = kwargs.pop("domain", None)
        offset_provider = kwargs.pop("offset_provider", None)

        # When called as a top-level program-like inline call, we are
        # responsible for installing offset_provider and closure_column_range
        # into the embedded context. When called from inside another
        # field_operator the context is already valid and we add nothing.
        new_context_kwargs: dict[str, Any] = {}
        if not embedded_context.within_valid_context():
            if offset_provider is not None:
                new_context_kwargs["offset_provider"] = offset_provider
            if domain is not None:
                new_context_kwargs["closure_column_range"] = _get_vertical_range(
                    utils.tree_map(common.domain)(domain)
                )

        # Decide which array namespace the inline call should use. The choice
        # is determined by the input fields' array_ns, not by what's importable
        # — this lets JAX and torch coexist in the same gt4py install with the
        # cloudsc2 driver pinning the namespace per call.
        input_ns = get_array_ns(*(arguments.extract(a) for a in (*args, *kwargs.values())))

        def _run():
            if jax is not None and jax.numpy is input_ns:
                # JAX inputs: jit the embedded operator so an inline
                # field_operator call inside a Python wrapper benefits from
                # tracing/fusion. Makes ``jax.jit`` / ``jax.jvp`` / ``jax.vjp``
                # over a wrapper that invokes a gt4py field operator do
                # something useful.
                return _jax_jitted(op)(*args, **kwargs)
            if torch is not None and torch is input_ns:
                # Torch inputs: run eagerly. ``torch.autograd`` / ``torch.func``
                # traverse the Python loop natively; we don't need ``torch.compile``
                # for correctness. The TorchCompileBackend (Phase E) is the
                # opt-in performance path. Fresh allocations (scan buffers,
                # 0-dim scalar tensors) follow the inputs' device.
                with field_utils.device_context(*args, *kwargs.values()):
                    return op(*args, **kwargs)
            # Plain NumPy or any other namespace: run eagerly.
            return op(*args, **kwargs)

        if new_context_kwargs:
            with embedded_context.update(**new_context_kwargs):
                res = _run()
        else:
            res = _run()

        if domain is not None:
            return _tuple_slice_field(res, common.domain(domain))  # type: ignore[return-value]
        return res


@utils.tree_map
def _get_vertical_range(domain: common.Domain) -> common.NamedRange | eve.NothingType:
    vertical_dim_filtered = [nr for nr in domain if nr.dim.kind == common.DimensionKind.VERTICAL]
    assert len(vertical_dim_filtered) <= 1
    return vertical_dim_filtered[0] if vertical_dim_filtered else eve.NOTHING


def _tuple_slice_field(
    field: xtyping.MaybeNestedInTuple[common.Field],
    domain: xtyping.MaybeNestedInTuple[common.Domain],
) -> xtyping.MaybeNestedInTuple[common.Field]:
    @named_collections.tree_map_named_collection
    def impl(field: common.Field, domain: common.Domain) -> common.Field:
        return field[domain]

    if not isinstance(domain, tuple):
        domain = named_collections.tree_map_named_collection(lambda _: domain)(field)
    return impl(field, domain)


def _tuple_assign_field(
    target: xtyping.MaybeNestedInTuple[common.MutableField],
    source: xtyping.MaybeNestedInTuple[common.Field],
    domain: xtyping.MaybeNestedInTuple[common.Domain],
) -> None:
    @named_collections.tree_map_named_collection
    def impl(target: common.MutableField, source: common.Field, domain: common.Domain) -> None:
        if isinstance(source, common.Field):
            target[domain] = source[domain]
        else:
            # Under jax.jit tracing the source can be a JAX tracer or a 0-dim
            # jnp array, neither of which passes core_defs.is_scalar_type but
            # both of which the downstream broadcast-and-assign handles fine.
            # assert core_defs.is_scalar_type(source)
            target[domain] = source

    if not isinstance(domain, tuple):
        domain = named_collections.tree_map_named_collection(lambda _: domain)(target)  # type: ignore[assignment] # typing not precise enough
    impl(target, source, domain)


def _tuple_broadcast_field(f, dims):
    @utils.tree_map
    def impl(f):
        return fbuiltins.broadcast(f, dims)

    return impl(f)


def _intersect_scan_args(
    *args: xtyping.MaybeNestedInTuple[core_defs.Scalar | common.Field],
) -> common.Domain:
    return embedded_common.domain_intersection(
        *[arg.domain for arg in utils.flatten_nested_tuple(args) if isinstance(arg, common.Field)]
    )


def _tuple_assign_value(
    pos: Sequence[common.NamedIndex],
    target: xtyping.MaybeNestedInTuple[common.MutableField],
    source: xtyping.MaybeNestedInTuple[core_defs.Scalar],
) -> None:
    @utils.tree_map
    def impl(target: common.MutableField, source: core_defs.Scalar) -> None:
        target[pos] = source

    impl(target, source)


def _tuple_at(
    pos: Sequence[common.NamedIndex],
    field: xtyping.MaybeNestedInTuple[common.Field | core_defs.Scalar],
) -> core_defs.Scalar | tuple[core_defs.ScalarT | tuple, ...]:
    @named_collections.tree_map_named_collection
    def impl(field: common.Field | core_defs.Scalar) -> core_defs.Scalar:
        res = field[pos].as_scalar() if isinstance(field, common.Field) else field
        assert core_defs.is_scalar_type(res)
        return res

    return impl(field)  # type: ignore[return-value]
