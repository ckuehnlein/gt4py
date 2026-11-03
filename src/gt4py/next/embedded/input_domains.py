# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Input domains a field operator needs to produce a given output domain.

The embedded executor evaluates a field operator on the whole domain of its inputs and the
caller then keeps the part it wants. A compiled backend instead computes the output domain
only. The difference shows where inputs hold garbage outside the region of interest (halo
rows never written, say): the embedded executor evaluates there too, which emits warnings,
costs time and, under reverse-mode automatic differentiation, poisons the gradients with
``0 * nan``. :func:`accessed_input_domains` runs the compiled pipeline's domain inference on
the operator's IR for a concrete output domain, so the caller can restrict each input to the
points actually read before calling the operator inline.
"""

from __future__ import annotations

from typing import Any, TypeAlias

from gt4py import eve
from gt4py.next import common, utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import domain_utils, ir_makers as im
from gt4py.next.iterator.transforms import (
    concat_where,
    dead_code_elimination,
    expand_tuple_maps,
    infer_domain,
    infer_domain_ops,
    inline_dynamic_shifts,
    inline_fundefs,
    inline_lifts,
)
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.transforms.merge_let import MergeLet
from gt4py.next.iterator.transforms.normalize_shifts import NormalizeShifts
from gt4py.next.iterator.type_system import inference as itir_type_inference
from gt4py.next.type_system import type_specifications as ts


#: Per dimension ``(start, stop)``; ``None`` for an unbounded side.
Bounds: TypeAlias = tuple[int | None, int | None]
#: The domain read of one input: per dimension bounds, nested tuples for tuple inputs,
#: ``None`` for an input that is never read.
InputDomain: TypeAlias = dict[common.Dimension, Bounds] | tuple | None


def _literal(value: Any, type_: ts.TypeSpec) -> itir.Expr:
    """``value`` as a literal (a tuple of literals for a tuple) of the declared ``type_``."""
    if isinstance(type_, ts.TupleType):
        return im.make_tuple(*(_literal(v, t) for v, t in zip(value, type_.types, strict=True)))
    assert isinstance(type_, ts.ScalarType)
    return itir.Literal(value=str(value), type=type_)


class _SubstituteScalars(eve.NodeTranslator):
    """Replace references to scalar parameters by literals of their values."""

    def __init__(self, values: dict[str, Any], types: dict[str, ts.TypeSpec]) -> None:
        self._values = values
        self._types = types

    def visit_SymRef(self, node: itir.SymRef) -> itir.Expr:
        if node.id in self._values:
            return _literal(self._values[node.id], self._types[node.id])
        return node


def _domain_expr(domain: common.Domain) -> itir.FunCall:
    return im.domain(
        common.GridType.CARTESIAN,
        {
            dim: (im.literal_from_value(rng.start), im.literal_from_value(rng.stop))
            for dim, rng in zip(domain.dims, domain.ranges, strict=True)
        },
    )


def _bound(expr: itir.Expr) -> int | None:
    if isinstance(expr, itir.InfinityLiteral):
        return None
    if isinstance(expr, itir.Literal):
        return int(expr.value)
    raise ValueError(f"Domain bound '{expr}' is not static.")


def _to_bounds(domain: domain_utils.SymbolicDomain | None) -> dict[common.Dimension, Bounds] | None:
    if domain is None:
        return None
    folded = domain_utils.SymbolicDomain.from_expr(ConstantFolding.apply(domain.as_expr()))
    return {dim: (_bound(rng.start), _bound(rng.stop)) for dim, rng in folded.ranges.items()}


def accessed_input_domains(
    operator: Any,
    out_domain: common.Domain,
    scalar_args: dict[str, Any] | None = None,
    offset_provider: common.OffsetProvider | None = None,
) -> dict[str, InputDomain]:
    """Domain each parameter of ``operator`` is read on when the output is ``out_domain``.

    Args:
        operator: A ``FieldOperator`` (its ``__gt_gtir__`` and ``__gt_type__`` are used).
        out_domain: The (cartesian) domain of the output; all outputs of a tuple-returning
            operator share it.
        scalar_args: Values of the integer and boolean scalar parameters. They decide the
            domains that depend on them (``concat_where`` on an index compared to a scalar).
        offset_provider: The offset provider the operator is called with.

    Returns:
        The bounds read per parameter name, see :data:`InputDomain`. Scalar parameters are
        absent.

    Raises:
        ValueError: when a bound does not resolve to a number (e.g. a scalar parameter the
            domain depends on is missing from ``scalar_args``).
    """
    scalar_args = scalar_args or {}
    offset_provider = offset_provider or {}
    offset_provider_type = common.offset_provider_to_type(offset_provider)

    fundef = operator.__gt_gtir__()
    func_type = operator.__gt_type__().definition
    param_types = {**func_type.pos_or_kw_args, **func_type.kw_only_args}
    params = [itir.Sym(id=p.id, type=param_types[str(p.id)]) for p in fundef.params]

    fundefs: list[itir.FunctionDefinition] = []
    _collect_callee_fundefs(operator, fundefs)

    domain = _domain_expr(out_domain)
    expr = _SubstituteScalars(scalar_args, param_types).visit(fundef.expr)
    program = itir.Program(
        id=f"{fundef.id}_input_domains",
        function_definitions=fundefs,
        params=[*params, itir.Sym(id="out", type=func_type.returns)],
        declarations=[],
        body=[itir.SetAt(expr=expr, domain=domain, target=im.ref("out"))],
    )

    # The passes `pass_manager.apply_common_transforms` runs before its domain inference.
    uids = utils.IDGeneratorPool()
    program = MergeLet().visit(program)
    program = inline_fundefs.InlineFundefs().visit(program)
    program = inline_fundefs.prune_unreferenced_fundefs(program)
    program = itir_type_inference.infer(program, offset_provider_type=offset_provider_type)
    program = NormalizeShifts().visit(program)
    program = inline_lifts.InlineLifts().visit(program)
    program = concat_where.expand_tuple_args(program, offset_provider_type=offset_provider_type)
    program = expand_tuple_maps.ExpandTupleMaps.apply(
        program, uids=uids, offset_provider_type=offset_provider_type
    )
    program = dead_code_elimination.dead_code_elimination(
        program, uids=uids, offset_provider_type=offset_provider_type
    )
    program = inline_dynamic_shifts.InlineDynamicShifts.apply(
        program, offset_provider_type=offset_provider_type, uids=uids
    )
    program = infer_domain_ops.InferDomainOps.apply(program)
    program = concat_where.canonicalize_domain_argument(program)

    _, accessed = infer_domain.infer_expr(
        program.body[0].expr,
        domain_utils.SymbolicDomain.from_expr(domain),
        offset_provider=offset_provider,
    )
    return {
        str(p.id): utils.tree_map(_to_bounds)(accessed[str(p.id)])
        if str(p.id) in accessed
        else None
        for p in fundef.params
        if str(p.id) not in scalar_args
    }


def _collect_callee_fundefs(operator: Any, fundefs: list[itir.FunctionDefinition]) -> None:
    known = {f.id for f in fundefs}
    for value in operator.__gt_closure_vars__().values():
        if hasattr(value, "__gt_gtir__") and hasattr(value, "__gt_closure_vars__"):
            fundef = value.__gt_gtir__()
            if fundef.id not in known:
                fundefs.append(fundef)
                known.add(fundef.id)
                _collect_callee_fundefs(value, fundefs)


def restrict_to_bounds(field: common.Field, bounds: dict[common.Dimension, Bounds]) -> common.Field:
    """``field`` restricted to ``bounds`` clipped to its own domain (a view where possible)."""
    ranges = []
    for dim, rng in zip(field.domain.dims, field.domain.ranges, strict=True):
        if dim not in bounds:
            raise ValueError(f"No bounds for dimension '{dim}' of the field.")
        start, stop = bounds[dim]
        start = rng.start if start is None else max(start, rng.start)
        stop = rng.stop if stop is None else min(stop, rng.stop)
        ranges.append((dim, (start, max(start, stop))))
    new_domain = common.domain(ranges)
    if new_domain == field.domain:
        return field
    return field[new_domain]
