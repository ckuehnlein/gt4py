# -*- coding: utf-8 -*-
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""PyTorch scan for embedded execution: traced / compiled bodies with explicit AD rules.

Used by :class:`gt4py.next.embedded.operators.ScanOperatorTorch` when
``GT4PY_TORCH_SCAN`` is ``trace`` or ``compile`` (the default ``eager`` mode
is the plain Python loop of ``ScanOperatorVectorized``). Everything torch
specific about scans lives here; the operators module only dispatches.
"""

from __future__ import annotations

import contextlib
import os
from types import ModuleType
from typing import Any, Callable, Optional, Sequence

import torch

from gt4py.next import common, field_utils, utils
from gt4py.next.embedded import context as embedded_context
from gt4py.next.embedded.operators import (
    _intersect_scan_args,
    _tuple_assign_field,
    _weak_init_type,
    get_array_ns,
)
from gt4py.next.otf import arguments
from gt4py.next.type_system import type_specifications as ts


# --- PyTorch scan ----------------------------------------------------------
#
# ``strategy="torch"`` runs the K loop in Python with one horizontal slice per
# level (``ScanOperatorVectorized`` semantics). In its default ``eager`` mode
# the body is the plain Python scan function, which torch.autograd and
# ``torch.func.{jvp,vjp}`` traverse natively, but every level re-runs the
# whole gt4py embedded machinery (domain intersection, Field construction,
# dtype checks): on CLOUDSC2 that is ~22k builtin ops per call at ~60 us each,
# two orders of magnitude above the tensor work.
#
# ``GT4PY_TORCH_SCAN=trace|compile`` is the torch counterpart of
# ``jax.jit`` + ``lax.scan``: the body is traced once with ``make_fx`` into
# an aten graph on raw tensors (optionally ``torch.compile``d), and the K loop
# runs over that graph inside a ``torch.autograd.Function`` that carries its
# own differentiation rules -- ``jvp`` loops a traced JVP body forward and
# ``backward`` loops a traced VJP body in reverse over the saved per-level
# carries (the scan output). Tracing under a transform is thereby never
# needed: ``forward`` / ``jvp`` / ``backward`` see plain tensors even under
# ``torch.func``. Inputs are unbound along K once and outputs stacked once.
#
# ``torch._higher_order_ops.scan`` is not used: it has no forward-mode AD
# (tangents are dropped, also under torch.compile), which the tangent-linear
# model needs. Launch overhead on GPUs is addressed one level up, by
# capturing whole calls with ``runners.torch_compile.graphed``.
_TORCH_SCAN_CACHE: dict[Any, Any] = {}


def _torch_scan_mode() -> str:
    mode = os.environ.get("GT4PY_TORCH_SCAN", "eager")
    if mode not in ("eager", "trace", "compile"):
        raise ValueError(f"GT4PY_TORCH_SCAN must be eager, trace or compile (got {mode!r}).")
    return mode


def _torch_tensor_key(t: Any) -> tuple:
    return (tuple(t.shape), str(t.dtype), str(t.device))


def _inline_scalar_constants(gm: Any) -> Any:
    """Replace 0-dim tensor constants of a traced graph by ``scalar_tensor`` nodes.

    The embedded builtins turn Python scalars into 0-dim tensors
    (``torch.asarray(1e-12, dtype=...)``); make_fx captures those as tensor
    constants (``get_attr`` + ``lift_fresh_copy``), which cannot be used
    under ``torch.func`` transforms ("Cannot access data pointer of Tensor
    that doesn't have storage"). Regenerating them from the Python value
    per call is transform-agnostic and free.
    """
    graph = gm.graph
    for node in list(graph.nodes):
        if node.op != "get_attr":
            continue
        t = getattr(gm, node.target, None)
        if not isinstance(t, torch.Tensor) or t.ndim != 0:
            continue
        with graph.inserting_after(node):
            new = graph.call_function(
                torch.ops.aten.scalar_tensor.default,
                (t.item(),),
                {"dtype": t.dtype, "device": t.device},
            )
        node.replace_all_uses_with(new)
        graph.erase_node(node)
    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten.lift_fresh_copy.default
            and node.args[0].op == "call_function"
            and node.args[0].target is torch.ops.aten.scalar_tensor.default
        ):
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
    # functorch's forward-AD formulas leak a few ``prims`` ops into the JVP
    # graph; prims cannot run on functorch-wrapped tensors, aten can.
    # Explicit overloads: a packet call under a functorch transform is
    # resolved through the Python dispatcher (torch._refs), ~40x slower.
    prims_to_aten = {
        torch.ops.prims.add.default: torch.ops.aten.add.Tensor,
        torch.ops.prims.mul.default: torch.ops.aten.mul.Tensor,
        torch.ops.prims.sub.default: torch.ops.aten.sub.Tensor,
        torch.ops.prims.div.default: torch.ops.aten.div.Tensor,
        torch.ops.prims.neg.default: torch.ops.aten.neg.default,
    }
    for node in graph.nodes:
        if node.op == "call_function" and node.target in prims_to_aten:
            node.target = prims_to_aten[node.target]
    graph.lint()
    gm.recompile()
    return gm


class _TorchScanBodies:
    """Traced (optionally compiled) fwd / jvp / vjp graphs of one scan body.

    fwd and vjp are traced on construction. The jvp body is traced through
    ``torch.func.jvp``, which opens a forward-AD dual level and cannot nest
    inside one that is already active (a TL call under ``forward_ad`` /
    ``torch.func.jvp`` before any plain call); it is therefore traced lazily
    at the first opportunity without an active dual level. Until then the
    JVP rule falls back to forward AD through the traced fwd graph.
    """

    def __init__(self, fun: Callable, samples: Sequence[Any], n_acc: int, mode: str) -> None:
        self.fun = fun
        self.n = len(samples)
        self.n_acc = n_acc
        self.mode = mode
        self.shapes = [(tuple(t.shape), t.dtype, t.device) for t in samples]
        self.float_in = [t.is_floating_point() for t in samples]
        self.compiled: dict[str, Callable] = {}
        self.jvp: Optional[Callable] = None
        with self._trace_context():
            self.fwd = self._trace(self.fun, 1)
            self.vjp = self._trace(self._vjp_fn, 2, cots=True)
        if mode == "compile":
            # Compile eagerly so the one-time cost sits in the first call
            # rather than in whichever later call first needs a body.
            self.get("fwd")
            self.get("vjp")
        self.try_trace_jvp()

    def _fresh(self) -> list:
        return [torch.zeros(sh, dtype=dt, device=dev) for sh, dt, dev in self.shapes]

    def _trace_context(self):
        # Trace with the functorch interpreter stack popped: tracing first
        # triggered inside an outer ``torch.func`` transform otherwise
        # records a wrong graph (inputs captured as constants). Sample
        # tensors are created inside this context for the same reason.
        from torch._functorch.pyfunctorch import temporarily_pop_interpreter_stack

        popped = (
            temporarily_pop_interpreter_stack()
            if torch._C._functorch.peek_interpreter_stack() is not None
            else contextlib.nullcontext()
        )
        return _Both(popped, torch.no_grad())

    def _trace(self, fn: Callable, n_sets: int, cots: bool = False) -> Callable:
        from torch.fx.experimental.proxy_tensor import make_fx

        args = []
        for i in range(n_sets):
            fresh = self._fresh()
            args += fresh[: self.n_acc] if (cots and i == n_sets - 1) else fresh
        gm = _inline_scalar_constants(make_fx(fn, tracing_mode="real")(*args))
        return gm

    def _substitute(self, primals: list, fl: Sequence[Any]) -> list:
        full = list(primals)
        for i, v in zip([i for i, f in enumerate(self.float_in) if f], fl):
            full[i] = v
        return full

    def _jvp_fn(self, *ts):
        primals, tangents = list(ts[: self.n]), ts[self.n :]

        def f_float(*fl):
            outs = self.fun(*self._substitute(primals, fl))
            return tuple(o for o in outs if o.is_floating_point()), outs

        fp = tuple(p for p, f in zip(primals, self.float_in) if f)
        ft = tuple(t for t, f in zip(tangents, self.float_in) if f)
        _, tans, outs = torch.func.jvp(f_float, fp, ft, has_aux=True)
        it = iter(tans)
        return tuple(next(it) if o.is_floating_point() else torch.zeros_like(o) for o in outs)

    def _vjp_fn(self, *ts):
        primals, cots = list(ts[: self.n]), ts[self.n :]

        def f_float(*fl):
            outs = self.fun(*self._substitute(primals, fl))
            return tuple(o for o in outs if o.is_floating_point()), outs

        fp = tuple(p for p, f in zip(primals, self.float_in) if f)
        _, pullback, outs = torch.func.vjp(f_float, *fp, has_aux=True)
        fc = tuple(c for c, o in zip(cots, outs) if o.is_floating_point())
        grads = iter(pullback(fc))
        return tuple(
            next(grads) if f else torch.zeros_like(p) for p, f in zip(primals, self.float_in)
        )

    def try_trace_jvp(self) -> None:
        import torch.autograd.forward_ad as fwAD

        if self.jvp is not None or fwAD._current_level >= 0:
            return
        with self._trace_context():
            self.jvp = self._trace(self._jvp_fn, 2)
        if self.mode == "compile":
            self.get("jvp")

    def _jvp_via_fwd(self, *ts):
        """JVP rule fallback: forward AD through the traced fwd graph at the current dual level."""
        import torch.autograd.forward_ad as fwAD

        primals, tangents = ts[: self.n], ts[self.n :]
        duals = [
            fwAD.make_dual(p, t) if p.is_floating_point() else p for p, t in zip(primals, tangents)
        ]
        outs = self.fwd(*duals)
        res = []
        for o in outs:
            tan = fwAD.unpack_dual(o).tangent if o.is_floating_point() else None
            res.append(torch.zeros_like(o) if tan is None else tan)
        return tuple(res)

    def get(self, name: str, *tensors: Any) -> Callable:
        """Body ``name``: compiled when ``tensors`` are plain, traced otherwise.

        Dynamo-compiled code cannot run on functorch-wrapped tensors (e.g.
        inside ``backward`` under ``torch.func.vjp``, where functorch pops
        its interpreter stack but the tensors stay wrapped); the traced aten
        graph can. Plain ``torch.autograd`` / ``forward_ad`` get the
        compiled ones.
        """
        graph = getattr(self, name)
        if graph is None:
            assert name == "jvp"
            return self._jvp_via_fwd
        if self.mode == "compile" and not any(
            torch._C._functorch.is_functorch_wrapped_tensor(t) for t in tensors
        ):
            if name not in self.compiled:
                # Inductor directly on the aten graph (no TorchDynamo): Dynamo
                # re-analyses the whole GraphModule bytecode (minutes for the
                # JVP graph) and adds ~15 us of guard evaluation per call;
                # the artifact is called with positional tensors. Shapes,
                # dtypes and devices are fixed by the cache key of the bodies.
                from torch._inductor import compile as inductor_compile

                n_sets = {"fwd": 1, "jvp": 2, "vjp": 2}[name]
                example = []
                for i in range(n_sets):
                    fresh = self._fresh()
                    example += fresh[: self.n_acc] if (name == "vjp" and i == 1) else fresh
                compiled = inductor_compile(graph, example)

                def run(*ts, _c=compiled):
                    # Inductor artifacts assert the example strides
                    # (contiguous); per-level slices normally are (K-first
                    # layout below), this covers the remaining cases.
                    return _c(*[t if t.is_contiguous() else t.contiguous() for t in ts])

                self.compiled[name] = run
            return self.compiled[name]
        return graph


class _Both:
    """Enter two context managers (tiny helper avoiding ExitStack boilerplate)."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __enter__(self):
        self.a.__enter__()
        self.b.__enter__()

    def __exit__(self, *exc):
        self.b.__exit__(*exc)
        return self.a.__exit__(*exc)


def _scan_levels(t: Any, ax: int) -> tuple:
    """Per-level slices, each contiguous: one K-first copy instead of NLEV strided views."""
    return t.movedim(ax, 0).contiguous().unbind(0)


def _scan_stack(levels: list, n_acc: int, out_axis: int) -> tuple:
    """Stack per-level carries K-first (contiguous level slices), K viewed at out_axis."""
    return tuple(
        torch.stack([lv[i] for lv in levels], dim=0).movedim(0, out_axis) for i in range(n_acc)
    )


def _scan_carry_in(init: Sequence[Any], outs: Sequence[Any], out_axis: int, order: Sequence[int], pos: int) -> list:
    """Carry entering level ``order[pos]``: init or the previous level's output."""
    if pos == 0:
        return list(init)
    return [o.select(out_axis, order[pos - 1]) for o in outs]


def _scan_fwd_loop(bodies, n_acc, n_scan, axes, out_axis, order, *tensors):
    acc = list(tensors[:n_acc])
    scanned = [_scan_levels(t, ax) for t, ax in zip(tensors[n_acc : n_acc + n_scan], axes)]
    static = list(tensors[n_acc + n_scan :])
    outs: list = [None] * len(order)
    fwd = bodies.get("fwd", *acc)
    for k in order:
        acc = list(fwd(*acc, *(s[k] for s in scanned), *static))
        outs[k] = acc
    return _scan_stack(outs, n_acc, out_axis)


def _scan_jvp_loop(bodies, n_acc, n_scan, axes, out_axis, order, *args):
    """args = saved (inputs + outputs) followed by one tangent per input."""
    n_in = (len(args) - n_acc) // 2
    saved, tangents = args[: n_in + n_acc], args[n_in + n_acc :]
    scanned = [_scan_levels(t, ax) for t, ax in zip(saved[n_acc : n_acc + n_scan], axes)]
    static = list(saved[n_acc + n_scan : n_in])
    outs = saved[n_in:]
    acc_t = list(tangents[:n_acc])
    scanned_t = [_scan_levels(t, ax) for t, ax in zip(tangents[n_acc : n_acc + n_scan], axes)]
    static_t = list(tangents[n_acc + n_scan :])
    outs_t: list = [None] * len(order)
    jvp_body = bodies.get("jvp", *saved[:n_acc], *acc_t)
    for pos, k in enumerate(order):
        cin = _scan_carry_in(saved[:n_acc], outs, out_axis, order, pos)
        acc_t = list(
            jvp_body(
                *cin, *(s[k] for s in scanned), *static,
                *acc_t, *(s[k] for s in scanned_t), *static_t,
            )
        )
        outs_t[k] = acc_t
    return _scan_stack(outs_t, n_acc, out_axis)


def _scan_bwd_loop(bodies, n_acc, n_scan, axes, out_axis, order, *args):
    """args = saved (inputs + outputs) followed by one cotangent per output."""
    saved, grads_out = args[:-n_acc], args[-n_acc:]
    n_in = len(saved) - n_acc
    scanned = [_scan_levels(t, ax) for t, ax in zip(saved[n_acc : n_acc + n_scan], axes)]
    static = list(saved[n_acc + n_scan : n_in])
    outs = saved[n_in:]
    gacc = [torch.zeros_like(t) for t in saved[:n_acc]]
    gscan: list = [[None] * len(order) for _ in range(n_scan)]
    gstatic = [torch.zeros_like(t) for t in static]
    vjp_body = bodies.get("vjp", *saved[:n_acc], *grads_out)
    for pos in reversed(range(len(order))):
        k = order[pos]
        cin = _scan_carry_in(saved[:n_acc], outs, out_axis, order, pos)
        cot = [g.select(out_axis, k) + ga for g, ga in zip(grads_out, gacc)]
        g = vjp_body(*cin, *(s[k] for s in scanned), *static, *cot)
        gacc = list(g[:n_acc])
        for i in range(n_scan):
            gscan[i][k] = g[n_acc + i]
        for i in range(len(static)):
            gstatic[i] = gstatic[i] + g[n_acc + n_scan + i]
    gscan_stacked = [torch.stack(gs, dim=0).movedim(0, ax) for gs, ax in zip(gscan, axes)]
    return (*gacc, *gscan_stacked, *gstatic)


class _TorchScan(torch.autograd.Function):
    """K loop over traced bodies with explicit JVP / VJP rules.

    Tensor inputs: ``n_acc`` carry-init tensors, ``n_scan`` K-stacked
    tensors (K at ``axes[i]``), the rest level-invariant tensors. Outputs:
    ``n_acc`` K-stacked carries (K at ``out_axis``).
    """

    @staticmethod
    def forward(bodies, n_acc, n_scan, axes, out_axis, order, *tensors):
        return _scan_fwd_loop(bodies, n_acc, n_scan, axes, out_axis, order, *tensors)

    @staticmethod
    def setup_context(ctx, inputs, output):
        bodies, n_acc, n_scan, axes, out_axis, order, *tensors = inputs
        ctx.bodies = bodies
        ctx.meta = (n_acc, n_scan, axes, out_axis, order)
        ctx.save_for_backward(*tensors, *output)
        ctx.save_for_forward(*tensors, *output)
        non_diff = [o for o in output if not o.is_floating_point()]
        if non_diff:
            ctx.mark_non_differentiable(*non_diff)

    @staticmethod
    def jvp(ctx, *tangents):
        n_acc = ctx.meta[0]
        saved = ctx.saved_tensors
        tangents = tangents[6:]  # one slot per forward() argument; the first six are non-tensors
        tangents = [torch.zeros_like(s) if t is None else t for t, s in zip(tangents, saved)]
        stacked = _scan_jvp_loop(ctx.bodies, *ctx.meta, *saved, *tangents)
        return tuple(
            stacked[i] if saved[-n_acc + i].is_floating_point() else None for i in range(n_acc)
        )

    @staticmethod
    def backward(ctx, *grads_out):
        n_acc = ctx.meta[0]
        saved = ctx.saved_tensors
        grads_out = [
            torch.zeros_like(o) if g is None else g for g, o in zip(grads_out, saved[-n_acc:])
        ]
        grads = _scan_bwd_loop(ctx.bodies, *ctx.meta, *saved, *grads_out)
        return (None,) * 6 + tuple(
            g if t.is_floating_point() else None for g, t in zip(grads, saved)
        )



def scan_call(op: Any, args: Sequence[Any], kwargs: dict, mode: str, xp: ModuleType) -> Any:
    """Run ``op`` (a ScanOperatorTorch) with traced / compiled bodies."""
    import torch.utils._pytree as pytree

    all_args = [*args, *kwargs.values()]

    scan_range = embedded_context.get_closure_column_range()
    assert op.axis == scan_range.dim
    scan_axis = scan_range.dim
    domain_intersection = _intersect_scan_args(*all_args)
    non_scan_domain = common.Domain(*[nr for nr in domain_intersection if nr.dim != scan_axis])
    out_domain = common.Domain(
        *[scan_range if nr.dim == scan_axis else nr for nr in domain_intersection]
    )
    if scan_axis not in out_domain.dims:
        out_domain = common.Domain(*out_domain, (scan_range))
    out_axis = out_domain.dims.index(scan_axis)
    init_type = _weak_init_type(op.init, all_args)
    assert isinstance(init_type, ts.TupleType | ts.ScalarType | ts.NamedCollectionType)

    # Carry: pytree of TorchTensorFields over the non-scan domain.
    init = field_utils.field_from_typespec(
        init_type, non_scan_domain, xp, field_utils.device_of(*all_args)
    )
    _tuple_assign_field(target=init, source=op.init, domain=non_scan_domain)
    # TorchTensorField is a torch pytree node (leaves = tensors, context =
    # domain), so the spec rebuilds Fields over the non-scan domain.
    acc_leaves = utils.flatten_nested_tuple(init)
    acc_tensors = [f.ndarray for f in acc_leaves]
    _, acc_spec = pytree.tree_flatten(init)
    n_acc = len(acc_tensors)

    # Inputs: scanned fields (unbound along K per level), level-invariant
    # fields, and scalars (baked into the traced graph as constants).
    kw_names = tuple(kwargs)
    spec: list = []  # per argument: ("scan", idx, domain, axis) | ("static", idx, domain) | ("scalar", value)
    scan_tensors: list = []
    scan_axes: list = []
    static_tensors: list = []
    for arg in all_args:
        if isinstance(arg, common.Field):
            restricted = arg[common.Domain(*(nr for nr in out_domain if nr.dim in arg.domain.dims))]
            level_domain = common.Domain(*(nr for nr in restricted.domain if nr.dim != scan_axis))
            if scan_axis in restricted.domain.dims:
                ax = restricted.domain.dims.index(scan_axis)
                spec.append(("scan", len(scan_tensors), level_domain, ax))
                scan_tensors.append(restricted.ndarray)
                scan_axes.append(ax)
            else:
                spec.append(("static", len(static_tensors), level_domain))
                static_tensors.append(restricted.ndarray)
        else:
            spec.append(("scalar", arg))
    n_scan = len(scan_tensors)
    n_pos = len(args)
    fun = op.fun

    def body_tensors(*ts):
        acc = pytree.tree_unflatten(list(ts[:n_acc]), acc_spec)
        lvl = ts[n_acc:]
        rebuilt = []
        for sp in spec:
            if sp[0] == "scan":
                rebuilt.append(common._field(lvl[sp[1]], domain=sp[2]))
            elif sp[0] == "static":
                rebuilt.append(common._field(lvl[n_scan + sp[1]], domain=sp[2]))
            else:
                rebuilt.append(sp[1])
        res = fun(acc, *rebuilt[:n_pos], **dict(zip(kw_names, rebuilt[n_pos:])))
        res_flat = utils.flatten_nested_tuple(res)
        assert len(res_flat) == n_acc, "scan body must return the carry structure"
        out = []
        for r, ref in zip(res_flat, ts[:n_acc]):
            t = r.ndarray if isinstance(r, common.Field) else torch.as_tensor(r, dtype=ref.dtype, device=ref.device)
            if t.dtype != ref.dtype:
                t = t.to(ref.dtype)
            out.append(t if tuple(t.shape) == tuple(ref.shape) else torch.broadcast_to(t, ref.shape))
        return tuple(out)

    samples = [*acc_tensors, *(t.select(ax, 0) for t, ax in zip(scan_tensors, scan_axes)), *static_tensors]
    key = (
        op, mode, scan_range, out_domain, n_acc,
        tuple(_torch_tensor_key(t) for t in samples),
        tuple(sp if sp[0] == "scalar" else sp[:1] + sp[2:] for sp in spec),
    )
    bodies = _TORCH_SCAN_CACHE.get(key)
    if bodies is None:
        bodies = _TORCH_SCAN_CACHE[key] = _TorchScanBodies(body_tensors, samples, n_acc, mode)
    else:
        bodies.try_trace_jvp()

    order = list(range(len(scan_range.unit_range)))
    if not op.forward:
        order.reverse()
    outs = _TorchScan.apply(
        bodies, n_acc, n_scan, tuple(scan_axes), out_axis, tuple(order),
        *acc_tensors, *scan_tensors, *static_tensors,
    )
    return utils.tree_map(lambda f: common._field(f.ndarray, domain=out_domain))(
        pytree.tree_unflatten(list(outs), acc_spec)
    )
