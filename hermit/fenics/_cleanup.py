"""Explicit teardown for the FE-backed custom operations.

``csdl_alpha`` (tracked from ``main``) never releases a recorder: every
``Recorder`` appends itself to ``manager.constructed_recorders`` with no removal,
and every graph node keeps a ``.recorder`` back-reference, so every op ever built
stays reachable. Hermit's ops cache PETSc objects that each hold a duplicated MPI
communicator -- ``ShellSolveOp`` a MUMPS factorization, ``ShellFieldFormsOp`` one
per projection space -- and reference a ``ShellPDE`` whose dozen-odd
``FunctionSpace``\\s each duplicate the mesh communicator through their index map.
Left alone, a long-lived process (the full pytest suite in one interpreter) walks
into ``MPIR_Get_contextid_sparse_group: Too many communicators`` and ``MPI_Abort``
around the 2048-context MPICH limit.

Python-level cleanup (``gc.collect``, clearing ``constructed_recorders``) does not
reclaim these -- the retention is multi-anchored across csdl internals. What works
is destroying the communicator-holding resources directly: ``.destroy()`` the
cached PETSc objects, and drop every reference an op holds to FE objects so their
C++ destructors (which free the index-map communicators) run on refcount. The op
shell may stay alive in a leaked graph; it no longer pins a communicator.

:func:`release_fe_resources` sweeps every live op and is meant for a pytest
``pytest_runtest_teardown`` hook (see ``tests/conftest.py``); it is not part of a
normal solve path.
"""

import gc


def _destroy(obj) -> None:
    """``obj.destroy()`` if it is a live petsc4py handle; ignore anything else."""
    try:
        obj.destroy()
    except Exception:
        pass


def _clear(op, *attrs) -> None:
    for a in attrs:
        try:
            cur = getattr(op, a)
        except AttributeError:
            continue
        setattr(op, a, {} if isinstance(cur, dict) else None)


def release(op) -> None:
    """Destroy one op's cached PETSc objects and drop its FE references.

    Safe to call more than once, and safe on a partially constructed op (``__init__``
    can raise after some attributes are set). Only call it once the op will not be
    evaluated again -- at test teardown, never mid-graph.
    """
    from .ops import ShellSolveOp, ShellScalarFormsOp, ShellFieldFormsOp

    if isinstance(op, ShellSolveOp):
        for m in getattr(op, "_dRdf", {}).values():
            _destroy(m)
        _destroy(getattr(op, "_ksp", None))
        _destroy(getattr(op, "_A", None))
        _clear(op, "_dRdf", "_ksp", "_A", "_funcs", "_forms", "_dR_form",
               "residual", "dR_dw", "Vc", "pde", "bc")
    elif isinstance(op, ShellScalarFormsOp):
        _clear(op, "_func", "forms", "Vc", "pde")
    elif isinstance(op, ShellFieldFormsOp):
        for ksp in getattr(op, "_ksp", {}).values():
            _destroy(ksp)
        _clear(op, "_ksp", "_holder", "_func", "space", "_L", "_vol", "_Mform",
               "_iexpr", "_ijexpr", "_ipts", "fields", "Vc")


def release_fe_resources() -> int:
    """Release every live Hermit custom op, and drop PDEs cached on domains.

    Returns the number of ops released.
    """
    from .ops import ShellSolveOp, ShellScalarFormsOp, ShellFieldFormsOp

    op_types = (ShellSolveOp, ShellScalarFormsOp, ShellFieldFormsOp)
    try:
        from ..domain import ShellDomain
    except Exception:  # pragma: no cover - domain module should always import
        ShellDomain = ()

    n = 0
    # Type-check first and never blind-``getattr`` an arbitrary heap object: some
    # (e.g. ``six``'s lazy module proxies) run an import on attribute access, which
    # can raise from inside a teardown hook.
    for obj in gc.get_objects():
        if isinstance(obj, op_types):
            release(obj)
            n += 1
        elif ShellDomain and isinstance(obj, ShellDomain) and obj.__dict__.get("_pde") is not None:
            # a ShellDomain memoises its ShellPDE in `_pde`; break that link so the
            # PDE's function spaces (each an index-map communicator) are collectable
            # once the ops let go too
            obj._pde = None

    # Drop csdl_alpha's permanent hold on every recorder ever constructed so the
    # (already unreachable) graphs stop accumulating. Only this list -- never the
    # active-recorder stack, which the `recorder` fixture's own `rec.stop()` still
    # has to unwind after this hook runs.
    try:
        from csdl_alpha.api import manager
        manager.constructed_recorders.clear()
    except Exception:
        pass

    gc.collect()
    return n
