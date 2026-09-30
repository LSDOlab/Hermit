"""``release_fe_resources`` must let a finished solve's mesh go.

Every mesh and function space pins MPI communicators, and csdl_alpha never releases
a recorder's graph, so anything an op still references after the teardown sweep
leaks for the life of the process -- the full suite then hits MPICH's 2048-context
limit and dies with exit 15 (0.11) or 134 (0.9) part-way through. A penalty BC's
prescribed-value ``Function`` held on ``ShellSolveOp`` once leaked one mesh per solve
this way.

A second leak is still open: each solve on a *fresh* mesh loses ~5 contexts (0.11;
~6 on 0.9) even after that mesh has been collected, while repeat solves on one mesh
lose none. Until it is fixed, a test that loops over evaluations should build its
mesh once.
"""

import gc
import pathlib
import sys
import weakref

import csdl_alpha as csdl
import pytest

import hermit as hm
from hermit.fenics._cleanup import release_fe_resources

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples" / "verification"))
from _geometry import rect_plate  # noqa: E402


def _solve_and_forget(method):
    """Solve on a fresh mesh; return only a weak reference to that mesh."""
    rec = csdl.Recorder(inline=True); rec.start()
    domain = hm.ShellDomain(rect_plate(4.0, 1.0, 4, 2))
    material = hm.isotropic(domain, E=1.0e7, nu=0.3, thickness=0.1, density=1.0)
    state = hm.solve(domain, material, hm.pressure(domain, 1.0),
                     hm.clamp(domain, where=hm.near("x", 0.0), method=method))
    hm.compliance(state)
    rec.stop()
    return weakref.ref(domain.mesh)


def test_penalty_solve_releases_its_mesh():
    mesh = _solve_and_forget("penalty")
    release_fe_resources()
    gc.collect()
    assert mesh() is None


def test_strong_solve_releases_its_mesh():
    mesh = _solve_and_forget("strong")
    release_fe_resources()
    gc.collect()
    assert mesh() is None


def _free_contexts():
    """How many more communicators MPICH will hand out, by duplicating
    ``COMM_WORLD`` until it refuses (then freeing them all)."""
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    handler = world.Get_errhandler()
    world.Set_errhandler(MPI.ERRORS_RETURN)
    comms = []
    try:
        while len(comms) < 4096:
            comms.append(world.Dup())
    except MPI.Exception:
        pass
    finally:
        for c in comms:
            c.Free()
        world.Set_errhandler(handler)
        handler.Free()
    return len(comms)


@pytest.mark.xfail(strict=True, reason="open: each fresh-mesh solve leaks ~5 MPI contexts "
                                       "that outlive the mesh; see the module docstring")
def test_fresh_mesh_solves_leak_no_contexts():
    _solve_and_forget("penalty")          # warm every one-time cache first
    release_fe_resources(); gc.collect()
    before = _free_contexts()
    for _ in range(4):
        _solve_and_forget("penalty")
    release_fe_resources(); gc.collect()
    assert _free_contexts() == before
