"""``release_fe_resources`` must let a finished solve's mesh -- and every MPI
communicator hanging off it -- go.

Every mesh and function space pins MPI communicators, and csdl_alpha never releases
a recorder's graph, so anything an op still references after the teardown sweep
leaks for the life of the process. The full suite then hits MPICH's 2048-context
limit and dies part-way through with exit 15 (0.11) or 134 (0.9), with no Python
traceback. Three such references have been found:

* ``ShellSolveOp._penalty_target`` -- a penalty BC's prescribed-value ``Function``;
* ``ShellFieldFormsOp.pde`` -- never cleared by the sweep;
* ``_node_idx`` on every op -- a zero-copy NumPy view of the C++ geometry. A view
  keeps its owner alive through a reference ``gc`` cannot see, so the leaked mesh
  had no visible referrer at all.

``hm.project``'s op was not swept at all. The context-count gate below is the one
that catches the invisible kind: a weak reference to the Python ``Mesh`` wrapper
dies even while the C++ mesh lives on.
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


def _solve_and_forget(method, outputs=False):
    """Solve on a fresh mesh; return only a weak reference to that mesh."""
    rec = csdl.Recorder(inline=True); rec.start()
    domain = hm.ShellDomain(rect_plate(4.0, 1.0, 4, 2))
    material = hm.isotropic(domain, E=1.0e7, nu=0.3, thickness=0.1, density=1.0)
    state = hm.solve(domain, material, hm.pressure(domain, 1.0),
                     hm.clamp(domain, where=hm.near("x", 0.0), method=method))
    hm.compliance(state)
    if outputs:
        hm.stress_field(state)
        hm.project(material.thickness, ("Lagrange", 1))
    rec.stop()
    return weakref.ref(domain.mesh)


@pytest.mark.parametrize("method", ["penalty", "strong"])
def test_solve_releases_its_mesh(method):
    mesh = _solve_and_forget(method)
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


@pytest.mark.parametrize("method", ["penalty", "strong"])
def test_fresh_mesh_solves_leak_no_contexts(method):
    _solve_and_forget(method, outputs=True)   # warm every one-time cache first
    release_fe_resources(); gc.collect()
    before = _free_contexts()
    for _ in range(4):
        _solve_and_forget(method, outputs=True)
    release_fe_resources(); gc.collect()
    assert _free_contexts() == before
