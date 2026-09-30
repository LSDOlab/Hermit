"""``release_fe_resources`` must let a finished solve's mesh go.

Every mesh and function space pins MPI communicators, and csdl_alpha never releases
a recorder's graph, so anything an op still references after the teardown sweep
leaks for the life of the process -- the full suite then hits MPICH's 2048-context
limit and dies with exit 15 part-way through. A penalty BC's prescribed-value
``Function`` held on ``ShellSolveOp`` once leaked one mesh per solve this way.
"""

import gc
import pathlib
import sys
import weakref

import csdl_alpha as csdl

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
