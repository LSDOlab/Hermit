"""Closed and branched shell meshes: boxes, internal ribs and T-junctions.

Two defects surfaced on a wingbox (skins, spars and ribs meeting at edges):

* ``check_cell_orientation_consistency`` compared adjacent normals, so a closed box
  could never pass -- at a 90 degree corner the normals are perpendicular whatever
  the winding -- and ``hm.pressure`` was unusable on one. The check is now
  topological (shared edges traversed in opposite directions).
* A penalty BC located on facets shared by three or more cells constrained nothing:
  DOLFINx drops such facets from a ``MeshTags``-built ``dS``. The solve went
  singular without an error.

Meshes are built in memory from cyclic quads.
"""

import inspect

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit as hm

E, NU, H, RHO = 70.0e9, 0.3, 0.01, 2700.0


def _build(points, cyclic):
    import basix.ufl
    import ufl
    from dolfinx.mesh import create_mesh
    from mpi4py import MPI

    points = np.ascontiguousarray(points, dtype=np.float64)
    cells = np.ascontiguousarray(np.asarray(cyclic)[:, [0, 1, 3, 2]], dtype=np.int64)
    el = ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(3,)))
    if list(inspect.signature(create_mesh).parameters)[2] == "e":   # DOLFINx 0.11
        return create_mesh(MPI.COMM_WORLD, cells, el, points)
    return create_mesh(MPI.COMM_WORLD, cells, points, el)           # DOLFINx 0.9


def _assemble(patches):
    """Merge ``(origin, u, v, n)`` patches -- an ``n x n`` grid of quads wound so that
    ``u x v`` is the normal -- into ``(points, cyclic cells)`` with shared nodes."""
    pts, cells = [], []
    for origin, u, v, n in patches:
        s = np.linspace(0.0, 1.0, n + 1)
        grid = (np.asarray(origin, float) + s[:, None, None] * np.asarray(u, float)
                + s[None, :, None] * np.asarray(v, float)).reshape(-1, 3)
        k = lambda i, j: i * (n + 1) + j
        base = sum(len(p) for p in pts)
        cells += [[base + k(i, j), base + k(i + 1, j), base + k(i + 1, j + 1), base + k(i, j + 1)]
                  for i in range(n) for j in range(n)]
        pts.append(grid)
    pts = np.concatenate(pts)
    uniq, inverse = np.unique(np.round(pts, 12), axis=0, return_inverse=True)
    return uniq, inverse.ravel()[np.asarray(cells)]


def _box(n=4, rib=False):
    """Unit-cube surface wound outward; optionally an internal rib at ``x = 0.5``."""
    e = np.eye(3)
    patches = [((0, 0, 0), e[1], e[0], n), ((0, 0, 1), e[0], e[1], n),     # z = 0, z = 1
               ((0, 0, 0), e[0], e[2], n), ((0, 1, 0), e[2], e[0], n),     # y = 0, y = 1
               ((0, 0, 0), e[2], e[1], n), ((1, 0, 0), e[1], e[2], n)]     # x = 0, x = 1
    if rib:
        patches.append(((0.5, 0, 0), e[1], e[2], n))
    return _assemble(patches)


def _fins(n=4):
    """Three unit plates sharing the edge ``x = z = 0``: fins along +x, -x and +z."""
    e = np.eye(3)
    return _assemble([((0, 0, 0), e[0], e[1], n), ((0, 0, 0), e[1], -e[0], n),
                      ((0, 0, 0), e[1], e[2], n)])


def _material(domain):
    return hm.isotropic(domain, E=E, nu=NU, thickness=H, density=RHO)


def _compliance(state):
    return float(np.ravel(hm.compliance(state).value)[0])


# -- orientation ---------------------------------------------------------------

def test_vertex_loop_follows_the_frame_normal(plate_mesh):
    """The check's premise: the basix vertex loop (0, 1, 3, 2) of a quad turns about
    the same normal ``local_frames`` (hence ``CellNormal``) uses."""
    for mesh in (plate_mesh, _build(*_box())):
        d = hm.ShellDomain(mesh)
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim, 0)
        v = np.asarray(mesh.topology.connectivity(tdim, 0).array).reshape(d.n_cells, 4)
        x = mesh.geometry.x[v[:, [0, 1, 3, 2]]]   # topology vertex == geometry node for P1
        loop_n = np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 1])
        assert np.all(np.einsum("ij,ij->i", loop_n, d.local_frames()[:, 2]) > 0)


@pytest.mark.parametrize("rib", [False, True], ids=["closed-box", "box-with-rib"])
def test_closed_box_passes_orientation_check(rib):
    d = hm.ShellDomain(_build(*_box(rib=rib)))
    d.check_cell_orientation_consistency()


def test_flipped_cell_in_box_raises():
    points, cells = _box()
    cells[5] = cells[5][::-1]
    d = hm.ShellDomain(_build(points, cells))
    with pytest.raises(ValueError, match="inconsistently oriented"):
        hm.pressure(d, 1.0)


def test_tol_is_deprecated(plate_mesh):
    with pytest.warns(DeprecationWarning):
        hm.ShellDomain(plate_mesh).check_cell_orientation_consistency(tol=-0.5)


def test_pressure_on_box_face_matches_normal_traction(recorder):
    """``hm.pressure`` on the bottom of a closed box equals the traction ``p * n``."""
    points, cells = _box(rib=True)
    d = hm.ShellDomain(_build(points, cells))
    bottom = np.all(np.isclose(points[cells][:, :, 2], 0.0), axis=1)
    p = 1.0e3
    bcs = hm.clamp(d, where=hm.near("x", 0.0))
    by_pressure = hm.solve(d, _material(d), hm.pressure(d, hm.from_cells(d, np.where(bottom, p, 0.0))), bcs)
    t = np.zeros((d.n_cells, 3))
    t[bottom, 2] = -p                                  # outward normal of the bottom is -z
    by_traction = hm.solve(d, _material(d), hm.traction(d, hm.from_cells(d, t)), bcs)
    assert _compliance(by_pressure) > 0.0
    assert _compliance(by_pressure) == pytest.approx(_compliance(by_traction), rel=1e-10)


# -- penalty BCs on junction facets ----------------------------------------------

def _at_junction(x):
    return np.isclose(x[0], 0.0) & np.isclose(x[2], 0.0)


def _fin_solve(bcs_for):
    d = hm.ShellDomain(_build(*_fins()))
    loads = hm.traction(d, np.array([0.0, 0.0, -1.0e3]) + np.array([2.0e2, 0.0, 0.0]))
    return hm.solve(d, _material(d), loads, bcs_for(d))


def test_junction_facets_are_all_three_cell():
    """Guard the fixture: every facet the junction clamp selects joins three cells."""
    import dolfinx

    mesh = _build(*_fins())
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    f2c = mesh.topology.connectivity(tdim - 1, tdim)
    facets = dolfinx.mesh.locate_entities(mesh, tdim - 1, _at_junction)
    assert facets.size == 4
    assert {len(f2c.links(f)) for f in facets} == {3}


def test_penalty_clamp_on_junction_matches_strong(recorder):
    strong = _fin_solve(lambda d: hm.clamp(d, where=_at_junction, method="strong"))
    penalty = _fin_solve(lambda d: hm.clamp(d, where=_at_junction))
    ws, wp = (np.abs(s.disp_solid.value).max() for s in (strong, penalty))
    print(f"\nmax |u|  strong={ws:.6e}  penalty={wp:.6e}")
    assert ws < 1.0              # bounded: a mechanism gives ~1e9 here
    assert wp == pytest.approx(ws, rel=5e-3)
    assert _compliance(penalty) == pytest.approx(_compliance(strong), rel=5e-3)


def test_merged_penalty_regions_on_junction(recorder):
    """Two disjoint penalty regions on the junction (the shared-measure path) act
    like one clamp over both."""
    one = _fin_solve(lambda d: hm.clamp(d, where=_at_junction))

    def halves(d):
        lo = hm.clamp(d, where=lambda x: _at_junction(x) & (x[1] <= 0.5 + 1e-12))
        hi = hm.clamp(d, where=lambda x: _at_junction(x) & (x[1] >= 0.5 - 1e-12))
        return lo + hi

    two = _fin_solve(halves)
    assert len(halves(two.domain).penalty_terms) == 2
    assert np.abs(two.disp_solid.value).max() < 1.0
    assert _compliance(two) == pytest.approx(_compliance(one), rel=1e-10)


def test_penalty_clamp_on_junction_is_differentiable():
    """Thickness adjoint through a junction-clamped solve matches a central difference."""
    d = hm.ShellDomain(_build(*_fins()))
    bcs = hm.clamp(d, where=_at_junction)

    def run(scale, grad):
        rec = csdl.Recorder(inline=True)
        rec.start()
        s = csdl.Variable(value=float(scale), name="s")
        t = s * csdl.Variable(value=H * np.ones(d.n_cells))
        mat = hm.isotropic(d, E=E, nu=NU, thickness=hm.from_cells(d, t), density=RHO)
        c = hm.compliance(hm.solve(d, mat, hm.traction(d, np.array([2.0e2, 0.0, -1.0e3])), bcs))
        val = float(np.ravel(c.value)[0])
        g = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([c], [s])[c, s])[0]) if grad else None
        rec.stop()
        return val, g

    _, ana = run(1.0, True)
    eps = 1e-4                     # 1e-6 is round-off dominated (same on the strong path)
    fd = (run(1.0 + eps, False)[0] - run(1.0 - eps, False)[0]) / (2 * eps)
    assert ana == pytest.approx(fd, rel=1e-5)
