"""``hm.inertia_relief`` -- the balancing inertial load for a free structure.

The balance gate is independent of the implementation: each rigid-body mode is built
as a ``domain.W`` vector from the dof coordinates, and ``hm.compliance`` of a state
holding that vector is the work of the *discrete* loads on it -- distributed, edge,
inertial and point terms alike. ``loads + relief`` must do no work on any of them.
"""

import pathlib
import sys

import csdl_alpha as csdl
import numpy as np
import pytest

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples" / "verification"))
from _geometry import cylinder_sector, hyperbolic_paraboloid, rect_plate  # noqa: E402


E, NU, RHO, T = 1.0e7, 0.3, 3.0, 0.2
ALL_DOFS = ("ux", "uy", "uz", "rx", "ry", "rz")


def _rigid_mode_vectors(domain):
    """``(ndof_W, 6)``: translations, then rotations about the origin."""
    W = domain.W
    ndof = W.dofmap.index_map.size_local * W.dofmap.index_map_bs
    Vu, u_dofs = W.sub(0).collapse()
    _, t_dofs = W.sub(1).collapse()
    u_dofs = np.asarray(u_dofs).reshape(-1, 3)
    t_dofs = np.asarray(t_dofs).reshape(-1, 3)
    x = Vu.tabulate_dof_coordinates()
    phi = np.zeros((ndof, 6))
    for k in range(3):
        e = np.eye(3)[k]
        phi[u_dofs[:, k], k] = 1.0
        phi[u_dofs.reshape(-1), 3 + k] = np.cross(e, x).reshape(-1)
        phi[t_dofs[:, k], 3 + k] = 1.0
    return phi


def _work_on_rigid_modes(domain, material, loads):
    """The virtual work of ``loads`` on each rigid mode, through ``hm.compliance``."""
    out = []
    for mode in _rigid_mode_vectors(domain).T:
        state = hm.ShellState(domain, None, material, loads, hm.Loads(domain),
                              disp_solid=csdl.Variable(value=mode))
        out.append(float(hm.compliance(state).value[0]))
    return np.array(out)


def _unbalanced_loads(domain, material):
    """Pressure on part of the surface, an off-centre point force and moment, self
    weight, and an edge moment: every kind of term, with no symmetry."""
    x = domain.cell_centroids
    xmax = domain.node_coords[:, 0].max()
    return (hm.pressure(domain, hm.from_cells(domain, np.where(x[:, 0] < 0.4 * xmax, 5.0, 0.0)))
            + hm.point_load(domain, at=domain.node_coords[len(domain.node_coords) // 3],
                            force=[1.0, -2.0, 30.0], moment=[0.0, 5.0, 1.0])
            + hm.inertial_load(domain, material, acceleration=[0.0, 0.0, -9.81])
            + hm.edge_moment(domain, [0.0, 2.0, 0.0], where=hm.near("x", xmax)))


@pytest.mark.parametrize("mesh", [
    pytest.param(lambda: rect_plate(10.0, 2.0, 10, 4, cell="triangle"), id="flat-triangles"),
    pytest.param(lambda: rect_plate(10.0, 2.0, 10, 4, cell="quad"), id="flat-quads"),
    pytest.param(lambda: cylinder_sector(radius=5.0, length=8.0, half_angle=40.0, nx=8, nt=8),
                 id="curved-triangles"),
])
def test_relief_balances_every_rigid_mode(recorder, mesh):
    domain = hm.ShellDomain(mesh())
    x = domain.node_coords
    material = hm.isotropic(domain, E=E, nu=NU, density=RHO,
                            thickness=hm.from_nodal(domain, T + 0.01 * x[:, 0]))
    loads = _unbalanced_loads(domain, material)
    relief, _ = hm.inertia_relief(domain, material, loads, about=[1.0, -2.0, 0.5])
    before = _work_on_rigid_modes(domain, material, loads)
    after = _work_on_rigid_modes(domain, material, loads + relief)
    print(f"\nresultants before {before}\nafter  {after}")
    assert np.max(np.abs(after)) <= 1e-12 * np.max(np.abs(before))
    # the trap the docstring warns about: the relief term alone is not balanced
    assert np.max(np.abs(_work_on_rigid_modes(domain, material, relief))) > 0.1 * np.max(np.abs(before))


def test_relief_balances_on_a_moved_mesh(recorder):
    """With a live geometry the balance must hold on the *moved* mesh. A point load
    rides its material point, so its moment arm moves with the nodes; freezing the
    arms at the reference geometry breaks this gate (while a finite-difference check
    of the same frozen model still passes -- both sides share the wrong model)."""
    domain = hm.ShellDomain(rect_plate(10.0, 2.0, 10, 4, cell="triangle"))
    x = domain.node_coords
    node_disp = np.zeros_like(x)
    node_disp[:, 0] = 0.4 * x[:, 0] * x[:, 1]
    node_disp[:, 2] = 0.3 * np.sin(np.pi * x[:, 0] / 10.0)
    geometry = hm.geometry(domain, node_disp=node_disp)
    material = hm.isotropic(domain, E=E, nu=NU, thickness=T, density=RHO)
    loads = _unbalanced_loads(domain, material)
    relief, _ = hm.inertia_relief(domain, material, loads, geometry=geometry)

    # rigid modes of the moved mesh, from DOLFINx's own dof coordinates there
    mesh = domain.mesh
    saved = mesh.geometry.x.copy()
    mesh.geometry.x[:] = (x + node_disp)[domain.node_input_idx]
    try:
        modes = _rigid_mode_vectors(domain)
    finally:
        mesh.geometry.x[:] = saved

    def work(l):
        return np.array([float(hm.compliance(hm.ShellState(
            domain, geometry, material, l, hm.Loads(domain),
            disp_solid=csdl.Variable(value=m))).value[0]) for m in modes.T])

    before, after = work(loads), work(loads + relief)
    print(f"\nmoved mesh: before {before}\nafter {after}")
    assert np.max(np.abs(after)) <= 1e-12 * np.max(np.abs(before))


def test_warped_quads_balance_to_quadrature_accuracy(recorder):
    """Non-affine cells: resultant and residual integrands get different automatic
    quadrature, so the balance is no longer at round-off -- but it stays small."""
    domain = hm.ShellDomain(hyperbolic_paraboloid(nx=8, ny=4))
    material = hm.isotropic(domain, E=E, nu=NU, density=RHO, thickness=T)
    loads = _unbalanced_loads(domain, material)
    relief, _ = hm.inertia_relief(domain, material, loads)
    before = _work_on_rigid_modes(domain, material, loads)
    after = _work_on_rigid_modes(domain, material, loads + relief)
    print(f"\nwarped: max |after| / max |before| = {np.max(np.abs(after)) / np.max(np.abs(before)):.2e}")
    assert np.max(np.abs(after)) <= 1e-6 * np.max(np.abs(before))


def test_gauge_location_changes_only_a_rigid_motion(recorder):
    domain = hm.ShellDomain(rect_plate(10.0, 2.0, 10, 4, cell="triangle"))
    material = hm.isotropic(domain, E=E, nu=NU, thickness=T, density=RHO)
    loads = _unbalanced_loads(domain, material)
    relief, _ = hm.inertia_relief(domain, material, loads)
    states = [hm.solve(domain, material, loads + relief, hm.gauge(domain, at=p, dofs=ALL_DOFS))
              for p in ([0.0, 0.0, 0.0], [10.0, 2.0, 0.0], [5.0, 1.0, 0.0])]
    energy = [float(hm.elastic_energy(s).value[0]) for s in states]
    compliance = [float(hm.compliance(s).value[0]) for s in states]
    stress = [np.asarray(hm.stress_field(s).coeffs.value) for s in states]
    np.testing.assert_allclose(energy, energy[0], rtol=1e-9)
    np.testing.assert_allclose(compliance, compliance[0], rtol=1e-9)
    for s in stress[1:]:
        np.testing.assert_allclose(s, stress[0], rtol=0, atol=1e-8 * np.max(np.abs(stress[0])))


def test_force_through_the_center_of_gravity(recorder):
    """A pure force at the centre of gravity: a = -F/m there, no angular part."""
    domain = hm.ShellDomain(rect_plate(10.0, 2.0, 10, 4, cell="quad"))
    material = hm.isotropic(domain, E=E, nu=NU, thickness=T, density=RHO)
    force = np.array([3.0, -1.0, 12.0])
    cg = np.array([5.0, 1.0, 0.0])
    _, accel = hm.inertia_relief(domain, material, hm.point_load(domain, at=cg, force=force),
                                 about=cg)
    mass = RHO * T * 10.0 * 2.0
    np.testing.assert_allclose(np.asarray(accel.value), np.r_[-force / mass, 0.0, 0.0, 0.0],
                               atol=1e-12)


def _relief_compliance(params, *, derivative):
    """Compliance of a free, relieved plate under pressure and a point load, with
    its derivatives in the thickness scale, pressure and a shape perturbation."""
    rec = csdl.Recorder(inline=True); rec.start()
    domain = hm.ShellDomain(rect_plate(10.0, 2.0, 8, 4, cell="triangle"))
    x = domain.node_coords
    t = csdl.Variable(value=params["t"], name="t")
    p = csdl.Variable(value=params["p"], name="p")
    s = csdl.Variable(value=params["s"], name="s")
    bump = np.zeros_like(x)
    bump[:, 2] = np.sin(np.pi * x[:, 0] / 10.0) * x[:, 1] / 2.0
    bump[:, 0] = 0.3 * x[:, 1] * x[:, 0] / 10.0
    geometry = hm.geometry(domain, node_disp=s * csdl.Variable(value=bump))
    thickness = t * csdl.Variable(value=T + 0.01 * x[:, 0])
    material = hm.isotropic(domain, E=E, nu=NU, thickness=hm.from_nodal(domain, thickness),
                            density=RHO)
    loads = (hm.pressure(domain, hm.from_cells(domain, (domain.cell_centroids[:, 0] < 4.0) * p))
             + hm.point_load(domain, at=[8.0, 1.5, 0.0], force=[0.0, 2.0, 20.0]))
    relief, _ = hm.inertia_relief(domain, material, loads, geometry=geometry)
    state = hm.solve(domain, material, loads + relief,
                     hm.gauge(domain, at=[0.0, 0.0, 0.0], dofs=ALL_DOFS), geometry=geometry)
    c = hm.compliance(state)
    value = float(c.value[0])
    grads = None
    if derivative:
        totals = csdl.experimental.PySimulator(rec).compute_totals([c], [t, p, s])
        grads = {v.name: float(np.ravel(totals[c, v])[0]) for v in (t, p, s)}
    rec.stop()
    return value, grads


def test_derivatives_match_finite_differences():
    base = dict(t=1.0, p=5.0, s=0.05)
    _, grads = _relief_compliance(base, derivative=True)
    # steps sized against the solve's round-off; see test_inertial_load.py
    for name, h in (("t", 1e-4), ("p", 1e-3), ("s", 1e-3)):
        fp, _ = _relief_compliance({**base, name: base[name] + h}, derivative=False)
        fm, _ = _relief_compliance({**base, name: base[name] - h}, derivative=False)
        fd = (fp - fm) / (2 * h)
        print(f"\nd compliance / d {name}: adjoint={grads[name]:.10e} fd={fd:.10e}")
        assert grads[name] == pytest.approx(fd, rel=1e-5)


def test_rejects_inputs_from_another_domain(recorder):
    domain, other = (hm.ShellDomain(rect_plate(10.0, 2.0, 4, 2)) for _ in range(2))
    material = hm.isotropic(domain, E=E, nu=NU, thickness=T, density=RHO)
    with pytest.raises(ValueError, match="loads"):
        hm.inertia_relief(domain, material, hm.pressure(other, 1.0))
    with pytest.raises(ValueError, match="material"):
        hm.inertia_relief(other, material, hm.pressure(other, 1.0))
