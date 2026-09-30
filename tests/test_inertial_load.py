"""``hm.inertial_load`` -- the load of the structure's own mass.

Gates: it reproduces the equivalent hand-built ``traction`` wherever one exists
(uniform, and the per-cell thickness the STW example used); its force and moment
resultants agree with ``hm.mass`` / ``hm.center_of_gravity`` on a curved panel with
a nodal thickness, where no single traction field is exact; it composes linearly
with other loads (including a second inertial term); and its adjoint derivatives
match finite differences in every input, the mesh coordinates included.
"""

import functools
import pathlib
import sys

import csdl_alpha as csdl
import numpy as np
import pytest
import ufl
from dolfinx.fem import assemble_scalar, form

import hermit as hm
from hermit._solve import _pde_for
from hermit.fenics import assembly as fa
from hermit.fenics.elastic_model import load_work

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples" / "verification"))
from _geometry import cylinder_sector, rect_plate  # noqa: E402


L, WIDTH, E, NU = 10.0, 2.0, 1.0e7, 0.3
G = np.array([0.0, 0.0, -9.81])


def _plate(nx=8, ny=4, cell="quad"):
    return hm.ShellDomain(rect_plate(L, WIDTH, nx, ny, cell=cell))


def _clamp(domain):
    return hm.clamp(domain, where=hm.near("x", 0.0))


def _disp(state):
    return np.asarray(state.disp_solid.value)


def test_uniform_inertial_load_equals_traction(recorder):
    domain = _plate()
    material = hm.isotropic(domain, E=E, nu=NU, thickness=0.2, density=3.0)
    got = hm.solve(domain, material, hm.inertial_load(domain, material, acceleration=G),
                   _clamp(domain))
    want = hm.solve(domain, material, hm.traction(domain, 3.0 * 0.2 * G), _clamp(domain))
    assert np.max(np.abs(_disp(got) - _disp(want))) <= 1e-12 * np.max(np.abs(_disp(want)))
    assert float(hm.compliance(got).value[0]) == pytest.approx(
        float(hm.compliance(want).value[0]), rel=1e-12)


def test_per_cell_thickness_equals_hand_built_traction(recorder):
    """The STW example's hand-built weight: DG0 thickness times a constant density."""
    domain = _plate()
    rng = np.random.default_rng(0)
    t_cells = 0.1 + 0.2 * rng.random(domain.n_cells)
    rho, n = 2780.0, 2.5
    material = hm.isotropic(domain, E=E, nu=NU, thickness=hm.from_cells(domain, t_cells),
                            density=rho)
    got = hm.solve(domain, material,
                   hm.inertial_load(domain, material, acceleration=n * G), _clamp(domain))
    per_cell = np.outer(t_cells, n * rho * G)
    want = hm.solve(domain, material, hm.traction(domain, hm.from_cells(domain, per_cell)),
                    _clamp(domain))
    assert np.max(np.abs(_disp(got) - _disp(want))) <= 1e-12 * np.max(np.abs(_disp(want)))


def _resultants(domain, loads):
    """Force and moment (about the origin) of the one inertial term in ``loads``,
    as the virtual work of that term on the six rigid-body modes."""
    pde = _pde_for(domain)
    (term,) = loads.inertial_terms
    funcs = []
    for name, space, coeffs in term.coefficients("resultant_check"):
        f = pde.coefficient(name, space)
        fa.set_array(f, np.asarray(coeffs.value))
        funcs.append(f)
    X = ufl.SpatialCoordinate(domain.mesh)
    zero = ufl.as_vector([0.0, 0.0, 0.0])
    out = []
    for k in range(3):
        e = ufl.as_vector([1.0 if i == k else 0.0 for i in range(3)])
        out.append(assemble_scalar(form(load_work("inertial", tuple(funcs), e, zero, None, ufl.dx))))
    for k in range(3):
        e = ufl.as_vector([1.0 if i == k else 0.0 for i in range(3)])
        out.append(assemble_scalar(form(load_work("inertial", tuple(funcs), ufl.cross(e, X), e,
                                                  None, ufl.dx))))
    return np.array(out[:3]), np.array(out[3:])


def test_resultants_match_mass_and_center_of_gravity(recorder):
    """Curved panel, nodal (P1) thickness: ``F = m a + alpha x m (cg - p)``, and with
    no angular part ``M_origin = m cg x a``."""
    domain = hm.ShellDomain(cylinder_sector(radius=5.0, length=8.0, half_angle=40.0,
                                            nx=8, nt=8))
    x = domain.node_coords
    thickness = hm.from_nodal(domain, 0.05 + 0.01 * x[:, 0] + 0.02 * x[:, 1] ** 2)
    material = hm.isotropic(domain, E=E, nu=NU, thickness=thickness, density=7.0)
    state = hm.ShellState(domain, None, material, hm.Loads(domain), _clamp(domain),
                          disp_solid=csdl.Variable(value=np.zeros(1)))
    m = float(hm.mass(state).value[0])
    cg = np.asarray(hm.center_of_gravity(state).value)

    a = np.array([1.0, -2.0, 3.0])
    alpha = np.array([0.3, 0.5, -0.7])
    about = np.array([4.0, 1.0, 2.0])
    F, _ = _resultants(domain, hm.inertial_load(domain, material, acceleration=a,
                                                angular_acceleration=alpha, about=about))
    np.testing.assert_allclose(F, m * a + np.cross(alpha, m * (cg - about)), rtol=1e-12)

    F, M = _resultants(domain, hm.inertial_load(domain, material, acceleration=a))
    np.testing.assert_allclose(F, m * a, rtol=1e-12)
    np.testing.assert_allclose(M, np.cross(m * cg, a), rtol=1e-12)


def test_composes_linearly_with_other_loads(recorder):
    """Two inertial terms plus a pressure: the solve is the sum of the parts."""
    domain = _plate()
    material = hm.isotropic(domain, E=E, nu=NU, thickness=0.2, density=3.0)
    bcs = _clamp(domain)
    parts = [hm.pressure(domain, 50.0),
             hm.inertial_load(domain, material, acceleration=G),
             hm.inertial_load(domain, material, acceleration=[1.0, 0.0, 4.0],
                              angular_acceleration=[0.0, 0.2, 0.1], about=[L, 0.0, 0.0])]
    together = hm.solve(domain, material, parts[0] + parts[1] + parts[2], bcs)
    separate = sum(_disp(hm.solve(domain, material, p, bcs)) for p in parts)
    assert np.max(np.abs(_disp(together) - separate)) <= 1e-10 * np.max(np.abs(separate))
    # compliance must count the inertial work: for a linear solve it is twice the
    # stored energy, which the load terms never enter
    assert float(hm.compliance(together).value[0]) == pytest.approx(
        2.0 * float(hm.elastic_energy(together).value[0]), rel=1e-9)


@functools.cache
def _fd_mesh():
    """One mesh for every finite-difference evaluation -- the perturbations are of
    the inputs, not the mesh, so there is nothing to rebuild."""
    return rect_plate(L, WIDTH, 6, 3, cell="triangle")


def _compliance_and_gradients(params, *, derivative):
    """Compliance of a cantilever under pressure plus an inertial load with angular
    part, and its derivatives in every inertial input and a shape perturbation."""
    rec = csdl.Recorder(inline=True); rec.start()
    domain = hm.ShellDomain(_fd_mesh())
    x = domain.node_coords
    t_scale = csdl.Variable(value=params["t"], name="t")
    rho = csdl.Variable(value=params["rho"], name="rho")
    a = csdl.Variable(value=params["a"], name="a")
    alpha = csdl.Variable(value=params["alpha"], name="alpha")
    s = csdl.Variable(value=params["s"], name="s")
    bump = np.zeros_like(x)
    bump[:, 2] = np.sin(np.pi * x[:, 0] / L) * x[:, 1] / WIDTH
    geometry = hm.geometry(domain, node_disp=s * csdl.Variable(value=bump))
    t_nodal = t_scale * csdl.Variable(value=0.1 + 0.01 * x[:, 0])
    material = hm.isotropic(domain, E=E, nu=NU, thickness=hm.from_nodal(domain, t_nodal),
                            density=rho)
    loads = hm.pressure(domain, 20.0) + hm.inertial_load(
        domain, material, acceleration=a, angular_acceleration=alpha, about=[1.0, 0.5, 0.0])
    state = hm.solve(domain, material, loads, _clamp(domain), geometry=geometry)
    c = hm.compliance(state)
    value = float(c.value[0])
    grads = None
    if derivative:
        wrt = [t_scale, rho, a, alpha, s]
        totals = csdl.experimental.PySimulator(rec).compute_totals([c], wrt)
        grads = {v.name: np.ravel(np.asarray(totals[c, v])) for v in wrt}
    rec.stop()
    return value, grads


def test_derivatives_match_finite_differences():
    base = dict(t=1.0, rho=3.0, a=np.array([0.5, -1.0, -9.81]),
                alpha=np.array([0.2, -0.1, 0.3]), s=0.05)
    _, grads = _compliance_and_gradients(base, derivative=True)
    # Steps are large on purpose. The solve's round-off (~1e-9 relative in the
    # compliance, with or without the inertial term) swamps a 1e-6 difference
    # quotient; the shape derivative is small against the compliance on this warped
    # plate, so it needs a larger step again.
    for name, h in (("t", 1e-4), ("rho", 3e-4), ("a", 1e-3), ("alpha", 1e-4), ("s", 1e-3)):
        value = np.atleast_1d(base[name]).astype(float)
        fd = np.empty(value.size)
        for k in range(value.size):
            plus, minus = value.copy(), value.copy()
            plus[k] += h; minus[k] -= h
            fp, _ = _compliance_and_gradients({**base, name: plus if value.size > 1 else plus[0]},
                                              derivative=False)
            fm, _ = _compliance_and_gradients({**base, name: minus if value.size > 1 else minus[0]},
                                              derivative=False)
            fd[k] = (fp - fm) / (2 * h)
        print(f"\nd compliance / d {name}: adjoint={grads[name]} fd={fd}")
        np.testing.assert_allclose(grads[name], fd, rtol=1e-5, atol=1e-8 * np.max(np.abs(fd)))


def test_rejects_material_from_another_domain(recorder):
    domain, other = _plate(), _plate()
    material = hm.isotropic(other, E=E, nu=NU, thickness=0.2, density=3.0)
    with pytest.raises(ValueError, match="same|exact"):
        hm.inertial_load(domain, material, acceleration=G)


def test_rejects_wrong_acceleration_shape(recorder):
    domain = _plate()
    material = hm.isotropic(domain, E=E, nu=NU, thickness=0.2, density=3.0)
    with pytest.raises(ValueError, match="3 components"):
        hm.inertial_load(domain, material, acceleration=[0.0, -9.81])
    with pytest.raises(ValueError, match="3 components"):
        hm.inertial_load(domain, material, acceleration=G, angular_acceleration=np.zeros(6))


def test_thickness_only_material_is_accepted(recorder):
    """A surrogate needs no stiffness to describe the load."""
    domain = _plate()
    material = hm.thickness_only(domain, thickness=0.2, density=3.0)
    loads = hm.inertial_load(domain, material, acceleration=G)
    assert len(loads.inertial_terms) == 1
