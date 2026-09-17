"""Regression gates for selected exterior-facet (line) loads.

The moment test deliberately keeps the incorrect equal vertex split alongside the
edge load: it fails by O(1e-3), so the near-exact assertion cannot pass merely
because both paths accidentally use the same lumped RHS.
"""

import pathlib
import sys

import csdl_alpha as csdl
import numpy as np
import pytest
import ufl
from dolfinx.fem import assemble_scalar, form

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples" / "verification"))
from _geometry import rect_plate  # noqa: E402


L, WIDTH, THICKNESS, E, MOMENT = 10.0, 2.0, 0.2, 4.32e8, 1.0
I = WIDTH * THICKNESS**3 / 12.0
THETA_REFERENCE = MOMENT * L / (E * I)


def _tip_rotation(load, *, n=8):
    domain = hm.ShellDomain(rect_plate(L, WIDTH, n, n // 2, cell="quad"), element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=0.0, thickness=THICKNESS, density=1.0)
    state = hm.solve(domain, material, load(domain), hm.clamp(domain, where=hm.near("x", 0.0)))
    node = np.argmin(np.linalg.norm(domain.node_coords - [L, WIDTH / 2, 0.0], axis=1))
    return abs(float(hm.nodal_rotation(state).value.reshape(-1, 3)[node, 1]))


def test_edge_moment_is_consistent_while_equal_point_split_is_not():
    """Constant curvature is represented exactly by CG2/CG1 on this flat plate."""
    rec = csdl.Recorder(inline=True); rec.start()

    edge = _tip_rotation(lambda d: hm.edge_moment(
        d, [0.0, MOMENT / WIDTH, 0.0], where=hm.near("x", L)))

    def lumped(d):
        nodes = np.flatnonzero(np.isclose(d.node_coords[:, 0], L))
        return sum((hm.point_load(d, at=d.node_coords[k], moment=[0.0, MOMENT / len(nodes), 0.0])
                    for k in nodes), start=hm.Loads(d))

    point = _tip_rotation(lumped)
    rec.stop()
    edge_error = abs(edge / THETA_REFERENCE - 1.0)
    point_error = abs(point / THETA_REFERENCE - 1.0)
    print(f"\nedge relative error={edge_error:.3e}; equal-point relative error={point_error:.3e}")
    assert edge_error < 1e-8
    # This is the falsification half of the gate: replacing edge_moment with the
    # statically equivalent equal split makes the preceding assertion fail.
    assert point_error > 1e-3


def test_edge_traction_resultant_matches_equivalent_area_traction():
    """On a L-by-W rectangle, q on its end has resultant qW = (q/L)(LW)."""
    rec = csdl.Recorder(inline=True); rec.start()
    domain = hm.ShellDomain(rect_plate(L, WIDTH, 4, 2, cell="quad"))
    load = hm.edge_traction(domain, [0.0, 0.0, 3.0], where=hm.near("x", L))
    edge = load.edge_traction_terms[0]
    # This form is intentionally a rigid virtual displacement (not a shell solve):
    # it anchors the new selected-ds integration to the established dx convention
    # in the one case where their work/resultants are analytically identical.
    q = 3.0
    got = assemble_scalar(form(q * edge.ds))
    want = assemble_scalar(form((q / L) * ufl.dx(domain=domain.mesh)))
    rec.stop()
    print(f"\nedge resultant={got:.12e}; equivalent-area resultant={want:.12e}")
    assert got == pytest.approx(want, rel=1e-13, abs=1e-13)


def test_edge_moment_coefficient_derivative_matches_finite_difference():
    def run(magnitude, derivative):
        rec = csdl.Recorder(inline=True); rec.start()
        domain = hm.ShellDomain(rect_plate(L, WIDTH, 6, 3, cell="quad"), element="CG2CG1")
        material = hm.isotropic(domain, E=E, nu=0.0, thickness=THICKNESS, density=1.0)
        m = csdl.Variable(value=np.array([0.0, magnitude / WIDTH, 0.0]), name="edge_m")
        state = hm.solve(domain, material, hm.edge_moment(domain, m, where=hm.near("x", L)),
                         hm.clamp(domain, where=hm.near("x", 0.0)))
        c = hm.compliance(state)
        value = float(c.value[0])
        grad = None
        if derivative:
            grad = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([c], [m])[c, m])[1]) / WIDTH
        rec.stop()
        return value, grad

    m0, step = MOMENT, 1.0e-3
    _, analytic = run(m0, True)
    plus, _ = run(m0 + step, False)
    minus, _ = run(m0 - step, False)
    fd = (plus - minus) / (2 * step)
    rel = abs(analytic - fd) / abs(fd)
    print(f"\nd(compliance)/d(M): adjoint={analytic:.8e}; fd={fd:.8e}; rel={rel:.3e}")
    assert rel < 1e-6


def test_prescribed_penalty_bc_and_edge_load_compose():
    """The seam where prescribed BCs (#8) and edge loads (#9) meet.

    A penalty term now carries both a prescribed target ``g`` and the located facet
    ids that let an edge-load measure share one exterior-facet MeshTags object --
    DOLFINx refuses independently tagged ``ds`` measures inside one compiled form.
    Each feature is gated on its own; nothing else exercises them in the same form.

    The problem is linear, so superposition is exact: solving with the prescribed
    target and the edge load together must equal the sum of solving with each
    alone. Dropping either one when both are present breaks it.
    """
    def run(bc_value, with_load):
        rec = csdl.Recorder(inline=True); rec.start()
        domain = hm.ShellDomain(rect_plate(L, WIDTH, 6, 3, cell="quad"), element="CG2CG1")
        material = hm.isotropic(domain, E=E, nu=0.0, thickness=THICKNESS, density=1.0)
        loads = (hm.edge_traction(domain, [0.0, 0.0, 0.5], where=hm.near("x", L))
                 if with_load else hm.Loads(domain))
        state = hm.solve(domain, material, loads,
                         hm.clamp(domain, where=hm.near("x", 0.0), value=bc_value))
        d = np.asarray(hm.displacement_field(state).coeffs.value).copy()
        rec.stop()
        return d

    v = np.array([0.002, -0.001, 0.003, 0.0, 0.0, 0.0])
    only_bc, only_load, both = run(v, False), run(0.0, True), run(v, True)
    mag_bc = float(np.max(np.abs(only_bc)))
    mag_load = float(np.max(np.abs(only_load)))
    scale = max(mag_bc, mag_load)
    residual = float(np.max(np.abs(both - (only_bc + only_load))))
    print(f"\n|bc-only|={mag_bc:.3e}  |load-only|={mag_load:.3e}  "
          f"superposition residual={residual:.3e}")
    # Guard against a vacuous pass: superposition of two null solutions is trivially
    # exact, so both contributions must actually move the shell.
    assert mag_bc > 1e-4 and mag_load > 1e-4
    assert residual < 1e-10 * scale


def test_edge_pressure_matches_equivalent_edge_traction_solve():
    """An edge pressure of p applies p along the shell normal. For a flat plate in the
    XY plane, the shell normal is exactly [0, 0, 1]. The solved displacement field
    must match an explicit edge_traction of [0, 0, p] to machine precision. This is
    a stronger gate than a virtual-displacement resultant because it exercises the
    live CellNormal inside the residual form assembly during the actual solve.
    """
    rec = csdl.Recorder(inline=True); rec.start()
    domain = hm.ShellDomain(rect_plate(L, WIDTH, 4, 2, cell="quad"))
    material = hm.isotropic(domain, E=E, nu=0.0, thickness=THICKNESS, density=1.0)
    bc = hm.clamp(domain, where=hm.near("x", 0.0))

    p = 3.0
    state_p = hm.solve(domain, material, hm.edge_pressure(domain, p, where=hm.near("x", L)), bc)
    disp_p = np.asarray(hm.displacement_field(state_p).coeffs.value).copy()

    state_t = hm.solve(domain, material, hm.edge_traction(domain, [0.0, 0.0, p], where=hm.near("x", L)), bc)
    disp_t = np.asarray(hm.displacement_field(state_t).coeffs.value).copy()
    rec.stop()

    diff = float(np.max(np.abs(disp_p - disp_t)))
    mag = float(np.max(np.abs(disp_t)))
    print(f"\n|disp_t|={mag:.3e} |disp_p - disp_t|={diff:.3e}")
    assert mag > 1e-4
    assert diff < 1e-13 * mag
