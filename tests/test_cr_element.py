"""CG2xCR1 element (Crouzeix-Raviart rotation) for triangle meshes."""

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit.material as hmat
import hermit.loads as hld
import hermit.bcs as hbc
import hermit.outputs as out
from hermit.domain import ShellDomain
from hermit._solve import solve
from conftest import clamped_at_x0

E, NU, H, RHO, PZ = 6.8e10, 0.35, 0.05, 2700.0, -30.0


def _compliance_and_tip(mesh, element):
    domain = ShellDomain(mesh, element=element)
    mat = hmat.isotropic(domain, E=E, nu=NU, thickness=H, density=RHO,
                         constitutive_space=("Lagrange", 1))
    state = solve(domain, mat, hld.pressure(domain, PZ),
                  hbc.clamp(domain, where=clamped_at_x0))
    return (float(np.ravel(out.compliance(state).value)[0]),
            float(np.abs(out.nodal_displacement(state).value[:, 2]).max()))


def test_cr_runs_and_tracks_cg(tri_mesh, recorder):
    c_cr, w_cr = _compliance_and_tip(tri_mesh, "CG2CR1")
    c_cg, w_cg = _compliance_and_tip(tri_mesh, "CG2CG1")
    print(f"\ntip |w|  CR={w_cr:.5e} CG={w_cg:.5e}   compliance CR={c_cr:.5e} CG={c_cg:.5e}")
    assert w_cr > 1e-4
    # same plate, same load; CR is a touch softer but the same ballpark
    assert abs(w_cr - w_cg) / w_cg < 0.15
    assert abs(c_cr - c_cg) / c_cg < 0.15


def test_cr_rejects_quad_mesh(plate_mesh):
    with pytest.raises(ValueError, match="simplex"):
        ShellDomain(plate_mesh, element="CG2CR1")


def test_cr_nodal_rotation_errors_clearly(tri_mesh, recorder):
    domain = ShellDomain(tri_mesh, element="CG2CR1")
    mat = hmat.isotropic(domain, E=E, nu=NU, thickness=H, density=RHO,
                         constitutive_space=("Lagrange", 1))
    state = solve(domain, mat, hld.pressure(domain, PZ),
                  hbc.clamp(domain, where=clamped_at_x0))
    with pytest.raises(RuntimeError, match="edge midpoints|nodal extraction"):
        out.nodal_rotation(state)


def test_cr_rotation_field_is_evaluable(tri_mesh, recorder):
    """state.rotation() (the native CR field) still works -- only nodal extraction doesn't."""
    domain = ShellDomain(tri_mesh, element="CG2CR1")
    mat = hmat.isotropic(domain, E=E, nu=NU, thickness=H, density=RHO,
                         constitutive_space=("Lagrange", 1))
    state = solve(domain, mat, hld.pressure(domain, PZ),
                  hbc.clamp(domain, where=clamped_at_x0))
    theta = out.rotation_field(state).values     # (n_cells, 3) at centroids, numpy
    assert theta.shape == (tri_mesh.topology.index_map(2).size_local, 3)
    assert np.isfinite(theta).all() and np.abs(theta).max() > 1e-6


def test_cr_thickness_derivative(tri_mesh):
    def run(hval, grad):
        rec = csdl.Recorder(inline=True); rec.start()
        t = csdl.Variable(value=np.atleast_1d(hval), name="t")
        domain = ShellDomain(tri_mesh, element="CG2CR1")
        mat = hmat.isotropic(domain, E=E, nu=NU, thickness=t, density=RHO,
                             constitutive_space=("Lagrange", 1))
        state = solve(domain, mat, hld.pressure(domain, PZ),
                      hbc.clamp(domain, where=clamped_at_x0))
        o = out.compliance(state)
        val = float(np.ravel(o.value)[0])
        g = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([o], [t])[o, t])[0]) if grad else None
        rec.stop()
        return val, g

    _, ana = run(H, True)
    d = 1e-4 * H
    vp, _ = run(H + d, False)
    vm, _ = run(H - d, False)
    fd = (vp - vm) / (2 * d)
    rel = abs(ana - fd) / abs(fd)
    print(f"\nCR d(compliance)/dt: ana={ana:+.6e} fd={fd:+.6e} rel={rel:.2e}")
    assert rel < 1e-4
