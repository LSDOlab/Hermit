"""``ShellDomain(quadrature_degree=...)`` -- the explicit degree every hand-pinned
integral uses.
"""

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit.bcs as hbc
import hermit.loads as hld
import hermit.material as hmat
import hermit.outputs as out
from hermit._solve import _pde_for, solve
from hermit.domain import ShellDomain
from hermit.fenics.ops import ShellFieldFormsOp, strain_fields
from hermit._laminate import Layup
from conftest import clamped_at_x0


def _ud():
    from caddee_materials import TransverseMaterial

    m = TransverseMaterial(name="ud", EA=138e9, ET=10e9, vA=0.34, vT=0.4, GA=7e9, density=1.6e3)
    m.set_strength(F1t=1500e6, F1c=1200e6, F2t=50e6, F2c=200e6, F12=70e6, F23=40e6)
    return m


def _layup(angles_deg, total_h=0.02):
    n = len(angles_deg)
    a = csdl.Variable(value=np.radians(np.asarray(angles_deg, dtype=float)), name="ply_angles")
    return Layup(_ud(), a, np.full(n, total_h / n), num_plies=n)


def _qdeg(measure):
    """The quadrature_degree a UFL measure was compiled with."""
    return measure.metadata()["quadrature_degree"]


def test_default_is_four_and_reaches_every_pinned_measure(plate_mesh, recorder):
    dom = ShellDomain(plate_mesh, element="CG2CG1")
    assert dom.quadrature_degree == 4
    assert _qdeg(hbc.clamp(dom, where=clamped_at_x0).penalty_terms[0].dss) == 4
    E, nu, h, rho = 4.32e8, 0.0, 0.2, 1.0
    mat = hmat.isotropic(dom, E=E * np.ones(dom.n_nodes), nu=nu * np.ones(dom.n_nodes),
                         thickness=h * np.ones(dom.n_nodes), density=rho * np.ones(dom.n_nodes),
                         constitutive_space=("Lagrange", 1))
    state = solve(dom, mat, hld.pressure(dom, 2.0), hbc.clamp(dom, where=clamped_at_x0))
    assert _qdeg(out._stress_measure(state, None)) == 4


@pytest.mark.parametrize("qdeg", [2, 4, 8])
def test_setting_reaches_bc_measures(plate_mesh, qdeg, recorder):
    dom = ShellDomain(plate_mesh, element="CG2CG1", quadrature_degree=qdeg)
    bcs = hbc.clamp(dom, where=clamped_at_x0)
    assert _qdeg(bcs.penalty_terms[0].dss) == qdeg
    assert _qdeg(bcs.penalty_terms[0].dSS) == qdeg


def test_midpoint_stays_at_zero_regardless(plate_mesh, recorder):
    """degree 0 for "midpoint" means *one centroid point* -- the definition of the
    method, not an accuracy choice -- so the setting must not touch it."""
    dom = ShellDomain(plate_mesh, element="CG2CG1", quadrature_degree=8)
    pde = _pde_for(dom)

    def form_qdeg(method):
        op = ShellFieldFormsOp(pde, strain_fields(pde, space=("DG", 0), method=method),
                               quadrature_degree=dom.quadrature_degree)
        return op._L["mid_strain"].integrals()[0].metadata()["quadrature_degree"]

    assert form_qdeg("midpoint") == 0
    assert form_qdeg("average") == 8


def _oriented_compliance(mesh, qdeg):
    """Compliance of an oriented composite -- the path whose energy measure the
    setting pins (the unoriented path keeps UFL's auto-estimated degree)."""
    rec = csdl.Recorder(inline=True); rec.start()
    dom = ShellDomain(mesh, element="CG2CG1", quadrature_degree=qdeg)
    layup = _layup([10.0, -25.0, 55.0])
    mat = hmat.laminate(dom, layup=layup, density=1.6e3 * np.ones(dom.n_cells),
                        orientation=hmat.fiber_angle(dom, 0.3),
                        constitutive_space=("DG", 0))
    state = solve(dom, mat, hld.pressure(dom, 1.0e3), hbc.clamp(dom, where=clamped_at_x0))
    c = float(np.ravel(out.compliance(state).value)[0])
    rec.stop()
    return c


def test_degree_two_is_coarse_but_four_and_eight_agree(plate_mesh):
    """The documented convergence behaviour of the oriented shell integrand: 2 is
    measurably off the plateau, 4 and 8 are both on it (module comment in
    hermit/fenics/elastic_model.py -- 2 is ~1% off, 4..24 agree to ~1e-8)."""
    c2, c4, c8 = (_oriented_compliance(plate_mesh, q) for q in (2, 4, 8))
    rel_2_4 = abs(c2 - c4) / abs(c4)
    rel_4_8 = abs(c4 - c8) / abs(c8)
    print(f"\nc2={c2:.10g} c4={c4:.10g} c8={c8:.10g}  rel(2,4)={rel_2_4:.3e} rel(4,8)={rel_4_8:.3e}")
    assert rel_2_4 > 1e-4, "degree 2 should be measurably coarser than the plateau"
    assert rel_4_8 < 1e-6, "degrees 4 and 8 should both be on the converged plateau"
