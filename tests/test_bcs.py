"""Strong (Dirichlet) boundary-condition path: ``hm.clamp(..., method="strong")``."""

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit
import hermit.bcs as hbc
import hermit.loads as hld
import hermit.material as hmat
import hermit.outputs as out
from hermit.domain import ShellDomain
from hermit._solve import solve
from conftest import clamped_at_x0


def _solve(plate_mesh, ref, method, thickness=None):
    nn = int(ref["n_nodes"])
    E, nu, rd, h = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val", "h_val"))
    pz = float(ref["pressure_z"])
    dom = ShellDomain(plate_mesh, element="CG2CG1")
    t = h * np.ones(nn) if thickness is None else hermit.from_nodal(dom, thickness)
    mat = hmat.isotropic(dom, E=E*np.ones(nn), nu=nu*np.ones(nn),
                         thickness=t, density=rd*np.ones(nn),
                         constitutive_space=("Lagrange", 1))
    return solve(dom, mat, hld.pressure(dom, pz),
                 hbc.clamp(dom, where=clamped_at_x0, method=method))


def test_strong_bc_matches_euler_bernoulli_and_penalty(plate_mesh, cantilever_ref, recorder):
    eb = float(cantilever_ref["eb_tip_deflection"])
    strong = _solve(plate_mesh, cantilever_ref, method="strong")
    penalty = _solve(plate_mesh, cantilever_ref, method="penalty")

    ws = np.abs(strong.disp_solid.value).max()
    wp = np.abs(penalty.disp_solid.value).max()
    print(f"\ntip w  strong={ws:.6e}  penalty={wp:.6e}  EB={eb:.6e}")
    assert ws == pytest.approx(eb, rel=5e-3)
    assert ws == pytest.approx(wp, rel=5e-3)
    assert float(np.ravel(out.compliance(strong).value)[0]) == pytest.approx(
        float(np.ravel(out.compliance(penalty).value)[0]), rel=5e-3)


def test_strong_bc_adjoint_wrt_thickness(plate_mesh, cantilever_ref):
    h, nn = float(cantilever_ref["h_val"]), int(cantilever_ref["n_nodes"])

    def run(scale, want_grad):
        rec = csdl.Recorder(inline=True); rec.start()
        s = csdl.Variable(value=float(scale), name="s")
        t = s * csdl.Variable(value=h * np.ones(nn))
        state = _solve(plate_mesh, cantilever_ref, method="strong", thickness=t)
        o = out.compliance(state)
        val = float(np.ravel(o.value)[0])
        g = None
        if want_grad:
            g = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([o], [s])[o, s])[0])
        rec.stop()
        return val, g

    _, ana = run(1.0, True)
    d = 1e-4
    vp, _ = run(1.0 + d, False)
    vm, _ = run(1.0 - d, False)
    cd = (vp - vm) / (2 * d)
    rel = abs(ana - cd) / abs(cd)
    print(f"\nstrong-BC d(compliance)/ds : analytic={ana:.6e}  cd={cd:.6e}  rel={rel:.2e}")
    assert rel < 2e-4
