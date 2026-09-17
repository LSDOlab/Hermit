"""Adjoints of the scalar postprocess outputs, driven by ``ShellDomain`` /
``hm.isotropic`` / ``hm.pressure`` / ``hm.clamp`` / ``hm.outputs``.

The forward values are gated by
``tests/test_outputs_api.py::test_scalar_outputs_match_reference``; what is unique to
this file are the two thickness adjoints below.
"""

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit.bcs as hbc
import hermit.loads as hld
import hermit.material as hmat
import hermit.outputs as out
from hermit.domain import ShellDomain
from hermit._solve import solve
from conftest import clamped_at_x0


def _state(plate_mesh, ref, thickness):
    E, nu, rho = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val"))
    pz, nn = float(ref["pressure_z"]), int(ref["n_nodes"])
    domain = ShellDomain(plate_mesh, element="CG2CG1")
    material = hmat.isotropic(domain, E=E * np.ones(nn), nu=nu * np.ones(nn),
                              thickness=thickness, density=rho * np.ones(nn),
                              constitutive_space=("Lagrange", 1))
    return solve(domain, material, hld.pressure(domain, pz),
                 hbc.clamp(domain, where=clamped_at_x0))


def _total(plate_mesh, ref, out_name, scale_value):
    h, nn = float(ref["h_val"]), int(ref["n_nodes"])
    rec = csdl.Recorder(inline=True); rec.start()
    scale = csdl.Variable(value=float(scale_value), name="s")
    t = scale * csdl.Variable(value=h * np.ones(nn))
    o = getattr(out, out_name)(_state(plate_mesh, ref, t))
    val = float(np.ravel(o.value)[0])
    ana = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([o], [scale])[o, scale])[0])
    rec.stop()
    return val, ana


@pytest.mark.parametrize("name", ["compliance", "elastic_energy", "mass"])
def test_scalar_output_adjoint_wrt_thickness(plate_mesh, cantilever_ref, name):
    _, ana = _total(plate_mesh, cantilever_ref, name, 1.0)
    d = 1e-4
    vp, _ = _total(plate_mesh, cantilever_ref, name, 1.0 + d)
    vm, _ = _total(plate_mesh, cantilever_ref, name, 1.0 - d)
    cd = (vp - vm) / (2 * d)
    rel = abs(ana - cd) / max(abs(cd), 1e-30)
    print(f"\n{name}: analytic={ana:.8e}  central-diff={cd:.8e}  rel err={rel:.2e}")
    assert rel < 2e-4


def _pnorm_total(plate_mesh, ref, scale_value):
    h, nn = float(ref["h_val"]), int(ref["n_nodes"])
    rec = csdl.Recorder(inline=True); rec.start()
    scale = csdl.Variable(value=float(scale_value), name="s")
    t = scale * csdl.Variable(value=h * np.ones(nn))
    # rho=2, m=1e-3 keeps (m*vm)**rho well away from float underflow (the femo
    # defaults 1e-6/100 put it at 1e-186 for this stress scale -- see test docstring).
    o = out.pnorm_stress(_state(plate_mesh, ref, t), rho=2.0, m=1e-3)
    val = float(np.ravel(o.value)[0])
    ana = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([o], [scale])[o, scale])[0])
    rec.stop()
    return val, ana


def test_pnorm_stress_adjoint_wrt_thickness(plate_mesh, cantilever_ref):
    """The raw p-norm stress form (deps on disp, thickness, E, nu) and its adjoint.

    (The p-norm itself has no floor to hide behind. ``m`` is chosen so ``(m*vm)**rho``
    stays mid-range: with the femo defaults m=1e-6, rho=100 it is 1e-186 here, which
    the aggregate now warns about -- see hermit.csdl_helpers._warn_if_degenerate.)
    """
    v0, ana = _pnorm_total(plate_mesh, cantilever_ref, 1.0)
    assert v0 > 1e-6  # not on the underflow floor
    d = 1e-4
    vp, _ = _pnorm_total(plate_mesh, cantilever_ref, 1.0 + d)
    vm, _ = _pnorm_total(plate_mesh, cantilever_ref, 1.0 - d)
    cd = (vp - vm) / (2 * d)
    rel = abs(ana - cd) / abs(cd)
    print(f"\npnorm_stress: value={v0:.4e}  analytic={ana:.6e}  central-diff={cd:.6e}  rel err={rel:.2e}")
    assert cd < 0  # thicker shell -> less stress
    assert rel < 1e-3
