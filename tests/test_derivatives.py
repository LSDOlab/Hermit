"""Total derivatives (adjoint) vs the femo reference vectors.

``rmshell_cantilever.npz`` carries ``dtot_<out>_dthickness`` computed by femo's own
adjoint (node / CG1 ordering, since ``thickness`` is a nodal Variable). Hermit's
``compute_totals`` must reproduce them -- this is what the thickness optimizer consumes.

The thickness is a **file**-order nodal ``csdl.Variable`` (``as_field`` ->
``from_nodal``), which is the ordering the femo reference vectors are in.
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
    E, nu, rd = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val"))
    pz, nn = float(ref["pressure_z"]), int(ref["n_nodes"])
    domain = ShellDomain(plate_mesh, element="CG2CG1")
    material = hmat.isotropic(domain, E=E * np.ones(nn), nu=nu * np.ones(nn),
                              thickness=thickness, density=rd * np.ones(nn),
                              constitutive_space=("Lagrange", 1))
    return solve(domain, material, hld.pressure(domain, pz),
                 hbc.clamp(domain, where=clamped_at_x0))


@pytest.mark.parametrize("name", ["compliance", "mass"])
def test_total_derivative_wrt_thickness_matches_femo(plate_mesh, cantilever_ref, name):
    h, nn = float(cantilever_ref["h_val"]), int(cantilever_ref["n_nodes"])
    rec = csdl.Recorder(inline=True); rec.start()
    t = csdl.Variable(value=h * np.ones(nn), name="thickness")
    state = _state(plate_mesh, cantilever_ref, t)
    o = getattr(out, name)(state)
    jac = csdl.experimental.PySimulator(rec).compute_totals([o], [t])[o, t]
    rec.stop()

    got = np.asarray(jac).ravel()
    want = np.asarray(cantilever_ref[f"dtot_{name}_dthickness"]).ravel()
    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
    print(f"\nd(total) {name}/d thickness : ||rel|| = {rel:.2e}")
    assert rel < 1e-6


def _aggregated_total(plate_mesh, ref, scale_value, m):
    """(value, d value / d uniform thickness scale) for the stress aggregate."""
    h, nn = float(ref["h_val"]), int(ref["n_nodes"])
    rec = csdl.Recorder(inline=True); rec.start()
    scale = csdl.Variable(value=float(scale_value), name="s")
    o = out.aggregated_stress(_state(plate_mesh, ref, scale * csdl.Variable(value=h * np.ones(nn))),
                              rho=100.0, m=m)
    val = float(np.ravel(o.value)[0])
    ana = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([o], [scale])[o, scale])[0])
    rec.stop()
    return val, ana


def test_aggregated_stress_derivative_is_no_longer_degenerate(plate_mesh, cantilever_ref):
    """Retired reference gate, replaced by a real one.

    ``cantilever_ref["dtot_aggregated_stress_dthickness"]`` is identically zero, and the
    old test asserted hermit reproduced that. It is an artifact: femo's aggregate was the
    constant softabs floor, so of course its gradient vanished. With the softabs removed
    a zero gradient would be a *bug* -- this is the quantity a stress-constrained
    thickness optimizer differentiates. Gated instead against a central difference of
    the same aggregate.
    """
    m = 1.0 / 1.4262e4                       # ~1/max(vm); see test_outputs_api
    v0, ana = _aggregated_total(plate_mesh, cantilever_ref, 1.0, m)
    d = 1e-4
    vp, _ = _aggregated_total(plate_mesh, cantilever_ref, 1.0 + d, m)
    vm, _ = _aggregated_total(plate_mesh, cantilever_ref, 1.0 - d, m)
    cd = (vp - vm) / (2 * d)
    rel = abs(ana - cd) / abs(cd)
    print(f"\naggregated_stress: value={v0:.6e}  analytic={ana:.6e}  central-diff={cd:.6e}"
          f"  rel err={rel:.2e}  d(log agg)/d(log t)={ana / v0:.4f}")
    assert np.isfinite(ana) and ana != 0.0
    assert cd < 0                            # thicker shell -> lower stress
    assert rel < 1e-3
