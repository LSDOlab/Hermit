"""Adjoint of the nodal displacement extraction (pure CSDL, no custom op).

Driven by ``ShellDomain`` / ``hm.isotropic`` / ``hm.pressure`` / ``hm.clamp`` /
``hm.outputs``. The forward values are gated by
``tests/test_outputs_api.py::test_nodal_outputs_match_reference``; the thickness
adjoint below is unique to this file.
"""

import numpy as np

import csdl_alpha as csdl
import hermit.bcs as hbc
import hermit.loads as hld
import hermit.material as hmat
import hermit.outputs as out
from hermit.domain import ShellDomain
from hermit._solve import solve
from conftest import clamped_at_x0


def test_nodal_displacement_adjoint_wrt_thickness(plate_mesh, cantilever_ref):
    ref = cantilever_ref
    h, nn = float(ref["h_val"]), int(ref["n_nodes"])
    E, nu, rd = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val"))
    pz = float(ref["pressure_z"])

    def sse(scale, grad):
        rec = csdl.Recorder(inline=True); rec.start()
        s = csdl.Variable(value=float(scale), name="s")
        t = s * csdl.Variable(value=h * np.ones(nn))
        domain = ShellDomain(plate_mesh, element="CG2CG1")
        material = hmat.isotropic(domain, E=E * np.ones(nn), nu=nu * np.ones(nn),
                                  thickness=t, density=rd * np.ones(nn),
                                  constitutive_space=("Lagrange", 1))
        state = solve(domain, material, hld.pressure(domain, pz),
                      hbc.clamp(domain, where=clamped_at_x0))
        o = csdl.sum(out.nodal_displacement(state)[:, 2] ** 2)
        val = float(np.ravel(o.value)[0])
        g = None
        if grad:
            g = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([o], [s])[o, s])[0])
        rec.stop()
        return val, g

    _, ana = sse(1.0, True)
    d = 1e-4
    vp, _ = sse(1.0 + d, False)
    vm, _ = sse(1.0 - d, False)
    cd = (vp - vm) / (2 * d)
    rel = abs(ana - cd) / abs(cd)
    print(f"\nd(sum w_z^2)/ds : analytic={ana:.6e}  cd={cd:.6e}  rel={rel:.2e}")
    assert rel < 1e-4
