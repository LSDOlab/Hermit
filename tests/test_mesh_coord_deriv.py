"""Validate mesh-coordinate ("shape") derivatives against finite differences.

Requires the UFL CoordinateDerivative patch (``hermit._ufl_compat``, applied on
import; works against both the UFL 2024.2 and 2026.1 ruleset APIs). FD is a
directional derivative  g . V  with a *smooth* warp and direction:
a random crumpled perturbation makes the penalty-BC shell system genuinely
ill-conditioned and the FD stops converging (this is physics, not a bug), so we use a
gentle bend + smooth V and step 1e-3.

The shape design variable is ``geometry=hm.geometry(domain, node_disp=nd)`` (``nd``
is ``(n_nodes, 3)`` in **file** order), and the outputs come from ``hermit.outputs``.
"""

import numpy as np
import pytest

import hermit  # applies the UFL patch on import
import hermit.bcs as hbc
import hermit.loads as hld
import hermit.material as hmat
import hermit.outputs as out
from hermit.domain import ShellDomain
from hermit._solve import solve
from conftest import clamped_at_x0


def test_ufl_patch_is_active():
    import hermit._ufl_compat as uc

    assert uc.installed_mode() in ("singledispatch", "multifunction")
    assert uc.active_reference_grad_handler().__module__ == "hermit._ufl_compat"


def _smooth_fields(gx, nn, L=10.0, W=2.0):
    bend = np.zeros((nn, 3))
    bend[:, 2] = 0.3 * (gx[:, 0] / L) ** 2
    bend[:, 0] = 0.02 * (gx[:, 0] / L)
    V = np.zeros((nn, 3))
    V[:, 2] = np.sin(np.pi * gx[:, 0] / L) * (0.5 + gx[:, 1] / W)
    V[:, 0] = 0.3 * gx[:, 0] / L
    V /= np.linalg.norm(V)
    return bend, V


def _warped_state(plate_mesh, ref, nd):
    """Solve with ``nd`` (file order) as a live shape design variable."""
    E, nu, h = (float(ref[k]) for k in ("E_val", "nu_val", "h_val"))
    rho, pz, nn = float(ref["rho_val"]), float(ref["pressure_z"]), int(ref["n_nodes"])
    domain = ShellDomain(plate_mesh, element="CG2CG1")
    material = hmat.isotropic(domain, E=E * np.ones(nn), nu=nu * np.ones(nn),
                              thickness=h * np.ones(nn), density=rho * np.ones(nn),
                              constitutive_space=("Lagrange", 1))
    return solve(domain, material, hld.pressure(domain, pz),
                 hbc.clamp(domain, where=clamped_at_x0),
                 geometry=hermit.geometry(domain, node_disp=nd))


@pytest.mark.parametrize("output", ["compliance", "mass", "elastic_energy"])
def test_output_wrt_node_disp_vs_central_difference(plate_mesh, cantilever_ref, output):
    import csdl_alpha as csdl

    nn = int(cantilever_ref["n_nodes"])
    gx = plate_mesh.geometry.x.copy()
    bend, Vdir = _smooth_fields(gx, nn)

    def run(nd_val, want_grad):
        assert np.allclose(plate_mesh.geometry.x, gx), "an op left mesh.geometry.x moved"
        rec = csdl.Recorder(inline=True); rec.start()
        nd = csdl.Variable(value=nd_val.copy(), name="node_disp")
        state = _warped_state(plate_mesh, cantilever_ref, nd)
        o = getattr(out, output)(state)
        val = float(np.ravel(o.value)[0])
        g = None
        if want_grad:
            g = np.asarray(csdl.experimental.PySimulator(rec).compute_totals(
                [o], [nd])[o, nd]).reshape(nn, 3)
        rec.stop()
        return val, g

    _, ana = run(bend, True)
    dd_ana = float((ana * Vdir).sum())
    step = 1e-3
    vp, _ = run(bend + step * Vdir, False)
    vm, _ = run(bend - step * Vdir, False)
    dd_fd = (vp - vm) / (2 * step)
    rel = abs(dd_ana - dd_fd) / max(abs(dd_fd), 1e-10)
    print(f"\nd({output})/d(node_disp) . V : ana={dd_ana:+.6e}  fd={dd_fd:+.6e}  rel={rel:.2e}")
    assert rel < 3e-3


def test_disp_solid_wrt_node_disp_vs_central_difference(plate_mesh, cantilever_ref):
    """The ShellSolveOp geometry adjoint in isolation (functional of the raw state)."""
    import csdl_alpha as csdl

    nn = int(cantilever_ref["n_nodes"])
    gx = plate_mesh.geometry.x.copy()
    bend, Vdir = _smooth_fields(gx, nn)

    def obj(nd_val, want_grad):
        rec = csdl.Recorder(inline=True); rec.start()
        nd = csdl.Variable(value=nd_val.copy(), name="node_disp")
        state = _warped_state(plate_mesh, cantilever_ref, nd)
        o = csdl.sum(state.disp_solid**2)
        val = float(np.ravel(o.value)[0])
        g = None
        if want_grad:
            g = np.asarray(csdl.experimental.PySimulator(rec).compute_totals(
                [o], [nd])[o, nd]).reshape(nn, 3)
        rec.stop()
        return val, g

    _, ana = obj(bend, True)
    dd_ana = float((ana * Vdir).sum())
    step = 1e-3
    vp, _ = obj(bend + step * Vdir, False)
    vm, _ = obj(bend - step * Vdir, False)
    dd_fd = (vp - vm) / (2 * step)
    rel = abs(dd_ana - dd_fd) / max(abs(dd_fd), 1e-10)
    print(f"\nd(sum disp^2)/d(node_disp) . V : ana={dd_ana:+.6e}  fd={dd_fd:+.6e}  rel={rel:.2e}")
    assert rel < 3e-3
