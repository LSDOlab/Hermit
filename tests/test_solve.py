"""Forward solve + adjoint through ``hm.solve`` -> ``ShellState``, driven by
``ShellDomain`` / ``hm.isotropic`` / ``hm.pressure`` / ``hm.load_vector`` /
``hm.clamp``.
"""

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit
import hermit.bcs as hbc
import hermit.loads as hld
import hermit.material as hmat
from hermit.domain import ShellDomain
from hermit._solve import _pde_for, solve
from conftest import clamped_at_x0


def _build(plate_mesh, ref, thickness):
    E, nu = float(ref["E_val"]), float(ref["nu_val"])
    rho, pz = float(ref["rho_val"]), float(ref["pressure_z"])
    nn = int(ref["n_nodes"])
    dom = ShellDomain(plate_mesh, element="CG2CG1")
    mat = hmat.isotropic(dom, E=E * np.ones(nn), nu=nu * np.ones(nn),
                         thickness=hermit.from_nodal(dom, thickness),
                         density=rho * np.ones(nn), constitutive_space=("Lagrange", 1))
    return solve(dom, mat, hld.pressure(dom, pz), hbc.clamp(dom, where=clamped_at_x0))


def test_forward_solve_matches_reference(plate_mesh, cantilever_ref, recorder):
    h, nn = float(cantilever_ref["h_val"]), int(cantilever_ref["n_nodes"])
    t = csdl.Variable(value=h * np.ones(nn), name="thickness")
    state = _build(plate_mesh, cantilever_ref, t)
    ref = cantilever_ref["disp_solid"]
    rel = np.linalg.norm(state.disp_solid.value - ref) / np.linalg.norm(ref)
    print(f"\n||disp - ref||/||ref|| = {rel:.3e}")
    assert rel < 1e-8


def _total_dobj_dscale(plate_mesh, cantilever_ref, scale_value):
    h, nn = float(cantilever_ref["h_val"]), int(cantilever_ref["n_nodes"])
    rec = csdl.Recorder(inline=True); rec.start()
    scale = csdl.Variable(value=float(scale_value), name="t_scale")
    t = scale * csdl.Variable(value=h * np.ones(nn))
    state = _build(plate_mesh, cantilever_ref, t)
    obj = csdl.sum(state.disp_solid**2)
    val = float(np.ravel(obj.value)[0])
    ana = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([obj], [scale])[obj, scale])[0])
    rec.stop()
    return val, ana


def test_adjoint_wrt_thickness_matches_central_difference(plate_mesh, cantilever_ref):
    _, ana = _total_dobj_dscale(plate_mesh, cantilever_ref, 1.0)
    d = 1e-4
    op, _ = _total_dobj_dscale(plate_mesh, cantilever_ref, 1.0 + d)
    om, _ = _total_dobj_dscale(plate_mesh, cantilever_ref, 1.0 - d)
    cd = (op - om) / (2 * d)
    rel = abs(ana - cd) / abs(cd)
    print(f"\nanalytic={ana:.10e}  central-diff={cd:.10e}  rel err={rel:.2e}")
    assert rel < 1e-4


def test_direct_load_vector_forward_and_adjoint(plate_mesh, cantilever_ref, recorder):
    """A generalized load vector (VF -> mixed W dofs) matches the equivalent pressure."""
    h, nn = float(cantilever_ref["h_val"]), int(cantilever_ref["n_nodes"])
    E, nu, rho = (float(cantilever_ref[k]) for k in ("E_val", "nu_val", "rho_val"))
    pz = float(cantilever_ref["pressure_z"])
    dom = ShellDomain(plate_mesh, element="CG2CG1")
    pde = _pde_for(dom)
    # generalized load vector = assemble(inner(pressure, du_mid) dx) on the W space
    import ufl
    from dolfinx.fem import Function, form
    from dolfinx.fem.petsc import assemble_vector
    dw = ufl.TestFunction(pde.W)
    du_mid, _ = ufl.split(dw)
    pf = Function(pde.VF)
    pf.x.array.reshape(-1, 3)[:, 2] = pz
    lv = assemble_vector(form(ufl.inner(pf, du_mid) * ufl.dx)).getArray().copy()

    s = csdl.Variable(value=1.0, name="lv_scale")
    lvv = s * csdl.Variable(value=lv)
    mat = hmat.isotropic(dom, E=E*np.ones(nn), nu=nu*np.ones(nn),
                         thickness=h*np.ones(nn), density=rho*np.ones(nn),
                         constitutive_space=("Lagrange", 1))
    state = solve(dom, mat, hld.load_vector(dom, lvv), hbc.clamp(dom, where=clamped_at_x0))

    rel = np.linalg.norm(state.disp_solid.value - cantilever_ref["disp_solid"]) \
        / np.linalg.norm(cantilever_ref["disp_solid"])
    print(f"\nload-vector vs pressure: ||disp-ref||/||ref|| = {rel:.3e}")
    assert rel < 1e-8

    obj = csdl.sum(state.disp_solid**2)
    ana = float(np.ravel(csdl.experimental.PySimulator(recorder).compute_totals([obj], [s])[obj, s])[0])
    # dobj/ds for a linear system with RHS = s*lv: obj ~ s^2 => dobj/ds = 2*obj/s
    assert ana == pytest.approx(2.0 * float(np.ravel(obj.value)[0]), rel=1e-6)
