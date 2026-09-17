"""``hm.solve`` / ``ShellState`` end-to-end gates.

Unlike ``test_material_api.py`` / ``test_loads_api.py`` (which drive ``ShellSolveOp``
directly), this file drives the real public ``hermit.solve.solve`` end to end:
``ShellDomain`` / ``Material`` / ``Loads`` / ``BoundaryConditions`` / ``Geometry``
all feeding it, exactly the way a user would call ``hm.solve``.

Gates that need a scalar value assemble it directly off the ``ShellPDE`` a solved
``ShellState`` used (``ShellPDE.compliance_form`` / ``mass_form``), through
``ShellScalarFormsOp`` -- the same "drive the raw ops" pattern the other op-level
test files use.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit
import hermit.material as hmat
import hermit.loads as hld
import hermit.bcs as hbc
from hermit.domain import ShellDomain
from hermit._solve import solve, ShellState, _pde_for, _load_terms
from hermit.fenics.ops import ShellScalarFormsOp
from conftest import assert_matches_legacy, clamped_at_x0


# -- helpers: assemble compliance/mass off a solved ShellState, driving the raw ops
#    (test scaffolding, not the public postprocess functions) --------------------

def _compliance(state):
    """Assembled *against the state's own geometry* -- ``differentiable_geometry``
    follows ``state.geometry.is_differentiable`` and ``mesh_nodes`` is
    ``state.geometry.nodes`` (not a fixed reference), so a shape-derivative gate
    (Gate 3) exercises the pressure term's live ``CellNormal`` / ``dx`` in *both* the
    residual (via ``disp_solid(node_disp)``, already exact -- ``ShellSolveOp``'s own
    ``dR/d(mesh_nodes)``) and this reduction to a scalar, not just the former."""
    pde = _pde_for(state.domain)
    _, spec, values = _load_terms(state.loads)
    coefficients = {name: sp for name, _, sp in spec}
    load_terms = [(kind, pde.coefficient(name, sp)) for name, kind, sp in spec]
    form = pde.compliance_form(loads=load_terms)
    arg_names = tuple(name for name, _, _ in spec) + ("disp_solid",)
    diff_geom = state.geometry.is_differentiable
    sop = ShellScalarFormsOp(pde, {"compliance": (form, arg_names)},
                             differentiable_geometry=diff_geom, coefficients=coefficients)
    fe = dict(values)
    fe["disp_solid"] = state.disp_solid
    fe["mesh_nodes"] = state.geometry.nodes
    out = sop.evaluate(SimpleNamespace(**fe))
    direct = state.loads.direct_vector()
    return out.compliance + csdl.vdot(direct, state.disp_solid)


def _mass(state):
    pde = _pde_for(state.domain)
    form = pde.mass_form()   # fixed-space self.h/self.density -- valid when the
                              # Material's thickness/density share that default space
    sop = ShellScalarFormsOp(pde, {"mass": (form, ("thickness", "density"))})
    fe = SimpleNamespace(thickness=state.material.thickness.coeffs,
                        density=state.material.density.coeffs,
                        mesh_nodes=csdl.Variable(value=state.domain.node_coords))
    return sop.evaluate(fe).mass


def _isotropic_default(dom, E, nu, h, rho):
    """Material on the domain's default (Lagrange, 1) spaces -- matches a freshly
    built ShellPDE's fixed VABD/VT space (element_wise_material=False), so
    combined_traction-free hm.solve output lines up with the femo reference and with
    _mass()'s fixed-space form."""
    nn = dom.n_nodes
    return hmat.isotropic(dom, E=E * np.ones(nn), nu=nu * np.ones(nn),
                          thickness=h * np.ones(nn), density=rho * np.ones(nn),
                          constitutive_space=("Lagrange", 1))


# -- Gate 1: reproduce the captured femo reference -----------------------------

def test_solve_matches_femo_reference(plate_mesh, cantilever_ref, recorder):
    ref = cantilever_ref
    E, nu, rho, h, pz = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val", "h_val", "pressure_z"))

    dom = ShellDomain(plate_mesh, element="CG2CG1")
    mat = _isotropic_default(dom, E, nu, h, rho)
    loads = hld.pressure(dom, pz, space=("DG", 0))
    bcs = hbc.clamp(dom, where=clamped_at_x0)

    state = solve(dom, mat, loads, bcs)

    rel_disp = np.linalg.norm(state.disp_solid.value - ref["disp_solid"]) / np.linalg.norm(ref["disp_solid"])
    print(f"\n||disp - ref||/||ref|| = {rel_disp:.3e}")
    assert rel_disp < 1e-8

    c = _compliance(state)
    m = _mass(state)
    c_v, m_v = float(np.ravel(c.value)[0]), float(np.ravel(m.value)[0])
    rel_c = abs(c_v - float(ref["compliance"])) / abs(float(ref["compliance"]))
    rel_m = abs(m_v - float(ref["mass"])) / abs(float(ref["mass"]))
    print(f"compliance  new={c_v:.10e}  ref={float(ref['compliance']):.10e}  rel={rel_c:.2e}")
    print(f"mass        new={m_v:.10e}  ref={float(ref['mass']):.10e}  rel={rel_m:.2e}")
    assert c_v == pytest.approx(float(ref["compliance"]), rel=1e-8)
    assert m_v == pytest.approx(float(ref["mass"]), rel=1e-10)

    # geometry=None normalises to the reference configuration
    assert state.geometry is not None
    assert state.geometry.is_differentiable is False
    assert np.array_equal(np.asarray(state.geometry.nodes.value), dom.node_coords)


# -- Gate 2: hm.traction bit-identical to the legacy nodal_pressure path, through
#    the real hm.solve (not raw ops) -------------------------------------------

def test_traction_matches_legacy_nodal_pressure_through_solve(plate_mesh, cantilever_ref,
                                                              legacy_ref, recorder):
    ref = cantilever_ref
    nn = int(ref["n_nodes"])
    E, nu, rho, h, pz = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val", "h_val", "pressure_z"))
    p = np.zeros((nn, 3)); p[:, 2] = pz
    legacy_disp = legacy_ref["traction_nodal_pressure__disp_solid"]
    legacy_c = float(legacy_ref["traction_nodal_pressure__compliance"])

    dom = ShellDomain(plate_mesh, element="CG2CG1")
    mat = _isotropic_default(dom, E, nu, h, rho)
    t_field = hermit.from_nodal(dom, p)
    loads = hld.traction(dom, t_field, space=("Lagrange", 1))
    bcs = hbc.clamp(dom, where=clamped_at_x0)
    state = solve(dom, mat, loads, bcs)

    diff = np.abs(legacy_disp - state.disp_solid.value).max()
    print(f"\nmax|disp_legacy - disp_new| = {diff:.3e}")
    assert_matches_legacy(state.disp_solid.value, legacy_disp)

    c_new = float(np.ravel(_compliance(state).value)[0])
  # 1e-7, not 1e-9: the constant-normal curvature (#7) is analytically identical
    # on this flat fixture but evaluates a different expression tree, which the
    # beta=1e15 penalty system amplifies to ~7e-9. See tests/conftest.py,
    # assert_matches_legacy, for the full argument.
    assert c_new == pytest.approx(legacy_c, rel=1e-7)


# -- Gate 3: pressure shape derivative vs central FD --

def _smooth_bend(gx, nn, L=10.0, W=2.0):
    bend = np.zeros((nn, 3))
    bend[:, 2] = 0.3 * (gx[:, 0] / L) ** 2
    bend[:, 0] = 0.02 * (gx[:, 0] / L)
    V = np.zeros((nn, 3))
    V[:, 2] = np.sin(np.pi * gx[:, 0] / L) * (0.5 + gx[:, 1] / W)
    V[:, 0] = 0.3 * gx[:, 0] / L
    V /= np.linalg.norm(V)
    return bend, V


def test_pressure_compliance_shape_derivative_matches_central_difference(plate_mesh, cantilever_ref):
    ref = cantilever_ref
    E, nu, rho, h, pz = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val", "h_val", "pressure_z"))
    gx = plate_mesh.geometry.x.copy()
    nn = int(ref["n_nodes"])
    bend, Vdir = _smooth_bend(gx, nn)

    def run(nd_val, want_grad):
        rec = csdl.Recorder(inline=True); rec.start()
        dom = ShellDomain(plate_mesh, element="CG2CG1")
        mat = _isotropic_default(dom, E, nu, h, rho)
        loads = hld.pressure(dom, pz, space=("DG", 0))
        bcs = hbc.clamp(dom, where=clamped_at_x0)
        nd = csdl.Variable(value=nd_val.copy(), name="node_disp")
        geom = hermit.geometry(dom, node_disp=nd)
        state = solve(dom, mat, loads, bcs, geometry=geom)
        c = _compliance(state)
        val = float(np.ravel(c.value)[0])
        g = None
        if want_grad:
            g = np.asarray(csdl.experimental.PySimulator(rec).compute_totals(
                [c], [nd])[c, nd]).reshape(nn, 3)
        rec.stop()
        return val, g

    _, ana = run(bend, True)
    dd_ana = float((ana * Vdir).sum())
    step = 1e-3
    vp, _ = run(bend + step * Vdir, False)
    vm, _ = run(bend - step * Vdir, False)
    dd_fd = (vp - vm) / (2 * step)
    rel = abs(dd_ana - dd_fd) / max(abs(dd_fd), 1e-10)
    print(f"\nd(compliance)/d(node_disp) . V [pressure]: ana={dd_ana:+.6e}  fd={dd_fd:+.6e}  rel={rel:.2e}")
    assert rel < 3e-3


# -- Gate 4: material fields on non-default spaces run + are differentiable --------

def test_material_on_non_default_spaces_runs_and_is_differentiable(plate_mesh):
    E, nu, rho, pz = 70e9, 0.3, 2700.0, 1.0e3

    def run(t_scale, want_grad):
        rec = csdl.Recorder(inline=True); rec.start()
        dom = ShellDomain(plate_mesh, element="CG2CG1")
        n_t = dom.function_space(("Lagrange", 2)).dofmap.index_map.size_local
        s = csdl.Variable(value=float(t_scale), name="t_scale")
        t = s * csdl.Variable(value=0.01 * np.ones(n_t))
        t_field = hermit.from_coeffs(dom, ("Lagrange", 2), t)
        mat = hmat.isotropic(dom, E=E, nu=nu, thickness=t_field, density=rho,
                             constitutive_space=("DG", 1))
        assert mat.A.space[:2] == ("DG", 1)
        assert mat.thickness.space[:2] == ("Lagrange", 2)
        loads = hld.pressure(dom, pz, space=("DG", 0))
        bcs = hbc.clamp(dom, where=clamped_at_x0)
        state = solve(dom, mat, loads, bcs)
        c = _compliance(state)
        val = float(np.ravel(c.value)[0])
        g = None
        if want_grad:
            g = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([c], [s])[c, s])[0])
        rec.stop()
        return val, g

    v0, ana = run(1.0, True)
    assert np.isfinite(v0) and v0 > 0
    # D ~ t^3 (isotropic closed form), so compliance ~ 1/t_scale^3 in a bending-
    # dominated response -- FD truncation error grows for a large step, roundoff
    # noise for a tiny one (checked at d in [1e-7, 1e-2]: this is the best-converged
    # step, ~2e-4 relative agreement).
    d = 3e-4
    vp, _ = run(1.0 + d, False)
    vm, _ = run(1.0 - d, False)
    fd = (vp - vm) / (2 * d)
    rel = abs(ana - fd) / abs(fd)
    print(f"\nd(compliance)/d(t_scale) [DG1 material, Lagrange2 thickness]: "
         f"ana={ana:.8e}  fd={fd:.8e}  rel={rel:.2e}")
    assert rel < 5e-4


# -- Gate 5: distinct compositions -> distinct cached forms + correct answers ------

def test_distinct_compositions_get_distinct_cached_forms(plate_mesh, recorder):
    E, nu, rho = 70e9, 0.3, 2700.0
    dom = ShellDomain(plate_mesh, element="CG2CG1")
    mat = _isotropic_default(dom, E, nu, 0.01, rho)
    bcs = hbc.clamp(dom, where=clamped_at_x0)

    pde = _pde_for(dom)
    cache = pde._solve_form_cache
    n0 = len(cache)

    loads_p = hld.pressure(dom, 1.0e3, space=("DG", 0))
    state_p = solve(dom, mat, loads_p, bcs)
    n1 = len(cache)
    assert n1 == n0 + 1

    loads_t = hld.traction(dom, [1.0e3, 0.0, 0.0])   # in-plane -- a different deformation
                                                       # mode than the +z pressure, so the
                                                       # two compliances below are genuinely
                                                       # different, not coincidentally equal
    state_t = solve(dom, mat, loads_t, bcs)
    n2 = len(cache)
    assert n2 == n1 + 1   # different load composition -> a new compiled form

    loads_pt = loads_p + loads_t
    state_pt = solve(dom, mat, loads_pt, bcs)
    n3 = len(cache)
    assert n3 == n2 + 1   # yet another composition -> another new compiled form

    mat2 = hmat.isotropic(dom, E=E, nu=nu, thickness=0.01, density=rho,
                          constitutive_space=("DG", 0))   # a *different* material space
    state_p2 = solve(dom, mat2, loads_p, bcs)
    n4 = len(cache)
    assert n4 == n3 + 1   # different constitutive_space -> another new compiled form

    # repeating an already-seen composition reuses the cached form
    loads_p_again = hld.pressure(dom, 5.0e3, space=("DG", 0))
    state_p_again = solve(dom, mat, loads_p_again, bcs)
    assert len(cache) == n4

    # ... and each solve is physically correct: pressure-only and traction-only
    # differ, and the sum's compliance is the independent work-conjugate oracle.
    c_p = float(np.ravel(_compliance(state_p).value)[0])
    c_t = float(np.ravel(_compliance(state_t).value)[0])
    c_pt = float(np.ravel(_compliance(state_pt).value)[0])
    assert c_p != pytest.approx(c_t, rel=1e-3)

    pde_ = _pde_for(dom)
    _, spec_p, values_p = _load_terms(loads_p)
    _, spec_t, values_t = _load_terms(loads_t)
    coefficients = {name: sp for name, _, sp in spec_p + spec_t}
    load_terms = [(kind, pde_.coefficient(name, sp)) for name, kind, sp in spec_p + spec_t]
    form = pde_.compliance_form(loads=load_terms)
    arg_names = tuple(name for name, _, _ in spec_p + spec_t) + ("disp_solid",)
    sop = ShellScalarFormsOp(pde_, {"compliance": (form, arg_names)}, coefficients=coefficients)
    fe = dict(values_p); fe.update(values_t); fe["disp_solid"] = state_pt.disp_solid
    fe["mesh_nodes"] = csdl.Variable(value=dom.node_coords)
    c_oracle = float(np.ravel(sop.evaluate(SimpleNamespace(**fe)).compliance.value)[0])
    print(f"\ncompliance  sum={c_pt:.10e}  oracle={c_oracle:.10e}")
    assert c_pt == pytest.approx(c_oracle, rel=1e-10)

    # the re-solved (different pressure magnitude, same composition) state is a
    # sane, distinct solution -- not silently reusing a stale form's wrong physics
    assert not np.array_equal(state_p.disp_solid.value, state_p_again.disp_solid.value)
    assert np.allclose(5.0 * state_p.disp_solid.value, state_p_again.disp_solid.value, rtol=1e-8)


# -- Gate 6: adjoint check, d(compliance)/d(thickness) through hm.solve vs FD ------

def test_dcompliance_dthickness_matches_central_difference(plate_mesh, cantilever_ref):
    ref = cantilever_ref
    E, nu, rho, h, pz = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val", "h_val", "pressure_z"))
    nn = int(ref["n_nodes"])

    def run(scale, want_grad):
        rec = csdl.Recorder(inline=True); rec.start()
        dom = ShellDomain(plate_mesh, element="CG2CG1")
        s = csdl.Variable(value=float(scale), name="t_scale")
        t = s * csdl.Variable(value=h * np.ones(nn))
        mat = hmat.isotropic(dom, E=E * np.ones(nn), nu=nu * np.ones(nn),
                             thickness=hermit.from_nodal(dom, t), density=rho * np.ones(nn),
                             constitutive_space=("Lagrange", 1))
        loads = hld.pressure(dom, pz, space=("DG", 0))
        bcs = hbc.clamp(dom, where=clamped_at_x0)
        state = solve(dom, mat, loads, bcs)
        c = _compliance(state)
        val = float(np.ravel(c.value)[0])
        g = None
        if want_grad:
            g = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([c], [s])[c, s])[0])
        rec.stop()
        return val, g

    _, ana = run(1.0, True)
    d = 1e-4
    vp, _ = run(1.0 + d, False)
    vm, _ = run(1.0 - d, False)
    fd = (vp - vm) / (2 * d)
    rel = abs(ana - fd) / abs(fd)
    print(f"\nd(compliance)/d(t_scale): analytic={ana:.10e}  central-diff={fd:.10e}  rel={rel:.2e}")
    assert rel < 1e-5


# -- Surrogate drop-in ------------------------------------------------------------

def test_pure_csdl_surrogate_matches_new_signature(plate_mesh, recorder):
    dom = ShellDomain(plate_mesh, element="CG2CG1")
    mat = hmat.isotropic(dom, E=70e9, nu=0.3, thickness=0.01, density=2700.0)
    loads = hld.pressure(dom, 1.0e3, space=("DG", 0))
    bcs = hbc.clamp(dom, where=clamped_at_x0)
    ndof = dom.W.dofmap.index_map.size_local * dom.W.dofmap.index_map_bs

    def surrogate(domain, material, loads, bcs, *, geometry=None, linear=True):
        w = csdl.expand(csdl.sum(material.thickness.coeffs), (ndof,))   # pure CSDL, no FEniCS
        return ShellState(domain, geometry, material, loads, bcs, disp_solid=w)

    state = surrogate(dom, mat, loads, bcs)
    assert state.disp_solid.shape == (ndof,)
    # geometry=None still normalises, exactly like hm.solve
    assert state.geometry is not None and state.geometry.is_differentiable is False

    # differentiable end to end
    t = csdl.Variable(value=0.02, name="t")
    mat2 = hmat.isotropic(dom, E=70e9, nu=0.3, thickness=t, density=2700.0)
    state2 = surrogate(dom, mat2, loads, bcs)
    obj = csdl.sum(state2.disp_solid ** 2)
    g = csdl.experimental.PySimulator(recorder).compute_totals([obj], [t])[obj, t]
    assert np.isfinite(np.ravel(g)).all()


# -- "one W per domain" hard constraint -------------------------------------------

def test_solve_rejects_bcs_from_a_different_domain(plate_mesh, tri_mesh, recorder):
    dom1 = ShellDomain(plate_mesh, element="CG2CG1")
    dom2 = ShellDomain(tri_mesh, element="CG2CG1")
    mat = hmat.isotropic(dom1, E=70e9, nu=0.3, thickness=0.01, density=2700.0)
    loads = hld.pressure(dom1, 1.0e3, space=("DG", 0))
    bcs_wrong_domain = hbc.clamp(dom2, where=clamped_at_x0)
    with pytest.raises(ValueError, match="ShellDomain"):
        solve(dom1, mat, loads, bcs_wrong_domain)


def test_solve_rejects_thickness_only_material(plate_mesh, recorder):
    dom = ShellDomain(plate_mesh, element="CG2CG1")
    mat = hmat.thickness_only(dom, thickness=0.01, density=2700.0)
    loads = hld.pressure(dom, 1.0e3, space=("DG", 0))
    bcs = hbc.clamp(dom, where=clamped_at_x0)
    with pytest.raises(ValueError, match="A/B/D/As"):
        solve(dom, mat, loads, bcs)
