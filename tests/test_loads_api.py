"""``hermit.loads`` (``Loads``/``pressure``/``traction``/``moment``/``point_load``/
``load_vector``) against the load mechanisms it reduces onto -- see
``hermit/loads.py``'s module docstring for the reduction this implements and why.
Drives ``ShellSolveOp`` / ``ShellScalarFormsOp`` directly, like
``test_material_api.py`` / ``test_orientation_form.py``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit
import hermit.bcs as hbc
import hermit.material as hmat
import hermit.loads as hld
from hermit._solve import _pde_for
from hermit.domain import ShellDomain
from hermit.fenics.ops import ShellScalarFormsOp, ShellSolveOp
from hermit.fenics.shell_pde import ShellPDE
from conftest import assert_matches_legacy, clamped_at_x0


def _isotropic_material(dom, E, nu, h, rho, space):
    return hmat.isotropic(dom, E=E, nu=nu, thickness=h, density=rho, constitutive_space=space)


def _raw(dom, **pde_kw):
    """The raw FE operands these gates drive directly: ``dom``'s ``ShellPDE`` and a
    penalty-clamp ``BCData`` at ``clamped_at_x0``.

    ``pde_kw`` (``element_wise_material`` / ``elementwise_pressure``) are ``ShellPDE``'s
    own fixed-space switches; when given, a fresh ``ShellPDE`` is built on ``dom.W``
    (so BCs still locate against the one state space) instead of the domain's default
    cached one."""
    if pde_kw:
        pde = ShellPDE(dom.mesh, element=dom.element, W=dom.W, **pde_kw)
        pde._solve_form_cache = {}
    else:
        pde = _pde_for(dom)
    return pde, hbc.clamp(dom, where=clamped_at_x0).to_bc_data()


def _solve_new(pde, bc, dom, material, loads):
    """Drive ``ShellSolveOp`` / ``ShellScalarFormsOp`` off a new-surface
    ``Material`` + ``Loads`` (reduced onto ``f``/``m``/``load_vector`` -- see
    ``hermit/loads.py``). Returns ``(compliance, disp_solid)``."""
    nvf = pde.VF.dofmap.index_map.size_local
    f_field = loads.combined_traction()
    m_field = loads.combined_moment()
    f = f_field.coeffs if f_field is not None else csdl.Variable(value=np.zeros(nvf * 3))
    m_ = m_field.coeffs if m_field is not None else csdl.Variable(value=np.zeros(nvf * 3))
    mesh_nodes = csdl.Variable(value=dom.node_coords)

    arg_names = ["A", "B", "D", "As", "thickness", "f", "m"]
    fe_kwargs = dict(A=material.A.coeffs, B=material.B.coeffs, D=material.D.coeffs,
                     As=material.As.coeffs, thickness=material.thickness.coeffs,
                     f=f, m=m_, mesh_nodes=mesh_nodes)
    has_direct = loads.direct is not None
    if has_direct:
        direct = loads.direct_vector()
        arg_names.append("load_vector")
        fe_kwargs["load_vector"] = direct

    op = ShellSolveOp(pde, bc, tuple(arg_names), None, form_cache=pde._solve_form_cache)
    disp = op.evaluate(SimpleNamespace(**fe_kwargs))

    sop = ShellScalarFormsOp(pde, {"compliance": (pde.compliance_form(), ("disp_solid", "f", "m"))})
    fe2 = dict(disp_solid=disp, f=f, m=m_, mesh_nodes=mesh_nodes)
    out = sop.evaluate(SimpleNamespace(**fe2))
    c = out.compliance
    if has_direct:
        c = c + csdl.vdot(direct, disp)
    return c, disp


# -- gate: hm.traction on ("Lagrange", 1) identical to legacy nodal_pressure -----

def test_traction_default_space_identical_to_legacy_nodal_pressure(plate_mesh, cantilever_ref,
                                                                   legacy_ref, recorder):
    ref = cantilever_ref
    nn = int(ref["n_nodes"])
    E, nu, rho, h, pz = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val", "h_val", "pressure_z"))
    p = np.zeros((nn, 3)); p[:, 2] = pz
    legacy_disp = legacy_ref["traction_nodal_pressure__disp_solid"]
    legacy_c = float(legacy_ref["traction_nodal_pressure__compliance"])

    dom = ShellDomain(plate_mesh, element="CG2CG1")
    pde, bc = _raw(dom)
    mat = _isotropic_material(dom, E, nu, h, rho, ("Lagrange", 1))
    t_field = hermit.from_nodal(dom, p)                     # ("Lagrange", 1, (3,)), FE order
    loads = hld.traction(dom, t_field, space=("Lagrange", 1))
    c_new, disp_new = _solve_new(pde, bc, dom, mat, loads)

    print(f"\nmax|disp_legacy - disp_new| = {np.abs(legacy_disp - disp_new.value).max():.3e}")
    assert_matches_legacy(disp_new.value, legacy_disp)
  # 1e-7, not 1e-9: the constant-normal curvature (#7) is analytically identical
    # on this flat fixture but evaluates a different expression tree, which the
    # beta=1e15 penalty system amplifies to ~7e-9. See tests/conftest.py,
    # assert_matches_legacy, for the full argument.
    assert legacy_c == pytest.approx(float(np.ravel(c_new.value)[0]), rel=1e-7)


# -- gate: hm.point_load at a mesh vertex matches legacy nodal_forces -----------

def test_point_load_at_vertex_matches_legacy_nodal_forces(plate_mesh, legacy_ref, recorder):
    """Exact only when the state's displacement space matches VF's default space
    (both Lagrange-1) -- see hermit/loads.py's module docstring / point_load's
    docstring: legacy nodal_forces converts a *lumped* nodal force F into a
    "consistent" VF-space pressure field p via csdl.solve_linear(M, F) (M = the
    VF mass matrix), then feeds p through the *same* ``inner(f, du_mid)*dx`` term
    hm.traction uses. Since M @ p = F exactly (by construction of solve_linear),
    the assembled RHS in *VF's own* dof numbering is exactly F; that only equals
    the assembled RHS in the *state*'s dof numbering (what actually gets solved)
    when the two test spaces coincide -- element="CG1CG1" here, matching VF's
    default ("Lagrange", 1). hm.point_load's Dirac-delta scatter needs no such
    match -- it is exact for any element."""
    E, nu, rho, h = 70e9, 0.3, 2700.0, 0.01
    F0 = np.array([1.0e3, -2.0e3, 5.0e2])
    legacy_disp = legacy_ref["point_load_nodal_forces__disp_solid"]

    dom = ShellDomain(plate_mesh, element="CG1CG1")
    pde, bc = _raw(dom)
    vtx = int(np.argmax(dom.node_coords[:, 0]))                  # a non-clamped (tip) vertex, file order
    assert vtx == int(legacy_ref["point_load_nodal_forces__vtx"])
    assert np.array_equal(F0, legacy_ref["point_load_nodal_forces__F0"])
    mat = _isotropic_material(dom, E, nu, h, rho, ("Lagrange", 1))
    at = dom.node_coords[vtx]                                    # physical coordinate, file order
    assert np.array_equal(at, legacy_ref["point_load_nodal_forces__at"])
    loads = hld.point_load(dom, at=at, force=F0)
    c_new, disp_new = _solve_new(pde, bc, dom, mat, loads)

    diff = np.abs(legacy_disp - disp_new.value).max()
    print(f"\nmax|disp_legacy - disp_new| = {diff:.3e}  (max|disp|={np.abs(disp_new.value).max():.3e})")
    # Not literally np.array_equal: legacy builds its RHS via csdl.solve_linear(M, F)
    # (a dense LU solve of the VF mass matrix) while point_load scatters force*phi_i
    # directly -- two different floating-point paths to the *same* assembled RHS
    # (M @ (M^-1 F) == F exactly in exact arithmetic). The other traction /
    # nodal_pressure gates *are* bit-identical because they reuse literally the same
    # array with no intervening linear solve.
    #
    # atol is 1e-11, not the 1e-12 this used while the legacy side was recomputed
    # live in the same process. Against a stored fixture the two sides no longer
    # share round-off, and the difference is then dominated by the penalty system's
    # conditioning (beta = 1e15), which makes it sensitive to BLAS threading:
    # measured 2.029e-13 / 5.412e-13 / 2.697e-13 at 4 / 2 / 1 threads, against
    # max|disp| = 6.13e-2. 1e-12 left only ~1.9x margin and would fail on a machine
    # whose BLAS differs from the one that generated the fixture. 1e-11 is still
    # ~1.6e-10 *relative* -- a wrong RHS would differ by O(1) relative, so the gate
    # loses no power.
    assert np.allclose(legacy_disp, disp_new.value, rtol=0, atol=1e-11)


# -- gate: hm.pressure vs the legacy elementwise-pressure path ------------------

def test_pressure_matches_legacy_elementwise_pressure(plate_mesh, cantilever_ref, legacy_ref, recorder):
    """On this flat plate the shell normal is the single constant vector +z, so
    hm.pressure's p*n (domain.local_frames(), numpy, reference geometry) is
    exactly the per-cell vector the legacy elementwise ("DG", 0) nodal_pressure
    path already uses -- see hm.pressure's docstring for why this need not hold
    on a curved shell."""
    ref = cantilever_ref
    E, nu, rho, h, pz = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val", "h_val", "pressure_z"))
    # legacy: the same p*n_hat cell vectors fed through field_loads(nodal_pressure=...)
    # on a ("DG", 0) load space -- the stored `elementwise_pressure` fixture case
    legacy_disp = legacy_ref["elementwise_pressure__disp_solid"]
    legacy_c = float(legacy_ref["elementwise_pressure__compliance"])

    dom = ShellDomain(plate_mesh, element="CG2CG1")
    pde, bc = _raw(dom, elementwise_pressure=True)
    mat = _isotropic_material(dom, E, nu, h, rho, ("Lagrange", 1))
    loads = hld.pressure(dom, pz, space=("DG", 0))
    # reduce onto ("DG", 0, (3,)) to match the legacy elementwise VF space exactly
    # (_solve_new's default combine target is ("Lagrange", 1, (3,)), so this case is
    # driven by hand rather than through that helper)
    f_field = loads.combined_traction(space=("DG", 0, (3,)))
    m_zero = csdl.Variable(value=np.zeros(f_field.coeffs.shape))
    mesh_nodes = csdl.Variable(value=dom.node_coords)
    op = ShellSolveOp(pde, bc, ("A", "B", "D", "As", "thickness", "f", "m"), None,
                      form_cache=pde._solve_form_cache)
    disp_new2 = op.evaluate(SimpleNamespace(A=mat.A.coeffs, B=mat.B.coeffs, D=mat.D.coeffs,
                                            As=mat.As.coeffs, thickness=mat.thickness.coeffs,
                                            f=f_field.coeffs, m=m_zero, mesh_nodes=mesh_nodes))
    sop = ShellScalarFormsOp(pde, {"compliance": (pde.compliance_form(), ("disp_solid", "f", "m"))})
    out = sop.evaluate(SimpleNamespace(disp_solid=disp_new2, f=f_field.coeffs, m=m_zero, mesh_nodes=mesh_nodes))

    rel = abs(float(np.ravel(out.compliance.value)[0]) - legacy_c) / abs(legacy_c)
    print(f"\ncompliance  legacy={legacy_c:.12e}  "
         f"new={float(np.ravel(out.compliance.value)[0]):.12e}  rel={rel:.2e}")
    assert_matches_legacy(disp_new2.value, legacy_disp)


def test_pressure_raises_on_inconsistent_orientation(plate_mesh, recorder):
    """``check_cell_orientation_consistency`` (and hence ``hm.pressure``) must
    reject a mesh where two facet-adjacent cells' normals point opposite ways.
    Rather than hand-build a pathological (self-intersecting-risk) mesh, flip one
    real adjacent cell's memoized frame on the ordinary plate fixture -- exercises
    exactly the same check, without touching dolfinx mesh construction."""
    dom = hermit.ShellDomain(plate_mesh)
    tdim = dom.mesh.topology.dim
    fdim = tdim - 1
    dom.mesh.topology.create_connectivity(fdim, tdim)
    f2c = dom.mesh.topology.connectivity(fdim, tdim)
    pair = next((tuple(int(c) for c in f2c.links(f)) for f in range(f2c.num_nodes)
                if len(f2c.links(f)) == 2), None)
    assert pair is not None

    frames = dom.local_frames().copy()
    frames[pair[1]] = -frames[pair[1]]     # flip the whole frame -> flips the normal
    dom._frames = frames                    # same object local_frames() already returned
    with pytest.raises(ValueError, match="inconsistently oriented"):
        hld.pressure(dom, 1.0)


# -- compliance = work conjugate, for one load and for a sum -------------------

def _plate_setup(mesh, space=("Lagrange", 1)):
    dom = ShellDomain(mesh, element="CG2CG1")
    pde, bc = _raw(dom)
    mat = _isotropic_material(dom, 70e9, 0.3, 0.01, 2700.0, space)
    return pde, bc, dom, mat


def test_compliance_equals_work_conjugate_for_traction_and_sum(plate_mesh, recorder):
    pde, bc, dom, mat = _plate_setup(plate_mesh)
    nn = dom.n_nodes
    t1 = np.zeros((nn, 3)); t1[:, 2] = 1.0e3
    t2 = np.zeros((nn, 3)); t2[:, 0] = 2.0e2

    l1 = hld.traction(dom, hermit.from_nodal(dom, t1))
    l2 = hld.traction(dom, hermit.from_nodal(dom, t2))
    c_single, _ = _solve_new(pde, bc, dom, mat, l1)

    pde2, bc2, dom2, mat2 = _plate_setup(plate_mesh)
    l1b = hld.traction(dom2, hermit.from_nodal(dom2, t1))
    l2b = hld.traction(dom2, hermit.from_nodal(dom2, t2))
    c_sum, disp_sum = _solve_new(pde2, bc2, dom2, mat2, l1b + l2b)

    # independent "work conjugate" oracle: assemble sum(t . disp) using the *legacy*
    # compliance_form machinery directly on the summed traction, on the *same* solve
    f_sum = (hermit.from_nodal(dom2, t1).coeffs + hermit.from_nodal(dom2, t2).coeffs)
    sop = ShellScalarFormsOp(pde2, {"compliance": (pde2.compliance_form(), ("disp_solid", "f", "m"))})
    mesh_nodes = csdl.Variable(value=dom2.node_coords)
    m_zero = csdl.Variable(value=np.zeros(f_sum.shape))
    out = sop.evaluate(SimpleNamespace(disp_solid=disp_sum, f=f_sum, m=m_zero, mesh_nodes=mesh_nodes))
    c_oracle = float(np.ravel(out.compliance.value)[0])

    c_sum_v = float(np.ravel(c_sum.value)[0])
    print(f"\ncompliance  sum-of-loads={c_sum_v:.10e}  work-conjugate-oracle={c_oracle:.10e}")
    assert c_sum_v == pytest.approx(c_oracle, rel=1e-12)


def test_point_load_compliance_matches_displacement_field_at_point(plate_mesh, recorder):
    """point_load's compliance (F.u(x_p), folded into vdot(direct_vector, w) by
    construction -- see hermit/loads.py) matches independently evaluating the
    solved displacement at x_p through plain dolfinx (``Function.eval``), decoupled
    from hermit.loads._point_scatter's own machinery."""
    import dolfinx

    dom = ShellDomain(plate_mesh, element="CG2CG1")
    pde, bc = _raw(dom)
    mat = _isotropic_material(dom, 70e9, 0.3, 0.01, 2700.0, ("Lagrange", 1))
    at = np.array([8.0, 1.3, 0.0])
    F0 = np.array([0.0, 0.0, 750.0])
    loads = hld.point_load(dom, at=at, force=F0)
    c, disp = _solve_new(pde, bc, dom, mat, loads)

    W = dom.W
    Vsub, sub_dofs = W.sub(0).collapse()
    sub_dofs = np.asarray(sub_dofs).reshape(-1)
    disp_sub = np.asarray(disp.value)[sub_dofs]
    cell, ref = hld._locate_point(dom.mesh, at)
    fn = dolfinx.fem.Function(Vsub)
    fn.x.array[:] = disp_sub
    u_at_np = fn.eval(at.reshape(1, -1), np.array([cell], dtype=np.int32))

    work = csdl.vdot(csdl.Variable(value=F0), csdl.Variable(value=u_at_np))
    c_v, work_v = float(np.ravel(c.value)[0]), float(np.ravel(work.value)[0])
    print(f"\ncompliance={c_v:.10e}  F.u(x_p)={work_v:.10e}  rel={abs(c_v - work_v) / abs(c_v):.2e}")
    assert c_v == pytest.approx(work_v, rel=1e-10)


# -- FD checks: d(compliance)/d(force), d(compliance)/d(p) ----------------------

def test_dcompliance_dforce_matches_finite_difference(plate_mesh):
    at = np.array([8.0, 1.3, 0.0])

    def run(Fz, want_grad):
        rec = csdl.Recorder(inline=True); rec.start()
        dom = ShellDomain(plate_mesh, element="CG2CG1")
        pde, bc = _raw(dom)
        mat = _isotropic_material(dom, 70e9, 0.3, 0.01, 2700.0, ("Lagrange", 1))
        F = csdl.Variable(value=np.array([0.0, 0.0, Fz]), name="F")
        loads = hld.point_load(dom, at=at, force=F)
        c, _ = _solve_new(pde, bc, dom, mat, loads)
        val = float(np.ravel(c.value)[0])
        g = None
        if want_grad:
            g = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([c], [F])[c, F])[2])
        rec.stop()
        return val, g

    Fz0 = 750.0
    _, ana = run(Fz0, True)
    step = 1.0
    vp, _ = run(Fz0 + step, False)
    vm, _ = run(Fz0 - step, False)
    fd = (vp - vm) / (2 * step)
    rel = abs(ana - fd) / abs(fd)
    print(f"\nd(compliance)/d(Fz)  analytic={ana:.8e}  fd={fd:.8e}  rel={rel:.2e}")
    assert rel < 1e-6


def test_dcompliance_dp_matches_finite_difference(plate_mesh):
    def run(p0, want_grad):
        rec = csdl.Recorder(inline=True); rec.start()
        dom = ShellDomain(plate_mesh, element="CG2CG1")
        pde, bc = _raw(dom, elementwise_pressure=True, element_wise_material=True)
        nel = dom.n_cells
        mat = _isotropic_material(dom, 70e9 * np.ones(nel), 0.3 * np.ones(nel),
                                  0.01 * np.ones(nel), 2700.0 * np.ones(nel), ("DG", 0))
        p = csdl.Variable(value=float(p0), name="p")
        loads = hld.pressure(dom, p, space=("DG", 0))
        f_field = loads.combined_traction(space=("DG", 0, (3,)))
        mesh_nodes = csdl.Variable(value=dom.node_coords)
        m_zero = csdl.Variable(value=np.zeros(f_field.coeffs.shape))
        op = ShellSolveOp(pde, bc, ("A", "B", "D", "As", "thickness", "f", "m"), None,
                          form_cache=pde._solve_form_cache)
        disp = op.evaluate(SimpleNamespace(A=mat.A.coeffs, B=mat.B.coeffs, D=mat.D.coeffs,
                                           As=mat.As.coeffs, thickness=mat.thickness.coeffs,
                                           f=f_field.coeffs, m=m_zero, mesh_nodes=mesh_nodes))
        sop = ShellScalarFormsOp(pde, {"compliance": (pde.compliance_form(), ("disp_solid", "f", "m"))})
        out = sop.evaluate(SimpleNamespace(disp_solid=disp, f=f_field.coeffs, m=m_zero, mesh_nodes=mesh_nodes))
        c = out.compliance
        val = float(np.ravel(c.value)[0])
        g = None
        if want_grad:
            g = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([c], [p])[c, p])[0])
        rec.stop()
        return val, g

    p0 = 1.0e4
    _, ana = run(p0, True)
    step = 1.0
    vp, _ = run(p0 + step, False)
    vm, _ = run(p0 - step, False)
    fd = (vp - vm) / (2 * step)
    rel = abs(ana - fd) / abs(fd)
    print(f"\nd(compliance)/d(p)  analytic={ana:.8e}  fd={fd:.8e}  rel={rel:.2e}")
    assert rel < 1e-6


# -- Loads.__add__ --------------------------------------------------------------

def test_loads_add_merges_terms(plate_mesh, recorder):
    dom = hermit.ShellDomain(plate_mesh)
    a = hld.traction(dom, [0.0, 0.0, 1.0e3])
    b = hld.moment(dom, [1.0, 0.0, 0.0])
    c = hld.point_load(dom, at=[8.0, 1.3, 0.0], force=[0.0, 0.0, 1.0])
    merged = a + b + c
    assert len(merged.traction_terms) == 1
    assert len(merged.moment_terms) == 1
    assert merged.direct is not None


def test_loads_add_wrong_domain_raises(plate_mesh, tri_mesh, recorder):
    dom1, dom2 = hermit.ShellDomain(plate_mesh), hermit.ShellDomain(tri_mesh)
    a = hld.traction(dom1, [0.0, 0.0, 1.0e3])
    b = hld.traction(dom2, [0.0, 0.0, 1.0e3])
    with pytest.raises(ValueError, match="same ShellDomain"):
        a + b
