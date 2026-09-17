"""The in-form ``Teps(theta)`` orientation rotation in ``ElasticModel``, the
structural form-cache key, and the multi-region penalty sum.

Everything here drives ``ShellPDE`` / ``ShellSolveOp`` / ``ShellScalarFormsOp``
directly (like ``test_pde_raw.py`` / ``test_bcs_api.py``) rather than through the
public ``hm.solve``; it only has to prove the FE-core machinery is right.

Gate 1's reference side (an upstream-``rotate_abd`` path) is the stored fixture
``legacy_ref`` rather than a live call: the ``orientation_form_*`` keys were captured
from that path before it was removed.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit
import hermit.bcs as hbc
from hermit.domain import ShellDomain
from hermit.fenics.ops import ShellScalarFormsOp, ShellSolveOp
from hermit.fenics.shell_pde import ShellPDE
from hermit.outputs import compliance as _compliance
from hermit._solve import solve as _solve
from hermit._laminate import Layup, compute_clt
from conftest import clamped_at_x0


# -- shared helpers -----------------------------------------------------------

def _ud():
    from caddee_materials import TransverseMaterial

    # strongly orthotropic transverse shear (GA != G23) -- makes both the fiber
    # rotation *and* the transverse-shear ordering fix visible in compliance.
    return TransverseMaterial(name="ud", EA=138e9, ET=10e9, vA=0.34, vT=0.4, GA=7e9, density=1.6e3)


def _layup(angles_deg, total_h=0.02):
    n = len(angles_deg)
    a = csdl.Variable(value=np.radians(np.asarray(angles_deg, dtype=float)), name="ply_angles")
    return Layup(_ud(), a, np.full(n, total_h / n), num_plies=n), a


def _oriented_raw(mesh, elementwise=True, element="CG2CG1"):
    """The raw FE operands these gates drive: a ``ShellDomain``, a ``ShellPDE`` on its
    ``W`` (with the elementwise fixed material space the gates' per-cell ABD needs),
    and a penalty-clamp ``BCData`` at ``clamped_at_x0``."""
    dom = ShellDomain(mesh, element=element)
    pde = ShellPDE(mesh, element=element, W=dom.W, element_wise_material=elementwise)
    pde._solve_form_cache = {}
    return dom, pde, hbc.clamp(dom, where=clamped_at_x0).to_bc_data()


def _pressure_vec(pde, pz):
    n = pde.VF.dofmap.index_map.size_local
    arr = np.zeros((n, 3)); arr[:, 2] = pz
    return csdl.Variable(value=arr.reshape(-1))


def _new_path_abd(nel, layup, shear_correction=0.833):
    """Unrotated per-cell A/B/D/As (uniform layup, FE cell order -- broadcasting a
    single layup makes cell order irrelevant), standard (xz, yz) transverse-shear
    ordering. The new in-form path never rotates upstream -- this is exactly what
    ``hm.laminate`` builds, minus the CLT-seam permutation (kept
    inline here so the fix is exercised once, at the seam, not duplicated silently)."""
    A, B, D, A_star = compute_clt(layup)
    P = csdl.Variable(value=np.array([[0.0, 1.0], [1.0, 0.0]]))
    A_star = P @ A_star @ P
    tile2 = lambda M, s: csdl.expand(M, (nel, *s), action="ij->kij")
    thickness = csdl.expand(csdl.reshape(layup.h, (1,)), (nel,))
    return (tile2(A, (3, 3)), tile2(B, (3, 3)), tile2(D, (3, 3)),
           tile2(A_star * shear_correction, (2, 2)), thickness)


def _solve_new_path(dom, pde, bc, nel, layup, pz, *, orientation=None, node_disp=None):
    """The new in-form-rotation path, raw plumbing: unrotated ABD declared as
    ``ShellSolveOp`` args, plus (optionally) an orientation coefficient declared the
    same way -- see ``hermit.fenics.ops.ShellSolveOp``'s ``orientation=`` kwarg.

    ``orientation``: ``None``, or ``(name, space, value)`` with ``name`` in
    ``("fiber_angle", "fiber_direction")`` and ``value`` a ``csdl.Variable`` already
    sized to ``pde.coefficient(name, space).x.array.size``.

    Returns ``(compliance, disp_solid)``.
    """
    A, B, D, As, thickness = _new_path_abd(nel, layup)
    f = _pressure_vec(pde, pz)
    m_ = csdl.Variable(value=np.zeros(f.shape[0]))

    # ShellPDE.set_geometry expects mesh_nodes in *file* order (it gathers by
    # node_input_idx to land in local/FE order) -- ShellDomain.node_coords is that
    # file-order reference (mesh.geometry.x scattered by file id), matching what
    # hm.geometry feeds it. Using mesh.geometry.x (local order) directly here would
    # silently scramble the reference geometry.
    ref_nodes = dom.node_coords
    diff_geom = node_disp is not None
    mesh_nodes = (csdl.Variable(value=ref_nodes) + node_disp) if diff_geom \
        else csdl.Variable(value=ref_nodes)

    arg_names = ["A", "B", "D", "As", "thickness", "f", "m"]
    if diff_geom:
        arg_names.append("mesh_nodes")
    fe_kwargs = dict(A=A, B=B, D=D, As=As, thickness=thickness, f=f, m=m_, mesh_nodes=mesh_nodes)
    orient_arg = None
    if orientation is not None:
        oname, ospace, oval = orientation
        arg_names.append(oname)
        orient_arg = (oname, ospace)
        fe_kwargs[oname] = oval

    op = ShellSolveOp(pde, bc, tuple(arg_names), None, form_cache=pde._solve_form_cache,
                      orientation=orient_arg)
    disp = op.evaluate(SimpleNamespace(**fe_kwargs))

    sop = ShellScalarFormsOp(
        pde, {"compliance": (pde.compliance_form(), ("disp_solid", "f", "m"))},
        differentiable_geometry=diff_geom, orientation=orient_arg)
    fe2 = dict(disp_solid=disp, f=f, m=m_, mesh_nodes=mesh_nodes)
    if orientation is not None:
        fe2[orientation[0]] = orientation[2]
    out = sop.evaluate(SimpleNamespace(**fe2))
    return out.compliance, disp


def _legacy_compliance_and_grad(legacy_ref, mode):
    """The upstream-rotate path (``csdl_helpers.rotate_abd``), the reference
    everything here is checked against -- read from the stored fixture (see the module
    docstring). ``[10, -25, 55] deg`` UD laminate, element_wise_material, penalty
    clamp, uniform +z nodal pressure ``pz=1e3``; ``fiber_angle=0.4`` or
    ``fiber_direction=[0, 1, 0]``."""
    c = float(legacy_ref[f"orientation_form_{mode}__compliance"])
    key = f"orientation_form_{mode}__dcompliance_dangles"
    g = legacy_ref[key] if key in legacy_ref else None
    return c, g


# -- gate 1: equivalence with the legacy pre-rotate path -----------------------
# Flat plate mesh -> the legacy path's per-cell-centroid frame equals the new path's
# quadrature-point frame exactly, so the *rotation math* agrees exactly (verified: an
# earlier version of this gate found fiber_angle matching legacy to 0.00e+00). But
# exactness there does not carry over to an exact match against the *legacy reference*
# for either orientation kind, for a reason that has nothing to do with frames or
# rotation formulas -- see the docstrings below (ElasticModel._ORIENTED_QUADRATURE_DEGREE
# has the full account). Both sub-cases are checked to the same, explained, ~1e-6
# residual instead.

def _run_gate1_new(plate_mesh, mode, layup, angles, nel, pz, theta_val, direction, rec):
    dom, pde, bc = _oriented_raw(plate_mesh)
    if mode == "fiber_angle":
        space = ("DG", 0)
        n = pde.coefficient("fiber_angle", space).x.array.size
        oval = csdl.expand(csdl.Variable(value=theta_val), (n,))
    else:
        space = ("DG", 0, (3,))
        n = pde.coefficient("fiber_direction", space).x.array.size
        oval = csdl.Variable(value=np.tile(np.asarray(direction, dtype=float), n // 3))
    c, _ = _solve_new_path(dom, pde, bc, nel, layup, pz, orientation=(mode, space, oval))
    g = csdl.experimental.PySimulator(rec).compute_totals([c], [angles])[c, angles]
    return float(np.ravel(c.value)[0]), np.asarray(g).ravel()


def test_in_form_rotation_matches_legacy_fiber_angle_compliance_and_gradient(plate_mesh, legacy_ref):
    """``fiber_angle`` -- theta is given directly (no atan2 round-trip anywhere).

    This does *not* match the legacy pre-rotate path to machine precision, and the
    reason is quadrature, not rotation math. The legacy path rotates the ABD upstream
    (numpy) and feeds a plain DG0 coefficient into the same *unoriented*-shaped form,
    which UFL auto-estimates a quadrature degree for (measured: 18/30/30/28 per energy
    term on this fixture). The new in-form path's oriented energy integral instead
    carries an explicit, deliberately chosen ``_ORIENTED_QUADRATURE_DEGREE = 4`` (see
    ``hermit.fenics.elastic_model`` -- the point is that the answer must no longer
    silently depend on UFL's degree-estimation heuristic, which is what broke
    fiber_direction, below). Degree 4 and UFL's auto degree are two different (both
    "reasonable") quadrature rules for an integrand that is not exactly polynomial
    (CellDiameter-based drilling stabilization, a bilinear quad's non-affine terms) --
    they converge to two slightly different answers. Measured: ~1e-7 to ~4e-7 relative,
    stable from degree 4 through 30 (i.e. not a convergence issue -- picking a higher
    explicit degree does not close this gap, it is intrinsic to comparing two different
    quadrature rules on a non-polynomial integrand). d(compliance)/d(ply angles) is
    checked at the same tolerance for the same reason.
    """
    nel = plate_mesh.topology.index_map(2).size_local
    pz, ang0 = 1.0e3, [10.0, -25.0, 55.0]

    def run_new():
        rec = csdl.Recorder(inline=True); rec.start()
        layup, angles = _layup(ang0)
        val, grad = _run_gate1_new(plate_mesh, "fiber_angle", layup, angles, nel, pz, 0.4, None, rec)
        rec.stop()
        return val, grad

    c_legacy, g_legacy = _legacy_compliance_and_grad(legacy_ref, "fiber_angle")
    c_new, g_new = run_new()
    rel_c = abs(c_new - c_legacy) / abs(c_legacy)
    rel_g = np.linalg.norm(g_new - g_legacy) / np.linalg.norm(g_legacy)
    print(f"\n[fiber_angle] compliance  legacy={c_legacy:.12e}  new={c_new:.12e}  rel={rel_c:.2e}")
    print(f"[fiber_angle] d(compliance)/d(ply angles)  rel(||.||)={rel_g:.2e}  "
         f"legacy={g_legacy}  new={g_new}")
    assert rel_c < 1e-6
    assert rel_g < 1e-6


def test_in_form_rotation_matches_legacy_fiber_direction_compliance(plate_mesh, legacy_ref):
    """``fiber_direction`` -- compliance only (see below for why the gradient isn't a
    meaningful comparison here).

    Earlier investigation of this gate wrongly blamed an ``atan2`` round-trip in the
    legacy reference (falsified: the gap is present, and larger, at generic angles
    like 30 deg where cos/sin have no special-value residue, and its sign is not
    consistent with that mechanism). The real cause, confirmed numerically: UFL's
    automatic quadrature-degree estimator does not treat the tangent-plane projection
    in ``orientation_cos_sin`` (a ``sqrt`` + division, from ``unit()``) as "the same
    order" as ``fiber_angle``'s plain ``cos``/``sin`` of a coefficient -- estimated
    degree per energy term was 46-86 for the direction form vs 18-30 for the angle/
    unoriented forms (measured via ``ufl.algorithms.compute_form_data`` with
    ``do_estimate_degrees=True`` on this fixture). Forcing the *same* explicit degree
    on both oriented forms (``_ORIENTED_QUADRATURE_DEGREE``, see
    ``hermit.fenics.elastic_model``) brings fiber_angle and fiber_direction into
    agreement with *each other* to ~1e-9 at every angle tested (30/45/90 deg) -- down
    from ~1.2e-6 -- confirming Teps/R/congruent_transform/the coefficient plumbing were
    never the issue; only the quadrature degree of this one branch was.

    That fix does not, and cannot, bring fiber_angle itself to an exact match against
    the legacy reference either (see the fiber_angle test above) -- pinning a degree
    necessarily departs from whatever degree UFL happened to auto-estimate for the
    structurally different legacy-shaped form. So post-fix, fiber_direction's residual
    against legacy is the *same* ~1e-7 order as fiber_angle's, for the same
    quadrature-rule-choice reason -- not a residual "direction-specific" defect.

    The legacy path's ply-angle gradient is *not* checked here: ``rotate_abd``
    detaches the CSDL graph whenever ``theta`` isn't itself a ``csdl.Variable``
    (``hermit.csdl_helpers.orientation_angle`` always returns plain numpy), which is
    unconditionally true for ``fiber_direction`` -- so the legacy
    d(compliance)/d(ply angles) is identically zero here regardless of direction, a
    limitation of that path, not something the in-form rotation changes.
    ``fiber_direction``'s dR/d(theta)-equivalent and shape derivative are
    checked against finite differences instead (gates 4 and 5, below).
    """
    nel = plate_mesh.topology.index_map(2).size_local
    pz, ang0, direction = 1.0e3, [10.0, -25.0, 55.0], [0.0, 1.0, 0.0]

    def run_new():
        rec = csdl.Recorder(inline=True); rec.start()
        layup, angles = _layup(ang0)
        val, _ = _run_gate1_new(plate_mesh, "fiber_direction", layup, angles, nel, pz,
                                None, direction, rec)
        rec.stop()
        return val

    c_legacy, _ = _legacy_compliance_and_grad(legacy_ref, "fiber_direction")
    c_new = run_new()
    rel_c = abs(c_new - c_legacy) / abs(c_legacy)
    print(f"\n[fiber_direction] compliance  legacy={c_legacy:.12e}  new={c_new:.12e}  rel={rel_c:.2e}")
    assert rel_c < 1e-6


# -- gate 2 (isotropic unchanged) is covered by the unmodified suite --------
# (test_pde_raw.py / test_solve.py / test_post.py); nothing new to add here.


# -- gate 3: an orientation field on a continuous (CG) space -------------------

def test_fiber_angle_on_cg_space_matches_uniform_dg_case(plate_mesh):
    """A uniform value is physically the same field whether it lives on CG1 or DG0 --
    exercising 'any space' this way gives an actual numeric oracle, not just 'ran
    without raising'."""
    nel = plate_mesh.topology.index_map(2).size_local
    pz, theta0 = 1.0e3, 0.25

    def run(space):
        rec = csdl.Recorder(inline=True); rec.start()
        dom, pde, bc = _oriented_raw(plate_mesh)
        layup, _ = _layup([15.0, -40.0, 5.0])
        n = pde.coefficient("fiber_angle", space).x.array.size
        oval = csdl.Variable(value=np.full(n, theta0))
        c, disp = _solve_new_path(dom, pde, bc, nel, layup, pz, orientation=("fiber_angle", space, oval))
        val, disp_val = float(np.ravel(c.value)[0]), np.asarray(disp.value).copy()
        rec.stop()
        return val, disp_val

    c_dg, w_dg = run(("DG", 0))
    c_cg, w_cg = run(("Lagrange", 1))
    rel = abs(c_cg - c_dg) / c_dg
    print(f"\ncompliance  DG0={c_dg:.10e}  CG1={c_cg:.10e}  rel={rel:.2e}")
    assert np.isfinite(c_cg) and c_cg > 0
    # Both spaces' oriented energy form share the same explicit
    # ElasticModel._ORIENTED_QUADRATURE_DEGREE (see that module) -- before that fix
    # this gap was 8.58e-08 (UFL auto-estimating a different degree for a DG0- vs a
    # CG1-valued fiber_angle coefficient, the same class of bug gate 1's
    # fiber_direction case hit, just milder here since both branches use plain
    # cos/sin, no sqrt); pinning the degree brought it down to ~1e-9, consistent with
    # ordinary partition-of-unity float round-off (CG1's basis functions sum to 1.0
    # only up to rounding) rather than a quadrature-rule mismatch. 1e-7 keeps
    # comfortable margin above the observed noise floor while still being a real
    # numeric check, not just "finite and positive".
    assert c_cg == pytest.approx(c_dg, rel=1e-7)
    assert np.allclose(w_cg, w_dg, rtol=1e-7, atol=1e-10)


# -- gate 4: shape derivative with a fiber_direction orientation ---------------
# Mirrors test_mesh_coord_deriv.py's directional-central-difference pattern.

def _smooth_fields(gx, nn, L=10.0, W=2.0):
    bend = np.zeros((nn, 3))
    bend[:, 2] = 0.3 * (gx[:, 0] / L) ** 2
    bend[:, 0] = 0.02 * (gx[:, 0] / L)
    V = np.zeros((nn, 3))
    V[:, 2] = np.sin(np.pi * gx[:, 0] / L) * (0.5 + gx[:, 1] / W)
    V[:, 0] = 0.3 * gx[:, 0] / L
    V /= np.linalg.norm(V)
    return bend, V


def test_shape_derivative_wrt_node_disp_with_fiber_direction(plate_mesh):
    nel = plate_mesh.topology.index_map(2).size_local
    nn = plate_mesh.geometry.x.shape[0]
    gx = plate_mesh.geometry.x.copy()
    bend, Vdir = _smooth_fields(gx, nn)
    pz = 5.0e2

    def run(nd_val, want_grad):
        assert np.allclose(plate_mesh.geometry.x, gx), "an op left mesh.geometry.x moved"
        rec = csdl.Recorder(inline=True); rec.start()
        dom, pde, bc = _oriented_raw(plate_mesh)
        layup, _ = _layup([10.0, -25.0, 55.0])
        space = ("DG", 0, (3,))
        n = pde.coefficient("fiber_direction", space).x.array.size
        oval = csdl.Variable(value=np.tile([0.0, 1.0, 0.3], n // 3))
        nd = csdl.Variable(value=nd_val.copy(), name="node_disp")
        c, _ = _solve_new_path(dom, pde, bc, nel, layup, pz,
                               orientation=("fiber_direction", space, oval), node_disp=nd)
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
    print(f"\nd(compliance)/d(node_disp) . V [fiber_direction] : "
         f"ana={dd_ana:+.6e}  fd={dd_fd:+.6e}  rel={rel:.2e}")
    assert rel < 3e-3


# -- gate 5: dR/dtheta (d compliance / d fiber_angle) vs central FD -----------

def test_compliance_derivative_wrt_fiber_angle_vs_finite_difference(plate_mesh):
    nel = plate_mesh.topology.index_map(2).size_local
    pz = 1.0e3

    def run(theta0, want_grad):
        rec = csdl.Recorder(inline=True); rec.start()
        dom, pde, bc = _oriented_raw(plate_mesh)
        layup, _ = _layup([10.0, -25.0, 55.0])
        space = ("DG", 0)
        n = pde.coefficient("fiber_angle", space).x.array.size
        theta = csdl.Variable(value=np.full(n, theta0), name="theta")
        c, _ = _solve_new_path(dom, pde, bc, nel, layup, pz, orientation=("fiber_angle", space, theta))
        val = float(np.ravel(c.value)[0])
        g = None
        if want_grad:
            g = np.asarray(csdl.experimental.PySimulator(rec).compute_totals([c], [theta])[c, theta]).ravel()
        rec.stop()
        return val, g

    theta0 = 0.35
    _, g = run(theta0, True)
    assert np.all(np.isfinite(g)) and np.linalg.norm(g) > 0
    # theta is uniform (same value on every dof) -- a directional derivative along the
    # all-ones direction is just sum(g)
    ana = float(g.sum())
    # this material is strongly anisotropic (EA/ET ~ 14), so a too-small step is
    # dominated by MUMPS/round-off noise in the compliance rather than truncation
    # error -- empirically 1e-3 is in the well-converged regime (1e-4..1e-3 agree to
    # <0.01%; 1e-6 is already noise-dominated, off by ~0.3%). Same order of step as
    # test_mesh_coord_deriv.py's mesh-coordinate FD, for the same reason.
    step = 1e-3
    vp, _ = run(theta0 + step, False)
    vm, _ = run(theta0 - step, False)
    fd = (vp - vm) / (2 * step)
    rel = abs(ana - fd) / abs(fd)
    print(f"\nd(compliance)/d(fiber_angle) . 1 : analytic={ana:.8e}  fd={fd:.8e}  rel={rel:.2e}")
    assert rel < 1e-4


# -- structural form-cache key --------------------------------------------------

def test_structurally_different_compositions_on_one_model_get_different_forms(plate_mesh):
    """One domain / one ShellPDE (so one form cache), three structurally different
    solves: no orientation, fiber_angle on DG0, fiber_direction on DG0 vector -- each
    must compile its own residual (never silently reuse another composition's), and
    all three must still give physically-correct answers."""
    nel = plate_mesh.topology.index_map(2).size_local
    pz = 1.0e3

    rec = csdl.Recorder(inline=True); rec.start()
    dom, pde, bc = _oriented_raw(plate_mesh)
    cache = pde._solve_form_cache
    layup, _ = _layup([10.0, -25.0, 55.0])

    def _residual(arg_names, orientation=None):
        return ShellSolveOp(pde, bc, tuple(arg_names), None, form_cache=cache,
                            orientation=orientation).residual

    c_plain, _ = _solve_new_path(dom, pde, bc, nel, layup, pz, orientation=None)
    op_plain_residual = _residual(["A", "B", "D", "As", "thickness", "f", "m"])

    space_a = ("DG", 0)
    n_a = pde.coefficient("fiber_angle", space_a).x.array.size
    theta_val = csdl.Variable(value=np.full(n_a, 0.4))
    c_angle, _ = _solve_new_path(dom, pde, bc, nel, layup, pz, orientation=("fiber_angle", space_a, theta_val))
    op_angle_residual = _residual(
        ["A", "B", "D", "As", "thickness", "f", "m", "fiber_angle"],
        orientation=("fiber_angle", space_a))

    space_d = ("DG", 0, (3,))
    n_d = pde.coefficient("fiber_direction", space_d).x.array.size
    dir_val = csdl.Variable(value=np.tile([0.0, 1.0, 0.0], n_d // 3))
    c_dir, _ = _solve_new_path(dom, pde, bc, nel, layup, pz, orientation=("fiber_direction", space_d, dir_val))
    op_dir_residual = _residual(
        ["A", "B", "D", "As", "thickness", "f", "m", "fiber_direction"],
        orientation=("fiber_direction", space_d))

    c_plain_v = float(np.ravel(c_plain.value)[0])
    c_angle_v = float(np.ravel(c_angle.value)[0])
    c_dir_v = float(np.ravel(c_dir.value)[0])
    rec.stop()

    print(f"\ncompliance  plain={c_plain_v:.6e}  fiber_angle={c_angle_v:.6e}  "
         f"fiber_direction={c_dir_v:.6e}")
    print(f"cache size (this ShellPDE) = {len(cache)}")

    # three structurally distinct compositions -> three distinct compiled residuals,
    # never a silently-reused wrong one
    assert op_plain_residual is not op_angle_residual
    assert op_plain_residual is not op_dir_residual
    assert op_angle_residual is not op_dir_residual
    assert len(cache) == 3

    # and each gives the physically correct answer: orientation must actually change
    # the compliance (not silently fall back to the unoriented form). Exact agreement
    # with the legacy pre-rotate path (this composition's oracle) is gate 1, above --
    # same numbers, checked there with its own recorder.
    assert c_angle_v != pytest.approx(c_plain_v, rel=1e-6)
    assert c_dir_v != pytest.approx(c_plain_v, rel=1e-6)


# -- multi-region penalty ------------------------------------------------------

def test_multi_region_penalty_sums_to_the_union_region(plate_mesh):
    """Two disjoint same-mask clamp regions (hm.clamp(A) + hm.clamp(B)) must equal a
    single clamp over their union -- ds/dS are additive over disjoint entity sets, so
    this is an independent numeric oracle for ElasticModel._penalty_residual's
    multi-term sum, not just 'runs without raising'."""
    from hermit.bcs import clamp

    x = plate_mesh.geometry.x
    x0 = x[np.isclose(x[:, 0], 0.0, atol=1e-9)]
    ymid = float(np.median(x0[:, 1]))
    where_lo = lambda xx: np.logical_and(np.less(xx[0], 1e-12), np.less_equal(xx[1], ymid))
    where_hi = lambda xx: np.logical_and(np.less(xx[0], 1e-12), np.greater_equal(xx[1], ymid))

    def run(wheres):
        rec = csdl.Recorder(inline=True); rec.start()
        dom = ShellDomain(plate_mesh, element="CG2CG1")
        bcs = None
        for w in wheres:
            term = clamp(dom, where=w)
            bcs = term if bcs is None else bcs + term

        nn = plate_mesh.geometry.x.shape[0]
        E, nu, h, rho, pz = 70e9, 0.3, 0.01, 2700.0, 1.0e4
        mat = hermit.isotropic(dom, E=E * np.ones(nn), nu=nu * np.ones(nn),
                               thickness=h * np.ones(nn), density=rho * np.ones(nn),
                               constitutive_space=("Lagrange", 1))
        p = np.zeros((nn, 3)); p[:, 2] = pz
        loads = hermit.traction(dom, hermit.from_nodal(dom, p), space=("Lagrange", 1))
        state = _solve(dom, mat, loads, bcs)
        val = float(np.ravel(_compliance(state).value)[0])
        disp = np.asarray(state.disp_solid.value).copy()
        rec.stop()
        return val, disp, bcs

    c_union, w_union, bcs_union = run([clamped_at_x0])
    c_split, w_split, bcs_split = run([where_lo, where_hi])
    assert len(bcs_union.penalty_terms) == 1
    assert len(bcs_split.penalty_terms) == 2   # a genuine multi-term BCData.penalty_terms case
    disp_diff = np.abs(w_split - w_union).max()
    print(f"\ncompliance  union={c_union:.12e}  split(2 terms)={c_split:.12e}  "
         f"rel={abs(c_union - c_split) / c_union:.2e}")
    print(f"max|disp_split - disp_union| = {disp_diff:.3e}  (max|disp|={np.abs(w_union).max():.3e})")
    assert c_split == pytest.approx(c_union, rel=1e-10)
    # not bit-identical: the two-term assembly touches the same facets through a
    # differently-ordered sum of matrix entries (a separate Form per term, added
    # together) than the single-term assembly's one Form -- floating-point addition
    # isn't associative. 1e-6 comfortably clears that reordering noise.
    assert np.allclose(w_split, w_union, rtol=1e-6, atol=1e-9)


def test_multi_region_penalty_mixed_masks_end_to_end(plate_mesh):
    """hm.clamp(...) + hm.symmetry(...) on disjoint regions actually composes: it must
    give a *different*, still finite and sane, answer than either term alone --
    proving both terms' contributions reach the residual, not just the last one
    added."""
    from hermit.bcs import clamp, symmetry

    x = plate_mesh.geometry.x
    x0 = x[np.isclose(x[:, 0], 0.0, atol=1e-9)]
    ymid = float(np.median(x0[:, 1]))
    where_lo = lambda xx: np.logical_and(np.less(xx[0], 1e-12), np.less_equal(xx[1], ymid))
    where_hi = lambda xx: np.logical_and(np.less(xx[0], 1e-12), np.greater_equal(xx[1], ymid))

    def run(bcs_builder):
        rec = csdl.Recorder(inline=True); rec.start()
        dom = ShellDomain(plate_mesh, element="CG2CG1")

        nn = plate_mesh.geometry.x.shape[0]
        E, nu, h, rho, pz = 70e9, 0.3, 0.01, 2700.0, 1.0e4
        mat = hermit.isotropic(dom, E=E * np.ones(nn), nu=nu * np.ones(nn),
                               thickness=h * np.ones(nn), density=rho * np.ones(nn),
                               constitutive_space=("Lagrange", 1))
        p = np.zeros((nn, 3)); p[:, 2] = pz
        loads = hermit.traction(dom, hermit.from_nodal(dom, p), space=("Lagrange", 1))
        state = _solve(dom, mat, loads, bcs_builder(dom))
        val = float(np.ravel(_compliance(state).value)[0])
        rec.stop()
        return val

    c_clamp_only = run(lambda dom: clamp(dom, where=where_lo))
    c_sym_only = run(lambda dom: symmetry(dom, where=where_hi, normal=[1, 0, 0]))
    c_mixed = run(lambda dom: clamp(dom, where=where_lo) + symmetry(dom, where=where_hi, normal=[1, 0, 0]))
    print(f"\ncompliance  clamp-only={c_clamp_only:.6e}  symmetry-only={c_sym_only:.6e}  "
         f"mixed(both)={c_mixed:.6e}")
    assert np.isfinite(c_mixed) and c_mixed > 0
    # the extra constraint from combining both regions can only stiffen the structure
    # relative to either one alone (a strict subset of active constraints)
    assert c_mixed < c_clamp_only
    # No `c_mixed < c_sym_only`: symmetry alone on half an edge pins only ux/ry/rz and
    # leaves the rigid-body modes free, so that solve is singular and its compliance is
    # an arbitrary null-space artifact -- not a physical value. The strict
    # inequality above already proves the symmetry term reached the residual: if it had
    # not, c_mixed would equal c_clamp_only rather than being below it.
