"""``hermit.transfer`` -- ``interpolate`` / ``project`` space-to-space transfer of a
``Field`` on the same mesh.

``interpolate`` is a constant, geometry-independent sparse matrix; ``project`` is an
L2 projection, geometry-dependent through the ``dx`` measure on both sides. The
mesh-coordinate tests here use ``_FakeGeometry``, a minimal stand-in for the
structural contract ``project``'s ``geometry=`` keyword documents: any object with a
``.nodes`` attribute holding an ``(n_nodes, 3)`` file-order ``csdl.Variable``.
"""

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit
from hermit.transfer import interpolate, project, _interp_matrix


@pytest.fixture(scope="module")
def domain(plate_mesh):
    return hermit.ShellDomain(plate_mesh)


class _FakeGeometry:
    """Structural stand-in for ``hermit.geometry.Geometry`` -- just the ``.domain`` /
    ``.nodes`` contract ``project``'s ``geometry=`` keyword documents."""

    def __init__(self, domain, nodes):
        self.domain = domain
        self.nodes = nodes


def _smooth_bend_and_direction(domain, L=10.0, W=2.0):
    """Same construction as ``tests/test_mesh_coord_deriv.py::_smooth_fields`` -- a
    gentle bend + a smooth perturbation direction, so the FD step stays well
    conditioned."""
    gx = domain.node_coords
    nn = domain.n_nodes
    bend = np.zeros((nn, 3))
    bend[:, 2] = 0.3 * (gx[:, 0] / L) ** 2
    bend[:, 0] = 0.02 * (gx[:, 0] / L)
    Vdir = np.zeros((nn, 3))
    Vdir[:, 2] = np.sin(np.pi * gx[:, 0] / L) * (0.5 + gx[:, 1] / W)
    Vdir[:, 0] = 0.3 * gx[:, 0] / L
    Vdir /= np.linalg.norm(Vdir)
    return bend, Vdir


# --- exactness: a field already in the target space is reproduced exactly --------

def test_interpolate_linear_cg1_to_cg2_is_exact(domain, recorder):
    xy = domain.dof_coords(("Lagrange", 1))
    f = hermit.from_coeffs(domain, ("Lagrange", 1), 2.0 * xy[:, 0] + 3.0 * xy[:, 1] + 1.0)
    got = interpolate(f, ("Lagrange", 2))
    xy2 = domain.dof_coords(("Lagrange", 2))
    want = 2.0 * xy2[:, 0] + 3.0 * xy2[:, 1] + 1.0
    assert np.allclose(np.asarray(got.coeffs.value), want, atol=1e-12)


def test_interpolate_dg0_to_dg1_constant_is_exact(domain, recorder):
    vals_file = np.arange(domain.n_cells, dtype=float) * 0.7 + 1.0
    f = hermit.from_cells(domain, vals_file)
    got = interpolate(f, ("DG", 1))
    # every DG1 dof in a cell must hold that cell's DG0 (FE-order) value
    V1 = domain.function_space(("DG", 1))
    dofmap = np.asarray(V1.dofmap.list)
    cell_of_dof = np.empty(V1.dofmap.index_map.size_local, dtype=np.int64)
    cell_of_dof[dofmap.ravel()] = np.repeat(np.arange(dofmap.shape[0]), dofmap.shape[1])
    want = np.asarray(f.coeffs.value)[cell_of_dof]
    assert np.allclose(np.asarray(got.coeffs.value), want, atol=1e-12)


def test_project_same_space_is_identity_up_to_roundoff(domain, recorder):
    xy = domain.dof_coords(("Lagrange", 2))
    vals = np.sin(xy[:, 0]) + xy[:, 1] ** 2
    f = hermit.from_coeffs(domain, ("Lagrange", 2), vals)
    got = project(f, ("Lagrange", 2))
    assert np.allclose(np.asarray(got.coeffs.value), vals, atol=1e-9)


def test_project_linear_cg1_to_cg2_is_exact(domain, recorder):
    xy = domain.dof_coords(("Lagrange", 1))
    f = hermit.from_coeffs(domain, ("Lagrange", 1), 2.0 * xy[:, 0] + 3.0 * xy[:, 1] + 1.0)
    got = project(f, ("Lagrange", 2))
    xy2 = domain.dof_coords(("Lagrange", 2))
    want = 2.0 * xy2[:, 0] + 3.0 * xy2[:, 1] + 1.0
    assert np.allclose(np.asarray(got.coeffs.value), want, atol=1e-8)


# --- both directions, both transfers ----------------------------------------------

@pytest.mark.parametrize("src_space,tgt_space", [
    (("DG", 0, ()), ("Lagrange", 1, ())),      # DG0 -> CG1
    (("Lagrange", 1, ()), ("DG", 0, ())),      # CG1 -> DG0
    (("Lagrange", 1, ()), ("Lagrange", 2, ())),  # CG1 -> CG2
    (("Lagrange", 2, ()), ("DG", 2, ())),      # CG2 -> DG2
])
def test_interpolate_and_project_run_both_directions(domain, recorder, src_space, tgt_space):
    xy = domain.dof_coords(src_space)
    vals = 0.3 * xy[:, 0] - 0.1 * xy[:, 1] + 2.0
    f = hermit.from_coeffs(domain, src_space, vals)

    fi = interpolate(f, tgt_space)
    assert fi.space == tgt_space
    assert np.isfinite(np.asarray(fi.coeffs.value)).all()

    fp = project(f, tgt_space)
    assert fp.space == tgt_space
    assert np.isfinite(np.asarray(fp.coeffs.value)).all()


def test_transfer_vector_valued_space(domain, recorder):
    src_space = ("DG", 0, (3,))
    tgt_space = ("Lagrange", 1, (3,))
    n = domain.function_space(src_space).dofmap.index_map.size_local
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(n, 3))
    f = hermit.from_coeffs(domain, src_space, vals)

    fi = interpolate(f, tgt_space)
    assert fi.coeffs.shape == (domain.function_space(tgt_space).dofmap.index_map.size_local * 3,)

    fp = project(f, tgt_space)
    assert fp.coeffs.shape == fi.coeffs.shape


# --- interpolate: constant matrix, memoized, exactly linear -----------------------

def test_interpolate_matrix_is_memoized(domain, recorder):
    a = hermit.constant(domain, ("Lagrange", 1), 1.0)
    interpolate(a, ("Lagrange", 2))
    P1 = _interp_matrix(domain, ("Lagrange", 1, ()), ("Lagrange", 2, ()))
    P2 = _interp_matrix(domain, ("Lagrange", 1, ()), ("Lagrange", 2, ()))
    assert P1 is P2


def test_interpolate_is_exactly_linear(domain, recorder):
    xy = domain.dof_coords(("Lagrange", 1))
    f = hermit.from_coeffs(domain, ("Lagrange", 1), xy[:, 0])
    g = hermit.from_coeffs(domain, ("Lagrange", 1), xy[:, 1] ** 2 + 1.0)   # not in the CG2 target exactly

    combo = hermit.field_fn(f, g, fn=lambda x, y: 2.0 * x - 0.5 * y)
    lhs = interpolate(combo, ("Lagrange", 2))
    rhs_coeffs = 2.0 * interpolate(f, ("Lagrange", 2)).coeffs - 0.5 * interpolate(g, ("Lagrange", 2)).coeffs
    assert np.allclose(np.asarray(lhs.coeffs.value), np.asarray(rhs_coeffs.value), atol=1e-12)


def test_interpolate_returns_same_field_for_matching_space(domain, recorder):
    f = hermit.constant(domain, ("DG", 0), 3.0)
    assert interpolate(f, ("DG", 0)) is f


# --- differentiability wrt source coefficients, both transfers -------------------

def test_interpolate_derivative_wrt_src_coeffs_vs_fd(domain):
    xy = domain.dof_coords(("Lagrange", 2))
    base = np.sin(0.3 * xy[:, 0]) + 0.2 * xy[:, 1]

    def run(scale, grad):
        rec = csdl.Recorder(inline=True); rec.start()
        c = csdl.Variable(value=scale * base, name="c")
        f = hermit.from_coeffs(domain, ("Lagrange", 2), c)
        out = interpolate(f, ("DG", 2))
        o = csdl.sum(out.coeffs ** 2)
        val = float(np.ravel(o.value)[0])
        g = None
        if grad:
            g = np.asarray(csdl.experimental.PySimulator(rec).compute_totals([o], [c])[o, c]).ravel()
        rec.stop()
        return val, g

    _, g = run(1.0, True)
    d = 1e-6
    vp, _ = run(1.0 + d, False)
    vm, _ = run(1.0 - d, False)
    fd = (vp - vm) / (2 * d)
    an = float(g @ base)
    rel = abs(an - fd) / abs(fd)
    print(f"\ninterpolate d(sum c^2)/d(src)  an={an:+.6e}  fd={fd:+.6e}  rel={rel:.2e}")
    assert rel < 1e-6


def test_project_derivative_wrt_src_coeffs_vs_fd(domain):
    xy = domain.dof_coords(("Lagrange", 2))
    base = np.sin(0.3 * xy[:, 0]) + 0.2 * xy[:, 1]

    def run(scale, grad):
        rec = csdl.Recorder(inline=True); rec.start()
        c = csdl.Variable(value=scale * base, name="c")
        f = hermit.from_coeffs(domain, ("Lagrange", 2), c)
        out = project(f, ("DG", 1))
        o = csdl.sum(out.coeffs ** 2)
        val = float(np.ravel(o.value)[0])
        g = None
        if grad:
            g = np.asarray(csdl.experimental.PySimulator(rec).compute_totals([o], [c])[o, c]).ravel()
        rec.stop()
        return val, g

    _, g = run(1.0, True)
    d = 1e-6
    vp, _ = run(1.0 + d, False)
    vm, _ = run(1.0 - d, False)
    fd = (vp - vm) / (2 * d)
    an = float(g @ base)
    rel = abs(an - fd) / abs(fd)
    print(f"\nproject d(sum c^2)/d(src)  an={an:+.6e}  fd={fd:+.6e}  rel={rel:.2e}")
    assert rel < 1e-6


# --- project's mesh-coordinate derivative -----------------------------------------

def test_project_mesh_coordinate_derivative_vs_central_difference(domain):
    """d(sum(project(f).coeffs**2))/d(node_disp) . V vs central FD -- the risky
    quotient-rule path (``d(Mc)/dX`` on top of ``dL/dX``). Mirrors
    ``tests/test_mesh_coord_deriv.py``'s structure and tolerance. The target space
    is deliberately ('Lagrange', 1): a target that exactly reproduces the source
    per cell (e.g. projecting onto the source's own space, or same-family /
    same-degree) makes the projected *values* insensitive to the geometry-induced
    quadrature weight (the weighted L2 best-fit of an exactly-representable function
    doesn't depend on the weight) -- a genuine, non-buggy degeneracy, just not a
    useful gradient check."""
    gx = domain.node_coords
    orig_x = domain.mesh.geometry.x.copy()
    bend, Vdir = _smooth_bend_and_direction(domain)
    xy = domain.dof_coords(("Lagrange", 2))
    src_vals = np.sin(0.7 * xy[:, 0]) + 0.4 * xy[:, 1] ** 2   # fixed, geometry-independent coeffs

    def run(nd_val, want_grad):
        rec = csdl.Recorder(inline=True); rec.start()
        nd = csdl.Variable(value=nd_val.copy(), name="nd")
        geom = _FakeGeometry(domain, csdl.Variable(value=gx) + nd)
        f = hermit.from_coeffs(domain, ("Lagrange", 2), src_vals)
        out = project(f, ("Lagrange", 1), geometry=geom)
        o = csdl.sum(out.coeffs ** 2)
        val = float(np.ravel(o.value)[0])
        g = None
        if want_grad:
            g = np.asarray(csdl.experimental.PySimulator(rec).compute_totals(
                [o], [nd])[o, nd]).reshape(domain.n_nodes, 3)
        rec.stop()
        return val, g

    _, ana = run(bend, True)
    dd_ana = float((ana * Vdir).sum())
    step = 1e-3
    vp, _ = run(bend + step * Vdir, False)
    vm, _ = run(bend - step * Vdir, False)
    dd_fd = (vp - vm) / (2 * step)
    rel = abs(dd_ana - dd_fd) / max(abs(dd_fd), 1e-10)
    print(f"\nd(project sum c^2)/d(node_disp).V  ana={dd_ana:+.6e}  fd={dd_fd:+.6e}  rel={rel:.2e}")
    assert rel < 3e-3
    assert np.allclose(domain.mesh.geometry.x, orig_x), "an op left mesh.geometry.x moved"


def test_interpolate_has_no_geometry_dependence(domain):
    """Contrast: interpolate's matrix is built from *reference* coordinates, so a
    functional of an interpolated field has zero mesh-coordinate derivative."""
    gx = domain.node_coords
    bend, Vdir = _smooth_bend_and_direction(domain)
    xy = domain.dof_coords(("Lagrange", 2))
    src_vals = np.sin(0.7 * xy[:, 0]) + 0.4 * xy[:, 1] ** 2

    rec = csdl.Recorder(inline=True); rec.start()
    nd = csdl.Variable(value=bend.copy(), name="nd")   # unused by interpolate, but part of the graph
    f = hermit.from_coeffs(domain, ("Lagrange", 2), src_vals)
    out = interpolate(f, ("DG", 1))
    o = csdl.sum(out.coeffs ** 2)
    jac = csdl.experimental.PySimulator(rec).compute_totals([o], [nd])[o, nd]
    rec.stop()
    assert np.allclose(np.asarray(jac), 0.0)


# --- conservation: project preserves the integral onto a coarser space -----------

def test_project_conserves_integral_interpolate_does_not(domain, recorder):
    import ufl
    import dolfinx
    import hermit.fenics.assembly as fa

    xy = domain.dof_coords(("Lagrange", 2))
    vals = np.sin(0.3 * xy[:, 0]) + 0.2 * xy[:, 1]
    f = hermit.from_coeffs(domain, ("Lagrange", 2), vals)

    V = domain.function_space(("Lagrange", 2))
    fn = dolfinx.fem.Function(V)
    fn.x.array[:] = vals
    true_integral = fa.assemble_scalar(fn * ufl.dx)

    V0 = domain.function_space(("DG", 0))
    cell_vol = fa.assemble_vector(ufl.TestFunction(V0) * ufl.dx)

    projected = project(f, ("DG", 0))
    proj_integral = float(np.sum(np.asarray(projected.coeffs.value) * cell_vol))
    assert abs(proj_integral - true_integral) < 1e-8 * abs(true_integral)

    interpolated = interpolate(f, ("DG", 0))
    interp_integral = float(np.sum(np.asarray(interpolated.coeffs.value) * cell_vol))
    assert abs(interp_integral - true_integral) > 1e-4 * abs(true_integral)


# --- error paths --------------------------------------------------------------

def test_interpolate_value_shape_mismatch_raises(domain, recorder):
    a = hermit.constant(domain, ("DG", 0), 1.0)
    with pytest.raises(ValueError, match="value shape"):
        interpolate(a, ("DG", 0, (3,)))


def test_project_value_shape_mismatch_raises(domain, recorder):
    a = hermit.constant(domain, ("DG", 0), 1.0)
    with pytest.raises(ValueError, match="value shape"):
        project(a, ("DG", 0, (3,)))


def test_project_geometry_different_domain_raises(domain, tri_mesh, recorder):
    other = hermit.ShellDomain(tri_mesh)
    a = hermit.constant(domain, ("DG", 0), 1.0)
    geom = _FakeGeometry(other, csdl.Variable(value=other.node_coords))
    with pytest.raises(ValueError, match="different ShellDomain"):
        project(a, ("DG", 1), geometry=geom)


def test_interpolate_quadrature_source_raises(domain, recorder):
    n = domain.function_space(("Quadrature", 2)).dofmap.index_map.size_local
    qf = hermit.from_coeffs(domain, ("Quadrature", 2), np.ones(n))
    with pytest.raises(ValueError, match="project"):
        interpolate(qf, ("DG", 2))


def test_project_quadrature_source_works(domain, recorder):
    n = domain.function_space(("Quadrature", 2)).dofmap.index_map.size_local
    qf = hermit.from_coeffs(domain, ("Quadrature", 2), np.ones(n))
    out = project(qf, ("DG", 1))
    assert np.isfinite(np.asarray(out.coeffs.value)).all()


def test_project_mismatched_quadrature_degrees_raises(domain, recorder):
    n = domain.function_space(("Quadrature", 2)).dofmap.index_map.size_local
    qf = hermit.from_coeffs(domain, ("Quadrature", 2), np.ones(n))
    with pytest.raises(ValueError, match="quadrature degree"):
        project(qf, ("Quadrature", 3))
