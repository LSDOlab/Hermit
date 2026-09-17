"""``hermit.Field`` -- the input/output primitive, CSDL coefficients on a named FE
space over a ``ShellDomain``. Covers the ``eval`` / ``cell_values`` / ``to_frame`` /
``to_global`` behaviour -- a few assertions here mirror ``tests/test_fields.py``.
"""

import numpy as np
import pytest

import csdl_alpha as csdl
import dolfinx
import hermit


@pytest.fixture(scope="module")
def domain(plate_mesh):
    return hermit.ShellDomain(plate_mesh)


def _warped_domain():
    """A private mesh copy with a smooth non-affine bump (local frame not axis-aligned)
    -- same construction as ``tests/test_fields.py::_fresh_warped_mesh``."""
    import pathlib
    from mpi4py import MPI

    mesh_path = pathlib.Path(__file__).parent / "meshes" / "plate_2x10_quad_4x20.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, str(mesh_path), "r") as f:
        mesh = f.read_mesh(name="Grid")
    x = mesh.geometry.x
    u, v = x[:, 0] / 10.0, x[:, 1] / 2.0
    x[:, 2] += 0.35 * (u**2 + 0.8 * u * v)
    x[:, 0] += 0.15 * 0.35 * u * v
    return hermit.ShellDomain(mesh)


# --- construction ------------------------------------------------------------

def test_flat_vs_shaped_construction_agree(domain, recorder):
    sp = ("DG", 0, (3, 3))
    n = domain.function_space(sp).dofmap.index_map.size_local
    vals = np.arange(n * 9.0).reshape(n, 3, 3)
    f_shaped = hermit.from_coeffs(domain, sp, vals)
    f_flat = hermit.from_coeffs(domain, sp, vals.reshape(-1))
    assert np.allclose(f_shaped.coeffs.value, f_flat.coeffs.value)

    # and it matches dolfinx's own Function.x.array C-order flatten
    V = domain.function_space(sp)
    fn = dolfinx.fem.Function(V)
    fn.x.array[:] = vals.reshape(-1)
    assert np.allclose(f_shaped.coeffs.value, fn.x.array)


def test_field_wraps_numpy_and_python_scalar(domain, recorder):
    n = domain.function_space(("DG", 0)).dofmap.index_map.size_local
    f = hermit.from_coeffs(domain, ("DG", 0), list(range(n)))
    assert isinstance(f.coeffs, csdl.Variable)
    assert f.coeffs.shape == (n,)


def test_coeffs_wrong_size_raises(domain, recorder):
    with pytest.raises(ValueError, match="entries"):
        hermit.from_coeffs(domain, ("DG", 0), np.zeros(domain.n_cells - 1))


def test_bad_kind_raises(domain, recorder):
    with pytest.raises(ValueError, match="kind"):
        hermit.from_coeffs(domain, ("DG", 0), np.zeros(domain.n_cells), kind="not-a-kind")


def test_bad_space_descriptor_raises(domain):
    with pytest.raises(ValueError, match="space descriptor"):
        hermit.from_coeffs(domain, (1, 2, 3, 4), np.zeros(domain.n_cells))


# --- constant / from_function --------------------------------------------

def test_constant_scalar_and_block(domain, recorder):
    c = hermit.constant(domain, ("DG", 0), 4.0)
    assert np.allclose(c.coeffs.value, 4.0)

    block = np.array([[1.0, 2.0], [3.0, 4.0]])
    c2 = hermit.constant(domain, ("DG", 0, (2, 2)), block)
    got = np.asarray(c2.coeffs.value).reshape(domain.n_cells, 2, 2)
    assert np.allclose(got, np.broadcast_to(block, got.shape))


def test_constant_csdl_variable_keeps_derivative(domain, recorder):
    t = csdl.Variable(value=0.37, name="t")
    c = hermit.constant(domain, ("DG", 0), t)
    o = csdl.sum(c.coeffs)
    jac = csdl.experimental.PySimulator(recorder).compute_totals([o], [t])[o, t]
    # d(sum of n identical copies of t)/dt == n
    assert np.allclose(np.asarray(jac).ravel(), float(domain.n_cells))


def test_from_function_numpy_is_constant(domain, recorder):
    f = hermit.from_function(domain, ("DG", 1), lambda x: 0.1 * (1 + 0.3 * np.sin(x[:, 0])))
    assert not isinstance(f.coeffs, type(None))
    xyz = domain.dof_coords(("DG", 1))
    want = 0.1 * (1 + 0.3 * np.sin(xyz[:, 0]))
    assert np.allclose(np.asarray(f.coeffs.value), want)


def test_from_function_csdl_is_differentiable(domain, recorder):
    t = csdl.Variable(value=2.0, name="t")

    def fn(xyz):
        xv = csdl.Variable(value=xyz[:, 0])
        return t * xv

    f = hermit.from_function(domain, ("DG", 1), fn)
    o = csdl.sum(f.coeffs)
    jac = csdl.experimental.PySimulator(recorder).compute_totals([o], [t])[o, t]
    assert np.linalg.norm(np.asarray(jac)) > 0


# --- arithmetic --------------------------------------------------------------

def test_arithmetic_values(domain, recorder):
    a = hermit.constant(domain, ("DG", 0), 2.0)
    b = hermit.constant(domain, ("DG", 0), 3.0)
    assert np.allclose((a + b).coeffs.value, 5.0)
    assert np.allclose((a - b).coeffs.value, -1.0)
    assert np.allclose((2.0 * a).coeffs.value, 4.0)
    assert np.allclose((a * 2.0).coeffs.value, 4.0)
    assert np.allclose((a / b).coeffs.value, 2.0 / 3.0)
    assert np.allclose((6.0 / a).coeffs.value, 3.0)
    assert np.allclose((-a).coeffs.value, -2.0)
    assert np.allclose((5.0 - a).coeffs.value, 3.0)


def test_field_fn_matches_manual_arithmetic(domain, recorder):
    a = hermit.constant(domain, ("DG", 0), 2.0)
    b = hermit.constant(domain, ("DG", 0), 3.0)
    got = hermit.field_fn(a, b, lambda x, y: x * y + 1.0)
    assert np.allclose(got.coeffs.value, 7.0)
    got2 = hermit.field_fn(a, b, fn=lambda x, y: x * y + 1.0)
    assert np.allclose(got2.coeffs.value, 7.0)


def test_arithmetic_space_mismatch_raises(domain, recorder):
    a = hermit.constant(domain, ("DG", 0), 2.0)
    b = hermit.constant(domain, ("DG", 1), 3.0)
    with pytest.raises(ValueError, match="interpolate"):
        a + b


# --- eval / cell_values (mirrors tests/test_fields.py assertions) -----------

def test_cell_values_matches_dolfinx_eval(domain, recorder):
    """Mirrors ``test_fields.py::test_displacement_field_matches_dolfinx_eval`` --
    ``Field.cell_values()`` (file order) against a direct dolfinx ``Function.eval``
    (FE order), permuted through ``reverse_cell_idx``."""
    sp = ("DG", 1, (3,))
    V = domain.function_space(sp)
    n_scalar = V.dofmap.index_map.size_local
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(n_scalar, 3))
    f = hermit.from_coeffs(domain, sp, vals)

    fn = dolfinx.fem.Function(V)
    fn.x.array[:] = np.asarray(f.coeffs.value)
    nc = domain.n_cells
    ct = dolfinx.fem.functionspace(domain.mesh, ("DG", 0)).element.basix_element.cell_type
    import basix

    cen = basix.cell.geometry(ct).mean(axis=0)
    tree = dolfinx.geometry.bb_tree(domain.mesh, 2)
    xmid = dolfinx.mesh.compute_midpoints(domain.mesh, 2, np.arange(nc, dtype=np.int32))
    coll = dolfinx.geometry.compute_collisions_points(tree, xmid)
    fe_cells = [dolfinx.geometry.compute_colliding_cells(domain.mesh, coll, xmid).links(i)[0]
                for i in range(nc)]
    direct = fn.eval(xmid, fe_cells)              # FE cell order

    got = f.values                                 # file cell order
    want = direct[domain.reverse_cell_idx]
    assert np.allclose(got, want, atol=1e-12)


def test_eval_is_differentiable(domain, recorder):
    n = domain.function_space(("DG", 2, (3,))).dofmap.index_map.size_local
    coeffs = csdl.Variable(value=np.zeros(n * 3), name="c")
    f = hermit.from_coeffs(domain, ("DG", 2, (3,)), coeffs)
    val = f.eval([0, 10, 40], [[0.5, 0.5]])
    o = csdl.sum(val ** 2 + val)   # so the derivative isn't trivially zero at coeffs=0
    jac = csdl.experimental.PySimulator(recorder).compute_totals([o], [coeffs])[o, coeffs]
    assert np.linalg.norm(np.asarray(jac)) > 0


# --- to_frame / to_global (mirrors test_fields.py::test_strain_field_to_frame) ----

def test_to_frame_roundtrip_per_cell_dg(recorder):
    dom = _warped_domain()
    frames = dom.local_frames()   # FE order
    n = dom.n_cells
    rng = np.random.default_rng(1)
    coeffs = rng.normal(size=(n, 3))
    f = hermit.from_coeffs(dom, ("DG", 0, (3,)), coeffs, kind="strain2", frame=frames)

    g = f.to_global()
    # a manual rotation at one cell, mirroring test_fields.py's by-hand check
    k = 30
    fr = frames[dom.reverse_cell_idx][k]
    gx = np.eye(3)[0]
    gxt = gx - (gx @ fr[2]) * fr[2]
    gxt /= np.linalg.norm(gxt)
    th = np.arctan2(gxt @ fr[1], gxt @ fr[0])
    c, s = np.cos(th), np.sin(th)
    T = np.array([[c**2, s**2, s * c], [s**2, c**2, -s * c], [-2*s*c, 2*s*c, c**2 - s**2]])
    assert abs(th) > 1e-3   # the warp actually rotates the frame at this cell
    assert np.allclose(g.values[k], T @ f.values[k], rtol=1e-6, atol=1e-13)

    # exact round trip: file-order target frames bring it back to the original
    back = g.to_frame(frames[dom.reverse_cell_idx])
    assert np.allclose(back.values, f.values, rtol=1e-6, atol=1e-13)


def test_to_frame_needs_dg_space(domain, recorder):
    n = domain.function_space(("Lagrange", 1)).dofmap.index_map.size_local
    frames = np.broadcast_to(np.eye(3), (domain.n_cells, 3, 3))
    f = hermit.from_coeffs(domain, ("Lagrange", 1, (3,)), np.zeros((n, 3)), kind="strain2",
                     frame=frames)
    # a CG field with a frame set is nonsensical for to_frame (needs DG); guard it
    # explicitly since strain2 needs a (n_cells, 3, 3) frame keyed by *cell*, not dof
    with pytest.raises(ValueError, match="discontinuous"):
        f.to_frame([1.0, 0.0, 0.0])


def test_global_frame_roundtrip_and_manual(domain, recorder):
    n = domain.function_space(("Lagrange", 1)).dofmap.index_map.size_local
    rng = np.random.default_rng(2)
    coeffs = rng.normal(size=(n, 6))
    f = hermit.from_coeffs(domain, ("Lagrange", 1, (6,)), coeffs, kind="tensor3", global_frame=True)

    th = 0.7
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rot = f.to_frame(R)
    assert rot.global_frame and np.allclose(rot.frame, R)

    d = 7
    V6 = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    eng = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    old = np.asarray(f.coeffs.value).reshape(-1, 6)
    new = np.asarray(rot.coeffs.value).reshape(-1, 6)
    S = np.zeros((3, 3))
    for a, (i, j) in enumerate(V6):
        S[i, j] = S[j, i] = old[d, a] / eng[a]
    Sr = R @ S @ R.T
    want = np.array([Sr[i, j] for i, j in V6]) * eng
    assert np.allclose(new[d], want, rtol=1e-9, atol=1e-12)

    back = rot.to_global()
    assert np.allclose(np.asarray(back.coeffs.value), np.asarray(f.coeffs.value),
                       rtol=1e-9, atol=1e-12)


def test_to_frame_global_needs_full_frame(domain, recorder):
    n = domain.function_space(("Lagrange", 1)).dofmap.index_map.size_local
    f = hermit.from_coeffs(domain, ("Lagrange", 1, (6,)), np.zeros((n, 6)), kind="tensor3",
                     global_frame=True)
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        f.to_frame([1.0, 0.0, 0.0])


def test_scalar_kind_to_frame_is_noop(domain, recorder):
    f = hermit.constant(domain, ("DG", 0), 3.0)
    assert f.to_frame([1.0, 0.0, 0.0]) is f


# --- from_nodal / from_cells error paths -------------------------------------

def test_from_nodal_wrong_length_raises(domain, recorder):
    with pytest.raises(ValueError, match="n_nodes"):
        hermit.from_nodal(domain, np.zeros(domain.n_nodes - 1))


def test_from_cells_wrong_length_raises(domain, recorder):
    with pytest.raises(ValueError, match="n_cells"):
        hermit.from_cells(domain, np.zeros(domain.n_cells + 1))


# --- as_field ------------------------------------------------------------

def test_as_field_resolution(domain, recorder):
    f0 = hermit.constant(domain, ("DG", 0), 1.0)
    assert hermit.as_field(domain, f0) is f0

    scalar = hermit.as_field(domain, 3.0)
    assert scalar.space == ("Lagrange", 1, ())
    assert np.allclose(scalar.coeffs.value, 3.0)

    nodal = hermit.as_field(domain, np.ones(domain.n_nodes))
    assert nodal.space == ("Lagrange", 1, ())

    cellwise = hermit.as_field(domain, np.ones(domain.n_cells))
    assert cellwise.space == ("DG", 0, ())

    n_dg1 = domain.function_space(("DG", 1)).dofmap.index_map.size_local
    fe_order = hermit.as_field(domain, np.ones(n_dg1), space=("DG", 1))
    assert fe_order.space == ("DG", 1, ())


def test_as_field_ambiguous_raises(domain, recorder):
    n_dg1 = domain.function_space(("DG", 1)).dofmap.index_map.size_local
    bogus_len = n_dg1 + domain.n_nodes + domain.n_cells + 1
    with pytest.raises(ValueError, match="cannot resolve"):
        hermit.as_field(domain, np.zeros(bogus_len))
