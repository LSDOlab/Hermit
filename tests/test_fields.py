"""Strain / displacement field outputs: ``hm.strain_fields`` /
``hm.displacement_field`` and the ``Field`` orientation machinery behind them.

``tests/test_field_api.py`` covers the ``Field`` *primitive* (construction,
arithmetic, ``eval``, ``to_frame`` on synthetic coefficients). What is unique here:
the strain-field shape derivatives, interpolate-vs-project, CG/global-frame handling,
thickness gradients, and tri-mesh frame consistency -- all driven off a real solved
state.

**Coefficient ordering.** ``Field.coeffs`` is FE dof order (of the *canonical*,
freshly built space for that descriptor) and ``Field.values`` / ``Field.eval`` are
**file** cell order; ``domain.local_frames()`` is FE cell order, so a per-cell frame
handed to ``to_frame`` is permuted to file order first
(``local_frames()[domain.reverse_cell_idx]``). ``test_displacement_field_matches_
dolfinx_eval`` is the explicit guard on the displacement path, which reads a
collapsed sub-space of ``W`` (a different dof order from a freshly built space of the
same element).
"""

import pathlib

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit
import hermit.bcs as hbc
import hermit.loads as hld
import hermit.material as hmat
import hermit.outputs as out
from hermit.domain import ShellDomain
from hermit._solve import solve, _pde_for
from conftest import clamped_at_x0

MESH = pathlib.Path(__file__).parent / "meshes" / "plate_2x10_quad_4x20.xdmf"


def _fresh_warped_mesh(amp=0.35):
    """A private mesh copy with a smooth non-affine bump (local frame not axis-aligned)."""
    import dolfinx
    from mpi4py import MPI

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, str(MESH), "r") as f:
        mesh = f.read_mesh(name="Grid")
    x = mesh.geometry.x
    u, v = x[:, 0] / 10.0, x[:, 1] / 2.0
    x[:, 2] += amp * (u**2 + 0.8 * u * v)          # bend + twist -> in-plane frame rotates
    x[:, 0] += 0.15 * amp * u * v
    return mesh


def _solve(mesh, ref, pz=2.0, thickness=None, geometry=None, domain=None):
    """Solved state for the cantilever fixture (CG1 material fields)."""
    E, nu, h, rho = (float(ref[k]) for k in ("E_val", "nu_val", "h_val", "rho_val"))
    domain = ShellDomain(mesh, element="CG2CG1") if domain is None else domain
    nn = domain.n_nodes
    material = hmat.isotropic(
        domain, E=E * np.ones(nn), nu=nu * np.ones(nn),
        thickness=h * np.ones(nn) if thickness is None else thickness,
        density=rho * np.ones(nn), constitutive_space=("Lagrange", 1))
    state = solve(domain, material, hld.pressure(domain, pz),
                  hbc.clamp(domain, where=clamped_at_x0), geometry=geometry)
    return domain, state


def test_displacement_field_matches_dolfinx_eval(plate_mesh, cantilever_ref, recorder):
    import dolfinx

    domain, state = _solve(plate_mesh, cantilever_ref)
    d = out.displacement_field(state)
    assert isinstance(d, hermit.Field)

    # reconstruct the same FE function and eval it at cell centroids directly. This
    # goes through the *collapsed* sub-space dof order, which is not the order of the
    # freshly built space Field tabulates against -- the point of the check.
    V, dofs = domain.W.sub(0).collapse()
    # the collapsed order really is a different dof order from the canonical
    # (freshly built) space ``Field`` tabulates against -- 349 of 369 CG2 blocks move
    # on this mesh -- so this comparison does exercise the permutation, not a no-op
    assert not np.allclose(V.tabulate_dof_coordinates(),
                           domain.dof_coords(("Lagrange", 2, (3,))))
    fn = dolfinx.fem.Function(V)
    fn.x.array[:] = np.asarray(state.disp_solid.value)[np.asarray(dofs).reshape(-1)]
    nc = plate_mesh.topology.index_map(2).size_local
    tree = dolfinx.geometry.bb_tree(plate_mesh, 2)
    xmid = dolfinx.mesh.compute_midpoints(plate_mesh, 2, np.arange(nc, dtype=np.int32))
    coll = dolfinx.geometry.compute_collisions_points(tree, xmid)
    fe_cells = [dolfinx.geometry.compute_colliding_cells(plate_mesh, coll, xmid).links(i)[0]
                for i in range(nc)]
    direct = fn.eval(xmid, fe_cells)              # FE cell order

    got = d.values                               # user (file) cell order
    want = direct[domain.reverse_cell_idx]
    assert np.allclose(got, want, atol=1e-12)


def test_strain_field_to_frame(cantilever_ref, recorder):
    mesh = _fresh_warped_mesh()
    domain, state = _solve(mesh, cantilever_ref, pz=5.0)
    fields = dict(zip(("strain", "curvature", "shear_strain"), out.strain_fields(state)))
    frames_user = domain.local_frames()[domain.reverse_cell_idx]

    # rotate a chosen cell's tensor by hand and compare to to_global()
    k = 30
    fr = frames_user[k]
    gx = np.eye(3)[0]
    gxt = gx - (gx @ fr[2]) * fr[2]; gxt /= np.linalg.norm(gxt)
    th = np.arctan2(gxt @ fr[1], gxt @ fr[0])
    c, s = np.cos(th), np.sin(th)
    T_strain = np.array([[c**2, s**2, s*c], [s**2, c**2, -s*c], [-2*s*c, 2*s*c, c**2-s**2]])
    T_shear = np.array([[c, s], [-s, c]])
    assert abs(th) > 1e-3   # the warp actually rotates the in-plane frame at this cell

    for name, T in (("strain", T_strain), ("curvature", T_strain), ("shear_strain", T_shear)):
        f = fields[name]
        g = f.to_global().values
        assert np.allclose(g[k], T @ f.values[k], rtol=1e-6, atol=1e-14), name
        # exact round trip back to the local frame
        assert np.allclose(f.to_global().to_frame(frames_user).values, f.values,
                           rtol=1e-6, atol=1e-14), name


def test_strain_field_point_eval_is_differentiable(plate_mesh, cantilever_ref, recorder):
    nn = plate_mesh.geometry.x.shape[0]
    h = float(cantilever_ref["h_val"])

    rec = csdl.Recorder(inline=True); rec.start()
    t = csdl.Variable(value=h * np.ones(nn), name="t")
    _, state = _solve(plate_mesh, cantilever_ref, thickness=t)
    curvature = out.strain_fields(state)[1]
    val = curvature.eval([0, 10, 40], [[0.5, 0.5]])       # 3 cells, centroid
    o = csdl.sum(val ** 2)
    jac = csdl.experimental.PySimulator(rec).compute_totals([o], [t])[o, t]
    rec.stop()
    assert np.asarray(jac).shape == (1, nn)
    assert np.linalg.norm(np.asarray(jac)) > 0


# --- target space + method -------------------------------------------------

def _strain_state(mesh, ref, **kwargs):
    """Strain fixture: same material, pressure 5e3. The strain space / method / frame
    are postprocess arguments."""
    return _solve(mesh, ref, pz=5e3, **kwargs)


def test_strain_interpolate_matches_dolfinx(plate_mesh, cantilever_ref, recorder):
    """method='interpolate' == a direct dolfinx Expression interpolation of voigt2D(kappa)."""
    import dolfinx
    from hermit.fenics.kinematics import voigt2D
    import hermit.fenics.assembly as fa

    domain, state = _strain_state(plate_mesh, cantilever_ref)
    curvature = out.strain_fields(state, space=("DG", 2), method="interpolate")[1]

    pde = _pde_for(domain)
    em = pde.elastic_model()
    pde.set_geometry(np.asarray(state.geometry.nodes.value))
    fa.set_array(pde.w, np.asarray(state.disp_solid.value))
    V = dolfinx.fem.functionspace(plate_mesh, ("DG", 2, (3,)))
    fh = dolfinx.fem.Function(V)
    fh.interpolate(dolfinx.fem.Expression(voigt2D(em.kappa),
                                          hermit._compat.interpolation_points(V)))
    pde.restore_geometry()

    got = np.asarray(curvature.coeffs.value).reshape(-1, 3)
    want = fh.x.array.reshape(-1, 3)
    assert np.abs(want).max() > 1e-3
    assert np.allclose(got, want, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("space", [("DG", 1), ("DG", 2), ("DG", 3)])
def test_strain_space_override(plate_mesh, cantilever_ref, recorder, space):
    _, state = _strain_state(plate_mesh, cantilever_ref)
    curvature = out.strain_fields(state, space=space, method="interpolate")[1]
    assert curvature.V.ufl_element().degree == space[1]
    assert curvature.n_scalar_dofs == 80 * (space[1] + 1) ** 2   # DGk on 80 quads
    v = curvature.eval(list(range(0, 80, 7)), [[0.5, 0.5]])   # bending field, nonzero mid-span
    assert np.isfinite(np.asarray(v.value)).all() and np.abs(np.asarray(v.value)).max() > 1e-4


def test_interpolate_has_no_shape_derivative(plate_mesh, cantilever_ref, recorder):
    nn = plate_mesh.geometry.x.shape[0]
    domain = ShellDomain(plate_mesh, element="CG2CG1")
    nd = csdl.Variable(value=np.zeros((nn, 3)), name="nd")
    _, state = _strain_state(plate_mesh, cantilever_ref, domain=domain,
                             geometry=hermit.geometry(domain, node_disp=nd))
    with pytest.raises(NotImplementedError, match="project"):
        out.strain_fields(state, space=("DG", 2), method="interpolate")


@pytest.mark.parametrize("space", [("DG", 2), ("Lagrange", 1), ("Lagrange", 2)])
def test_interpolate_derivative_vs_fd(plate_mesh, cantilever_ref, space):
    """d(sum curvature**2)/d(thickness) through method='interpolate' vs central diff.

    Parametrised over a CG target as well as DG: the interpolation Jacobian scatters
    one point evaluation per cell into the target dofmap, and a shared (continuous)
    dof is written once per adjacent cell. Summing those duplicates -- which
    ``coo_matrix(...).tocsr()`` does -- scaled every shared row by the dof's
    multiplicity, ~3x on this mesh. Invisible on DG (no shared dofs), which is why
    only DG2 was covered before. See ``hermit.fenics.assembly.coo_dedup_last``."""
    nn = plate_mesh.geometry.x.shape[0]
    h = float(cantilever_ref["h_val"])
    V = np.linspace(0.5, 1.5, nn)

    def run(scale, grad):
        rec = csdl.Recorder(inline=True); rec.start()
        t = csdl.Variable(value=h * scale * V, name="t")
        _, state = _strain_state(plate_mesh, cantilever_ref, thickness=t)
        curvature = out.strain_fields(state, space=space, method="interpolate")[1]
        o = csdl.sum(curvature.coeffs ** 2)
        val = float(np.ravel(o.value)[0])
        g = np.asarray(csdl.experimental.PySimulator(rec).compute_totals([o], [t])[o, t]).ravel() if grad else None
        rec.stop()
        return val, g

    _, g = run(1.0, True)
    d = 1e-4
    vp, _ = run(1.0 + d, False)
    vm, _ = run(1.0 - d, False)
    dd_fd = (vp - vm) / (2 * d * h)
    dd_an = float(g @ V)
    rel = abs(dd_an - dd_fd) / abs(dd_fd)
    print(f"\nd(sum k^2)/dt.V  analytic={dd_an:+.6e}  fd={dd_fd:+.6e}  rel={rel:.2e}")
    assert rel < 1e-4


def test_project_dg0_equals_average(plate_mesh, cantilever_ref, recorder):
    """L2 projection onto ('DG', 0) is the cell average -- identical to method='average'."""
    _, state = _strain_state(plate_mesh, cantilever_ref)
    projected = out.strain_fields(state, space=("DG", 0), method="project")[1]
    averaged = out.strain_fields(state, space=("DG", 0), method="average")[1]
    assert np.allclose(np.asarray(projected.coeffs.value),
                       np.asarray(averaged.coeffs.value), rtol=1e-9, atol=1e-12)


def test_project_strain_shape_derivative(plate_mesh, cantilever_ref):
    """d(sum strain**2)/d(node_disp) via method='project' vs central difference."""
    nn = plate_mesh.geometry.x.shape[0]
    gx = plate_mesh.geometry.x.copy()
    L = 10.0
    bend = np.zeros((nn, 3)); bend[:, 2] = 0.3 * (gx[:, 0] / L) ** 2; bend[:, 0] = 0.02 * gx[:, 0] / L
    V = np.zeros((nn, 3))
    V[:, 2] = np.sin(np.pi * gx[:, 0] / L) * (0.5 + gx[:, 1] / 2.0); V[:, 0] = 0.3 * gx[:, 0] / L
    V /= np.linalg.norm(V)

    def run(nd_val, grad):
        assert np.allclose(plate_mesh.geometry.x, gx)
        rec = csdl.Recorder(inline=True); rec.start()
        domain = ShellDomain(plate_mesh, element="CG2CG1")
        nd = csdl.Variable(value=nd_val.copy(), name="nd")
        _, state = _strain_state(plate_mesh, cantilever_ref, domain=domain,
                                 geometry=hermit.geometry(domain, node_disp=nd))
        curvature = out.strain_fields(state, space=("DG", 1), method="project")[1]
        o = csdl.sum(curvature.coeffs ** 2)
        val = float(np.ravel(o.value)[0])
        g = None
        if grad:
            g = np.asarray(csdl.experimental.PySimulator(rec).compute_totals([o], [nd])[o, nd]).reshape(nn, 3)
        rec.stop()
        return val, g

    _, g = run(bend, True)
    dd_an = float((g * V).sum())
    step = 1e-3
    vp, _ = run(bend + step * V, False)
    vm, _ = run(bend - step * V, False)
    dd_fd = (vp - vm) / (2 * step)
    rel = abs(dd_an - dd_fd) / abs(dd_fd)
    print(f"\nd(sum k^2)/d(nd).V  analytic={dd_an:+.6e}  fd={dd_fd:+.6e}  rel={rel:.2e}")
    assert rel < 5e-4


def test_to_frame_roundtrip_dg2(cantilever_ref, recorder):
    """to_frame on a genuinely multi-dof-per-cell (DG2) oriented field, warped mesh."""
    mesh = _fresh_warped_mesh()
    domain, state = _strain_state(mesh, cantilever_ref)
    membrane, _, shear = out.strain_fields(state, space=("DG", 2), method="interpolate")
    frames_user = domain.local_frames()[domain.reverse_cell_idx]
    for name, f in (("strain", membrane), ("shear_strain", shear)):
        back = f.to_global().to_frame(frames_user)
        assert np.allclose(np.asarray(back.coeffs.value), np.asarray(f.coeffs.value),
                           rtol=1e-8, atol=1e-12), name


# --- CG (continuous) strain fields: global Cartesian frame -----------------

def _manual_global_project(state, expr_name, space):
    """A hand-rolled global-frame L2 projection of a strain expression -- the
    reference for method='project' onto a CG space."""
    import dolfinx
    import ufl
    import hermit.fenics.assembly as fa
    from hermit.fenics.kinematics import (voigt3D, vec2D_local_to_global,
                                          strain2D_local_to_global as l2g)

    pde = _pde_for(state.domain)
    em = pde.elastic_model()
    T = em.E01
    glob = {"strain": voigt3D(l2g(em.eps, T)), "curvature": voigt3D(l2g(em.kappa, T)),
            "shear_strain": vec2D_local_to_global(em.gamma, T)}[expr_name]
    nc = 3 if expr_name == "shear_strain" else 6
    pde.set_geometry(np.asarray(state.geometry.nodes.value))
    fa.set_array(pde.w, np.asarray(state.disp_solid.value))
    V = dolfinx.fem.functionspace(state.domain.mesh, (*space, (nc,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    M = fa.assemble_matrix(ufl.inner(u, v) * ufl.dx)
    b = fa.assemble_vector(ufl.inner(glob, v) * ufl.dx)
    x = M.createVecRight(); rhs = M.createVecRight()
    rhs.setArray(b); rhs.assemble()
    fa.ksp_mumps(M).solve(rhs, x)
    pde.restore_geometry()
    return x.getArray().copy().reshape(-1, nc)


@pytest.mark.parametrize("space", [("Lagrange", 1), ("Lagrange", 2)])
def test_cg_strain_is_global_and_matches_manual_projection(plate_mesh, cantilever_ref,
                                                           recorder, space):
    _, state = _strain_state(plate_mesh, cantilever_ref)
    membrane, curvature, shear = out.strain_fields(state, space=space, method="project")

    for name, f, kind, ncomp in (("strain", membrane, "tensor3", 6),
                                 ("curvature", curvature, "tensor3", 6),
                                 ("shear_strain", shear, "vector3", 3)):
        assert f.global_frame and f.kind == kind and not f.is_dg
        got = np.asarray(f.coeffs.value).reshape(-1, ncomp)
        want = _manual_global_project(state, name, space)
        assert np.allclose(got, want, rtol=1e-8, atol=1e-10), name
    # curvature / transverse shear are the non-trivial fields for a bending plate
    assert np.abs(np.asarray(curvature.coeffs.value)).max() > 1e-3
    assert np.abs(np.asarray(shear.coeffs.value)).max() > 1e-4


def test_cg_strain_thickness_gradient(plate_mesh, cantilever_ref, recorder):
    nn = plate_mesh.geometry.x.shape[0]
    h = float(cantilever_ref["h_val"])
    t = csdl.Variable(value=h * np.ones(nn), name="t")
    _, state = _strain_state(plate_mesh, cantilever_ref, thickness=t)
    curvature = out.strain_fields(state, space=("Lagrange", 1), method="project")[1]
    o = csdl.sum(curvature.coeffs ** 2)
    jac = np.asarray(csdl.experimental.PySimulator(recorder).compute_totals([o], [t])[o, t])
    assert jac.shape == (1, nn) and np.linalg.norm(jac) > 0


def test_cg_strain_shape_derivative(plate_mesh, cantilever_ref):
    """d(sum curvature**2)/d(node_disp) for a CG1 global-frame projected field vs FD."""
    nn = plate_mesh.geometry.x.shape[0]
    gx = plate_mesh.geometry.x.copy()
    L = 10.0
    bend = np.zeros((nn, 3)); bend[:, 2] = 0.3 * (gx[:, 0] / L) ** 2; bend[:, 0] = 0.02 * gx[:, 0] / L
    V = np.zeros((nn, 3))
    V[:, 2] = np.sin(np.pi * gx[:, 0] / L) * (0.5 + gx[:, 1] / 2.0); V[:, 0] = 0.3 * gx[:, 0] / L
    V /= np.linalg.norm(V)

    def run(nd_val, grad):
        rec = csdl.Recorder(inline=True); rec.start()
        domain = ShellDomain(plate_mesh, element="CG2CG1")
        nd = csdl.Variable(value=nd_val.copy(), name="nd")
        _, state = _strain_state(plate_mesh, cantilever_ref, domain=domain,
                                 geometry=hermit.geometry(domain, node_disp=nd))
        curvature = out.strain_fields(state, space=("Lagrange", 1), method="project")[1]
        o = csdl.sum(curvature.coeffs ** 2)
        val = float(np.ravel(o.value)[0])
        g = (np.asarray(csdl.experimental.PySimulator(rec).compute_totals([o], [nd])[o, nd]).reshape(nn, 3)
             if grad else None)
        rec.stop()
        return val, g

    _, g = run(bend, True)
    dd_an = float((g * V).sum())
    step = 1e-3
    vp, _ = run(bend + step * V, False)
    vm, _ = run(bend - step * V, False)
    dd_fd = (vp - vm) / (2 * step)
    rel = abs(dd_an - dd_fd) / abs(dd_fd)
    print(f"\nCG1 global project d(sum k^2)/d(nd).V  analytic={dd_an:+.6e}  fd={dd_fd:+.6e}  rel={rel:.2e}")
    assert rel < 5e-4


def test_cg_strain_frame_consistency_on_tri_mesh(tri_mesh, cantilever_ref, recorder):
    """On a triangle mesh the per-cell in-plane frame varies cell to cell; the global
    Cartesian projection is still well posed and matches a manual projection."""
    _, state = _strain_state(tri_mesh, cantilever_ref)
    curvature = out.strain_fields(state, space=("Lagrange", 1), method="project")[1]
    got = np.asarray(curvature.coeffs.value).reshape(-1, 6)
    want = _manual_global_project(state, "curvature", ("Lagrange", 1))
    assert np.abs(want).max() > 1e-4
    assert np.allclose(got, want, rtol=1e-8, atol=1e-10)


def test_cg_to_frame_global_roundtrip_and_manual(plate_mesh, cantilever_ref, recorder):
    from hermit._field import _global_component_transform

    _, state = _strain_state(plate_mesh, cantilever_ref)
    _, curvature, shear = out.strain_fields(state, space=("Lagrange", 1), method="project")
    th = 0.7
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])   # a proper rotation about z

    V6 = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    eng = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    for name, f, kind in (("curvature", curvature, "tensor3"),
                          ("shear_strain", shear, "vector3")):
        rot = f.to_frame(R)
        assert rot.global_frame and np.allclose(rot.frame, R)
        M = _global_component_transform(kind, R)          # new = M @ old, per dof
        nc = M.shape[0]
        old = np.asarray(f.coeffs.value).reshape(-1, nc)
        new = np.asarray(rot.coeffs.value).reshape(-1, nc)
        assert np.allclose(new, old @ M.T, rtol=1e-9, atol=1e-12), name
        # against a from-scratch tensor / vector rotation at one dof
        d = 7
        if kind == "vector3":
            want = R @ old[d]
        else:
            S = np.zeros((3, 3))
            for a, (i, j) in enumerate(V6):
                S[i, j] = S[j, i] = old[d, a] / eng[a]
            Sr = R @ S @ R.T
            want = np.array([Sr[i, j] for i, j in V6]) * eng
        assert np.allclose(new[d], want, rtol=1e-9, atol=1e-12), name
        # exact round trip
        back = rot.to_global()
        assert np.allclose(np.asarray(back.coeffs.value).reshape(-1, nc), old,
                           rtol=1e-9, atol=1e-12), name


def test_cg_interpolate_is_also_global(plate_mesh, cantilever_ref, recorder):
    """method='interpolate' onto a CG space also yields a global-frame field."""
    _, state = _strain_state(plate_mesh, cantilever_ref)
    k = out.strain_fields(state, space=("Lagrange", 1), method="interpolate")[1]
    assert k.global_frame and k.kind == "tensor3"
    assert np.abs(np.asarray(k.coeffs.value)).max() > 1e-3
    assert np.allclose(np.asarray(k.to_global().coeffs.value), np.asarray(k.coeffs.value))


def test_cg_to_frame_needs_full_frame(plate_mesh, cantilever_ref, recorder):
    _, state = _strain_state(plate_mesh, cantilever_ref)
    membrane = out.strain_fields(state, space=("Lagrange", 1), method="project")[0]
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        membrane.to_frame([1.0, 0.0, 0.0])


def test_strain_field_frame_local_on_cg_raises(plate_mesh, cantilever_ref, recorder):
    _, state = _strain_state(plate_mesh, cantilever_ref)
    with pytest.raises(ValueError, match="global"):
        out.strain_fields(state, space=("Lagrange", 1), method="project", frame="local")


def test_dg_strain_defaults_stay_local(plate_mesh, cantilever_ref, recorder):
    """Regression: the DG default is unchanged -- local in-plane frame, 3/2 components."""
    _, state = _strain_state(plate_mesh, cantilever_ref)
    membrane, _, shear = out.strain_fields(state, space=("DG", 2), method="project")
    assert not membrane.global_frame and membrane.kind == "strain2"
    assert np.asarray(membrane.coeffs.value).reshape(-1, 3).shape[1] == 3
    assert shear.kind == "shear2"
