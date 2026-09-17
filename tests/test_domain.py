"""``ShellDomain`` -- mesh + shell state space + index maps. No solve involved;
everything here is setup-time, no CSDL.
"""

import numpy as np
import pytest

import hermit


def test_construction_quad_and_tri(plate_mesh, tri_mesh):
    dq = hermit.ShellDomain(plate_mesh)
    dt = hermit.ShellDomain(tri_mesh)
    assert dq.n_nodes == plate_mesh.geometry.x.shape[0]
    assert dq.n_cells == plate_mesh.topology.index_map(2).size_local
    assert dt.n_nodes == tri_mesh.geometry.x.shape[0]
    assert dt.n_cells == tri_mesh.topology.index_map(2).size_local
    assert dq.n_cells != dt.n_cells   # quad vs tri subdivision of the same plate


def test_node_coords_file_order_roundtrip(plate_mesh):
    dom = hermit.ShellDomain(plate_mesh)
    assert dom.node_coords.shape == (dom.n_nodes, 3)
    # file order -> local order via node_input_idx must reproduce mesh.geometry.x
    assert np.allclose(dom.node_coords[dom.node_input_idx], plate_mesh.geometry.x)
    # and the reverse map is its inverse
    assert np.array_equal(dom.node_input_idx[dom.reverse_node_idx], np.arange(dom.n_nodes))


def test_cell_centroids_file_order_roundtrip(plate_mesh):
    import dolfinx

    dom = hermit.ShellDomain(plate_mesh)
    assert dom.cell_centroids.shape == (dom.n_cells, 3)
    local = dolfinx.mesh.compute_midpoints(
        plate_mesh, 2, np.arange(dom.n_cells, dtype=np.int32))
    assert np.allclose(dom.cell_centroids[dom.cell_input_idx], local)
    assert np.array_equal(dom.cell_input_idx[dom.reverse_cell_idx], np.arange(dom.n_cells))


@pytest.mark.parametrize("space,expected_n", [
    (("Lagrange", 1), 105),
    (("Lagrange", 2), None),
    (("DG", 0), 80),
    (("DG", 1), None),
])
def test_dof_coords_shape(plate_mesh, space, expected_n):
    dom = hermit.ShellDomain(plate_mesh)
    coords = dom.dof_coords(space)
    assert coords.shape[1] == 3
    if expected_n is not None:
        assert coords.shape[0] == expected_n
    assert coords.shape[0] == dom.function_space(space).dofmap.index_map.size_local


def test_dof_coords_vector_space_is_per_scalar_dof(plate_mesh):
    dom = hermit.ShellDomain(plate_mesh)
    coords = dom.dof_coords(("Lagrange", 1, (3,)))
    # per scalar dof, not per component -- same count as the scalar CG1 space
    assert coords.shape == (105, 3)
    assert np.allclose(coords, dom.dof_coords(("Lagrange", 1)))


def test_quadrature_space_constructs_and_round_trips(plate_mesh, recorder):
    dom = hermit.ShellDomain(plate_mesh)
    V = dom.function_space(("Quadrature", 2))
    n = V.dofmap.index_map.size_local
    assert n > dom.n_cells   # multiple quadrature points per cell

    coords = dom.dof_coords(("Quadrature", 2))
    assert coords.shape == (n, 3)

    coeffs = np.arange(n, dtype=float)
    fld = hermit.from_coeffs(dom, ("Quadrature", 2), coeffs)
    assert np.allclose(np.asarray(fld.coeffs.value), coeffs)


def _reference_local_frames(mesh):
    """An independent evaluation of the local element frames, as an oracle for
    ``ShellDomain.local_frames`` (``basix.cell.geometry(...).mean`` for the reference
    point rather than ``domain._cell_centroid_ref_point``)."""
    import basix
    import dolfinx

    from hermit.fenics.kinematics import local_basis_inplane

    E0, E1, E2 = local_basis_inplane(mesh)
    tdim = mesh.topology.dim
    nc = mesh.topology.index_map(tdim).size_local
    cells = np.arange(nc, dtype=np.int32)
    ct = dolfinx.fem.functionspace(mesh, ("DG", 0)).element.basix_element.cell_type
    pt = basix.cell.geometry(ct).mean(axis=0)[None, :]
    frames = np.zeros((nc, 3, 3))
    for i, Ei in enumerate((E0, E1, E2)):
        frames[:, i, :] = dolfinx.fem.Expression(Ei, pt).eval(mesh, cells).reshape(nc, 3)
    return frames


def test_local_frames_orthonormal_and_matches_reference(plate_mesh):
    dom = hermit.ShellDomain(plate_mesh)
    frames = dom.local_frames()
    assert frames.shape == (dom.n_cells, 3, 3)

    # orthonormal rows
    gram = np.einsum("nij,nkj->nik", frames, frames)
    eye = np.broadcast_to(np.eye(3), gram.shape)
    assert np.allclose(gram, eye, atol=1e-10)

    assert np.allclose(frames, _reference_local_frames(plate_mesh))


def test_local_frames_at_reproduces_local_frames(plate_mesh):
    dom = hermit.ShellDomain(plate_mesh)
    frames_fe = dom.local_frames()
    # centroid parametric point on a quad
    pt = np.array([[0.5, 0.5]])
    all_file_cells = np.arange(dom.n_cells)
    frames_via_at = dom.local_frames_at(pt, all_file_cells)
    # local_frames_at takes FILE-order cells; querying every cell in file order and
    # permuting local_frames() (FE order) into file order must agree
    assert np.allclose(frames_via_at, frames_fe[dom.reverse_cell_idx])


def test_local_frames_at_tri_mesh(tri_mesh):
    dom = hermit.ShellDomain(tri_mesh)
    frames_fe = dom.local_frames()
    pt = np.array([[1.0 / 3.0, 1.0 / 3.0]])
    frames_via_at = dom.local_frames_at(pt, np.arange(dom.n_cells))
    assert np.allclose(frames_via_at, frames_fe[dom.reverse_cell_idx])


def test_regions_and_cell_tags_stored(plate_mesh):
    dom = hermit.ShellDomain(plate_mesh)
    assert dom.regions == {}
    assert dom.cell_tags is None

    regions = {"root": 1, "tip": 2}
    dom2 = hermit.ShellDomain(plate_mesh, regions=regions, cell_tags="sentinel")
    assert dom2.regions == regions
    assert dom2.regions is not regions   # copied, not aliased
    assert dom2.cell_tags == "sentinel"


def test_function_space_is_memoized(plate_mesh):
    dom = hermit.ShellDomain(plate_mesh)
    V1 = dom.function_space(("Lagrange", 1))
    V2 = dom.function_space(("CG", 1))   # normalises to the same descriptor
    assert V1 is V2


def test_unsupported_element_raises(plate_mesh):
    with pytest.raises(ValueError, match="unsupported element"):
        hermit.ShellDomain(plate_mesh, element="bogus")


def test_cr_element_needs_simplex_mesh(plate_mesh):
    with pytest.raises(ValueError, match="simplex"):
        hermit.ShellDomain(plate_mesh, element="CG2CR1")


# --- the three-ordering translation test ------------------------------------

def test_from_nodal_from_cells_land_on_the_right_physical_entity(plate_mesh, recorder):
    """A per-vertex / per-cell ramp built in FILE order must land on the right
    physical vertex/cell once read back through the FE-ordered Field -- this would
    fail silently if the file<->FE permutation were dropped or inverted."""
    dom = hermit.ShellDomain(plate_mesh)

    # per-vertex ramp: value = x-coordinate, given in FILE order
    node_ramp_file = dom.node_coords[:, 0].copy()
    nodal_field = hermit.from_nodal(dom, node_ramp_file)
    assert nodal_field.space == ("Lagrange", 1, ())
    # read back at CG1 dof coordinates (FE order) -- must equal that dof's own x
    dof_xyz = dom.dof_coords(("Lagrange", 1))
    assert np.allclose(np.asarray(nodal_field.coeffs.value), dof_xyz[:, 0], atol=1e-10)

    # per-cell ramp: value = centroid x-coordinate, given in FILE order
    cell_ramp_file = dom.cell_centroids[:, 0].copy()
    cell_field = hermit.from_cells(dom, cell_ramp_file)
    assert cell_field.space == ("DG", 0, ())
    dg0_xyz = dom.dof_coords(("DG", 0))
    assert np.allclose(np.asarray(cell_field.coeffs.value), dg0_xyz[:, 0], atol=1e-10)

    # end-to-end: cell_values() (file order) must reproduce the original file-order
    # ramp exactly (round trip through the FE permutation and back)
    got = np.asarray(cell_field.cell_values().value).reshape(-1)
    assert np.allclose(got, cell_ramp_file, atol=1e-10)


def test_field_fe_order_path_matches_from_nodal(plate_mesh, recorder):
    """``hm.from_coeffs`` built directly from ``domain.dof_coords`` (the FE-order path) must
    agree with ``from_nodal`` of the same physical ramp (the file-order path)."""
    dom = hermit.ShellDomain(plate_mesh)
    xy = dom.dof_coords(("Lagrange", 1))
    f_direct = hermit.from_coeffs(dom, ("Lagrange", 1), xy[:, 0])

    node_ramp_file = dom.node_coords[:, 0]
    f_from_nodal = hermit.from_nodal(dom, node_ramp_file)

    assert np.allclose(np.asarray(f_direct.coeffs.value), np.asarray(f_from_nodal.coeffs.value))


def test_from_cells_matches_dolfinx_function_on_same_space(plate_mesh, recorder):
    """A ``from_cells`` field, read back as a real ``dolfinx.Function``, has the DG0
    dof at FE-cell ``c`` holding the physical value of file-cell
    ``cell_input_idx[c]``."""
    import dolfinx

    dom = hermit.ShellDomain(plate_mesh)
    values_file = np.arange(dom.n_cells, dtype=float) * 2.0 + 1.0
    fld = hermit.from_cells(dom, values_file)

    V = dom.function_space(("DG", 0))
    fn = dolfinx.fem.Function(V)
    fn.x.array[:] = np.asarray(fld.coeffs.value)

    # FE-order dof c should hold values_file[cell_input_idx[c]]
    assert np.allclose(fn.x.array, values_file[dom.cell_input_idx])
