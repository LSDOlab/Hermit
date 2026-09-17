"""Unit checks for the local-basis kinematics on a trivial flat quad."""

import numpy as np
import pytest


def test_local_basis_flat_xy_plate(plate_mesh):
    import dolfinx
    import ufl
    from hermit.fenics.kinematics import global_to_local_inplane, local_basis_inplane

    E0, E1, E2 = local_basis_inplane(plate_mesh)
    V = dolfinx.fem.functionspace(plate_mesh, ("DG", 0, (3,)))

    def proj(vec):
        f = dolfinx.fem.Function(V)
        from hermit.fenics.assembly import project
        project(vec, f)
        return f.x.array.reshape(-1, 3)

    e0, e1, e2 = proj(E0), proj(E1), proj(E2)
    # plate lies in z = 0: normal is +/- z, in-plane basis spans x, y
    assert np.allclose(np.abs(e2), [0, 0, 1], atol=1e-10)
    assert np.allclose(np.abs(e0[:, 2]), 0, atol=1e-10)
    # orthonormal
    assert np.allclose(np.einsum("ij,ij->i", e0, e0), 1, atol=1e-9)
    assert np.allclose(np.einsum("ij,ij->i", e0, e1), 0, atol=1e-9)
    assert np.allclose(np.einsum("ij,ij->i", e0, e2), 0, atol=1e-9)

    T = global_to_local_inplane(E0, E1)
    VT = dolfinx.fem.functionspace(plate_mesh, ("DG", 0, (2, 3)))
    ft = dolfinx.fem.Function(VT)
    from hermit.fenics.assembly import project
    project(T, ft)
    Tm = ft.x.array.reshape(-1, 2, 3)
    assert np.allclose(np.einsum("nik,njk->nij", Tm, Tm), np.eye(2), atol=1e-9)
