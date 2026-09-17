"""``ShellDomain`` -- mesh, shell state space and index maps, built once.

No CSDL, no per-evaluation cost. Carries the mixed state space ``W``, mesh tags and
regions, and the file/FE index maps the ``Field`` builders need. Deliberately does
not depend on ``ShellPDE``: the dependency runs the other way, with
``hermit._solve`` handing ``domain.W`` to the ``ShellPDE`` it caches on the domain.
"""

from __future__ import annotations

import numpy as np

from .fenics.spaces import make_space, normalize_space, state_space


class ShellDomain:
    """Mesh, shell state space, index maps and region tags, built once.

    The first object in every Hermit workflow. It carries no CSDL and no per-solve
    cost, so build it once and reuse it across every load case, sweep or optimizer
    iteration.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        A **serial**, straight-sided (P1 geometry) surface mesh of triangles or
        quadrilaterals.
    element : {'CG2CG1', 'CG1CG1', 'CG2CR1'}, optional
        Displacement x rotation element pair. Default ``'CG2CG1'``. ``'CG2CR1'``
        uses a Crouzeix-Raviart rotation that resists shear locking on thin shells,
        and is **triangle meshes only**.
    cell_tags : dolfinx.mesh.MeshTags, optional
        Cell tags, for per-region output integrals.
    regions : dict, optional
        Maps region names to tag values, so outputs can take ``region="skin"``.
    quadrature_degree : int, optional
        Degree for every integral this package pins by hand: the penalty-BC facet
        measures, the stress measures, the DG0 ``'average'`` field method and the
        oriented elastic energy. Default 4, which is deep in the converged regime
        for the shell integrand.

    Attributes
    ----------
    W : dolfinx.fem.FunctionSpace
        **The** mixed state space for this domain, shared by everything downstream.
    n_nodes, n_cells : int
    node_coords : ndarray
        ``(n_nodes, 3)`` vertex coordinates in **file** order.
    cell_centroids : ndarray
        ``(n_cells, 3)`` centroids in **file** order.
    node_input_idx, cell_input_idx : ndarray
        File-to-FE gather indices, ``fe_order = file_order[node_input_idx]``.
    reverse_node_idx, reverse_cell_idx : ndarray
        Their inverses, ``file_order = fe_order[reverse_node_idx]``.

    Raises
    ------
    ValueError
        If the mesh is not serial (its ``input_global_indices`` must form a node
        permutation), or if CG1 / DG0 dof order does not coincide with the mesh's
        own vertex / cell order.

    Notes
    -----
    **Serial meshes only.** This is a requirement of every use, not only of the
    shape-derivative path: an MPI-partitioned mesh raises here, before any solve.

    **Three coefficient orderings** coexist on any mesh: *file* (the mesh-file
    vertex and cell numbering, which external arrays use), *local* (DOLFINx's
    post-read reordering) and *FE dof* (per function space, indexing
    ``Function.x.array``). :func:`~hermit.from_nodal` and
    :func:`~hermit.from_cells` take file order and permute; :func:`~hermit.from_coeffs`
    takes FE dof order directly.

    **One state space per domain.** DOLFINx does not treat two independently built
    but structurally identical function spaces as interchangeable for boundary
    conditions: a strong BC located against a second instance is silently dropped,
    with no exception and a wrong answer. Everything downstream therefore shares
    ``domain.W``.

    Examples
    --------
    >>> domain = hm.ShellDomain(hm.read_mesh("plate.xdmf"), element="CG2CG1")
    """

    def __init__(self, mesh, *, element="CG2CG1", cell_tags=None, regions=None, quadrature_degree=4):
        self.mesh = mesh
        self.element = element
        self.W = state_space(mesh, element)
        self.cell_tags = cell_tags
        self.regions = dict(regions) if regions else {}
        self.quadrature_degree = quadrature_degree

        self.n_nodes = mesh.geometry.x.shape[0]
        tdim = mesh.topology.dim
        self.n_cells = mesh.topology.index_map(tdim).size_local

        self.node_input_idx = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
        self.cell_input_idx = np.asarray(mesh.topology.original_cell_index, dtype=np.int64)

        if not np.array_equal(np.sort(self.node_input_idx), np.arange(self.n_nodes)):
            raise ValueError(
                "ShellDomain needs a serial mesh whose input_global_indices form a "
                "node permutation (the same requirement the mesh-coordinate "
                "differentiation path already imposes) -- got a non-bijective index "
                "set, most likely an MPI-parallel mesh."
            )

        self.reverse_node_idx = np.argsort(self.node_input_idx)
        self.reverse_cell_idx = np.argsort(self.cell_input_idx)

        self._space_cache = {}
        self._frames = None
        self._orientation_ok = None    # cell-orientation consistency check, memoized

        # -- file-order geometry, derived from the local-order mesh arrays -----
        self.node_coords = mesh.geometry.x[self.reverse_node_idx].copy()
        local_centroids = _local_cell_centroids(mesh, self.n_cells)
        self.cell_centroids = local_centroids[self.reverse_cell_idx].copy()

        self._check_cg1_matches_local_vertices()
        self._check_dg0_matches_local_cells(local_centroids)

    # -- ordering sanity checks (cheap, run once) --------------------------
    def _check_cg1_matches_local_vertices(self):
        from scipy.spatial import cKDTree

        cg1 = self.function_space(("Lagrange", 1)).tabulate_dof_coordinates()
        dist, match = cKDTree(self.mesh.geometry.x).query(cg1)
        if dist.max() > 1e-10 or not np.array_equal(match, np.arange(len(match))):
            raise ValueError(
                "CG1 dof order does not match the local vertex order on this mesh -- "
                "hm.from_nodal assumes they coincide (it gathers file-order per-vertex "
                "values straight into CG1 dof order) and would silently scatter values "
                "to the wrong vertices here. This holds for any straight-sided, "
                "P1-geometry mesh; something unusual is going on with this one."
            )

    def _check_dg0_matches_local_cells(self, local_centroids):
        from scipy.spatial import cKDTree

        dg0 = self.function_space(("DG", 0)).tabulate_dof_coordinates()
        dist, match = cKDTree(local_centroids).query(dg0)
        if dist.max() > 1e-10 or not np.array_equal(match, np.arange(len(match))):
            raise ValueError(
                "DG0 dof order does not match the local cell order on this mesh -- "
                "hm.from_cells assumes they coincide (it gathers file-order per-cell "
                "values straight into DG0 dof order) and would silently scatter values "
                "to the wrong cells here. This holds for any straight-sided mesh; "
                "something unusual is going on with this one."
            )

    # -- spaces --------------------------------------------------------
    def function_space(self, space):
        """The DOLFINx function space for a space descriptor.

        Parameters
        ----------
        space : tuple
            ``(family, degree)`` or ``(family, degree, value_shape)``.

        Returns
        -------
        dolfinx.fem.FunctionSpace
            Memoized, one per normalized descriptor.
        """
        key = normalize_space(space)
        if key not in self._space_cache:
            self._space_cache[key] = make_space(self.mesh, key)
        return self._space_cache[key]

    def dof_coords(self, space):
        """Coordinates of a space's scalar dofs, in FE dof order.

        Parameters
        ----------
        space : tuple
            ``(family, degree)`` or ``(family, degree, value_shape)``.

        Returns
        -------
        ndarray
            Shape ``(n_scalar_dofs, 3)``. For a blocked (vector or tensor) space
            these are per scalar dof, not per component: all components of one dof
            share a coordinate.

        See Also
        --------
        hermit.from_coeffs, hermit.from_function : build coefficients on this space.
        """
        V = self.function_space(space)
        _, _, value_shape = normalize_space(space)
        if value_shape == ():
            return V.tabulate_dof_coordinates()
        Vs, _ = V.sub(0).collapse()
        return Vs.tabulate_dof_coordinates()

    # -- local orientation frames ---------------------------------------
    def local_frames(self):
        """The orthonormal local shell frame of every cell.

        Returns
        -------
        ndarray
            Shape ``(n_cells, 3, 3)``, rows ``e0``, ``e1`` in-plane and ``e2`` the
            element normal, at the reference geometry, in **FE** cell order.
            Memoized.
        """
        if self._frames is None:
            fe_cells = np.arange(self.n_cells, dtype=np.int32)
            self._frames = self._eval_frames(_cell_centroid_ref_point(self.mesh), fe_cells)
        return self._frames

    def local_frames_at(self, points, cells):
        """The local shell frame at chosen points inside chosen cells.

        Parameters
        ----------
        points : array_like
            Reference-cell coordinates: one point broadcast to every query, or one
            point per query.
        cells : array_like of int
            Cell ids in **file** order.

        Returns
        -------
        ndarray
            Shape ``(n, 3, 3)``; see :meth:`local_frames` for the row convention.
        """
        cells = np.atleast_1d(np.asarray(cells, dtype=np.int64))
        fe_cells = self.reverse_cell_idx[cells].astype(np.int32)
        return self._eval_frames(points, fe_cells)

    # -- cell-orientation consistency ------
    def check_cell_orientation_consistency(self, tol=0.0):
        """Verify that cell normals do not flip across shared interior facets.

        Called by :func:`~hermit.pressure` and :func:`~hermit.edge_pressure`, whose
        sign convention is only well defined on a consistently wound mesh.

        Parameters
        ----------
        tol : float, optional
            Two adjacent normals are inconsistent when their dot product is at or
            below this. Default 0.

        Raises
        ------
        ValueError
            Naming the offending cell pair. Fix the mesh winding upstream, or use
            :func:`~hermit.traction`, which takes an explicit global vector.

        Notes
        -----
        Cheap and memoized; a no-op after the first successful call.
        """
        if self._orientation_ok is not None:
            if not self._orientation_ok:
                raise ValueError(self._orientation_error)
            return
        import dolfinx

        tdim = self.mesh.topology.dim
        fdim = tdim - 1
        self.mesh.topology.create_connectivity(fdim, tdim)
        f2c = self.mesh.topology.connectivity(fdim, tdim)
        n = self.local_frames()[:, 2, :]
        bad = None
        for f in range(f2c.num_nodes):
            cells = f2c.links(f)
            if len(cells) != 2:
                continue
            a, b = int(cells[0]), int(cells[1])
            if np.dot(n[a], n[b]) <= tol:
                bad = (a, b)
                break
        if bad is not None:
            self._orientation_ok = False
            self._orientation_error = (
                f"ShellDomain: cell normals are inconsistently oriented across a "
                f"shared facet (e.g. FE cells {bad[0]}/{bad[1]} have normals "
                f"pointing opposite ways) -- hm.pressure's sign convention "
                f"(positive p acts along +n) is not well defined on this mesh. Fix "
                f"the mesh winding upstream, or use hm.traction instead (it takes "
                f"an explicit global vector and sidesteps CellNormal entirely)."
            )
            raise ValueError(self._orientation_error)
        self._orientation_ok = True

    def _eval_frames(self, points, fe_cells):
        """``(n, 3, 3)`` frames at reference ``points`` (broadcast, or one per query) in
        **FE**-order ``fe_cells`` -- ``dolfinx.fem.Expression(Ei, points).eval(mesh,
        fe_cells)`` per basis vector."""
        import dolfinx

        from .fenics.kinematics import local_basis_inplane

        E0, E1, E2 = local_basis_inplane(self.mesh)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        n, n_pts = len(fe_cells), len(points)
        frames = np.zeros((n, 3, 3))
        for i, Ei in enumerate((E0, E1, E2)):
            out = dolfinx.fem.Expression(Ei, points).eval(self.mesh, fe_cells).reshape(n, n_pts, 3)
            frames[:, i, :] = out[:, 0, :] if n_pts == 1 else out[np.arange(n), np.arange(n), :]
        return frames


def _local_cell_centroids(mesh, n_cells):
    """``(n_cells, 3)`` cell centroids, **local** order (straight-sided cells)."""
    import dolfinx

    tdim = mesh.topology.dim
    cells = np.arange(n_cells, dtype=np.int32)
    return dolfinx.mesh.compute_midpoints(mesh, tdim, cells)


def _cell_centroid_ref_point(mesh):
    """The reference-cell centroid (parametric coordinates), as a ``(1, tdim)`` array."""
    import basix
    import dolfinx

    ct = dolfinx.fem.functionspace(mesh, ("DG", 0)).element.basix_element.cell_type
    return basix.cell.geometry(ct).mean(axis=0)[None, :]


# -- module-level helpers --------------------------------------------------

def read_mesh(path, name="Grid"):
    """Read a mesh from an XDMF file.

    Parameters
    ----------
    path : str or pathlib.Path
    name : str, optional
        Grid name inside the file. Default ``"Grid"``.

    Returns
    -------
    dolfinx.mesh.Mesh
        Read on ``MPI.COMM_WORLD``. Run serially -- :class:`ShellDomain` requires a
        serial mesh.

    Examples
    --------
    >>> domain = hm.ShellDomain(hm.read_mesh("plate.xdmf"))
    """
    import dolfinx
    from mpi4py import MPI

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, str(path), "r") as f:
        return f.read_mesh(name=name)
