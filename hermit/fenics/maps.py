"""Setup-time index / operator maps between user ordering and FEniCSx ordering.

Port of the ``_material_indices`` / ``_pressure_indices`` / ``reference_mesh_nodes`` /
``construct_*_map`` machinery from ``RMShellModel`` + ``RMShellPDE`` (femo dev_coupling).
Pure numpy / scipy, built once.
"""

import numpy as np
import scipy.sparse as sp


class OrderingMaps:
    def __init__(self, mesh, pde, *, element_wise_material=False, elementwise_pressure=False):
        """``element_wise_material`` / ``elementwise_pressure`` select whether the
        material / pressure user->FE gather runs over cells or nodes -- the same two
        ``ShellPDE`` fixed-space switches, passed explicitly."""
        self.mesh = mesh
        self.node_input_idx = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
        self.cell_input_idx = np.asarray(mesh.topology.original_cell_index, dtype=np.int64)
        # user -> FE gather indices
        self.material_idx = (
            self.cell_input_idx if element_wise_material else self.node_input_idx
        )
        self.pressure_idx = (
            self.cell_input_idx if elementwise_pressure else self.node_input_idx
        )
        # FE -> user (for reordering field outputs back)
        self.reverse_node_idx = np.argsort(self.node_input_idx)
        self.reverse_cell_idx = np.argsort(self.cell_input_idx)

        if not np.array_equal(np.sort(self.node_input_idx), np.arange(self.node_input_idx.size)):
            raise ValueError(
                "mesh-coordinate differentiation needs a serial mesh whose "
                "input_global_indices form a node permutation"
            )
        self.reference_mesh_nodes = np.empty_like(mesh.geometry.x)
        self.reference_mesh_nodes[self.node_input_idx] = mesh.geometry.x

        self.vf_size = pde.f.x.array.size
        self.force_to_pressure = pde.force_to_pressure_matrix()
        self._pde = pde  # for lazily-built extraction maps (Phase 4b)
        self._nodal_map = {}

    @property
    def geom_shape(self):
        return self.mesh.geometry.x.shape

    # -- nodal field extraction (Phase 4b) -----------------------------
    # ``disp_solid`` (mixed CG2xCG1 dof vector) -> per-node (x, y, z) values at the
    # mesh vertices. The mixed displacement sub-element is CG2 and its rotation
    # sub-element CG1; the CG1 vertex nodes are a subset of the CG2 nodes, so this
    # is an exact *selection* (geometry-independent), assembled once as a sparse
    # (3*n_nodes, ndof) matrix. Component-major layout to match femo's
    # ``DisplacementExtractionModel`` (reshape (3, n_nodes) then transpose).
    def _nodal_selection_matrix(self, sub: int):
        from scipy.spatial import cKDTree

        W = self._pde.W
        ndof = W.dofmap.index_map.size_local * W.dofmap.index_map_bs
        Vsub, sub_dofs = W.sub(sub).collapse()
        sub_dofs = np.asarray(sub_dofs).reshape(-1)          # Vsub dof -> W dof
        scalar_x = Vsub.sub(0).collapse()[0].tabulate_dof_coordinates()
        dist, node_to_scalar = cKDTree(scalar_x).query(self.mesh.geometry.x)
        if dist.max() > 1e-8:
            raise RuntimeError(
                f"sub-space {sub} has no dof at every mesh vertex, so nodal extraction "
                f"is not defined for it (e.g. the Crouzeix-Raviart rotation of 'CG2CR1', "
                f"whose dofs live at edge midpoints). Use state.rotation() -- the native "
                f"field -- or evaluate it where you need it."
            )
        comp = [np.asarray(Vsub.sub(i).collapse()[1]).reshape(-1) for i in range(3)]  # scalar -> Vsub
        nn = self.mesh.geometry.x.shape[0]
        rows = np.arange(3 * nn)
        cols = np.array([sub_dofs[comp[i][node_to_scalar[k]]] for i in range(3) for k in range(nn)])
        return sp.csr_matrix((np.ones(3 * nn), (rows, cols)), shape=(3 * nn, ndof))

    @property
    def nodal_disp_map(self):
        if "disp" not in self._nodal_map:
            self._nodal_map["disp"] = self._nodal_selection_matrix(0)
        return self._nodal_map["disp"]

    @property
    def nodal_rot_map(self):
        if "rot" not in self._nodal_map:
            self._nodal_map["rot"] = self._nodal_selection_matrix(1)
        return self._nodal_map["rot"]
