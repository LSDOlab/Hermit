"""``Geometry`` -- the mesh coordinate field: reference configuration, reference +
displacement (a shape design variable), or an absolute coordinate array.

Resolves the mesh coordinate field onto a ``Field``, off
``hermit.csdl_helpers.resolve_mesh_nodes``.
"""

from __future__ import annotations

import numpy as np
import csdl_alpha as csdl

from . import _compat
from ._field import Field

# The mesh coordinate element is fixed by the mesh, not a free choice -- for Hermit's
# straight-sided (P1 geometry) meshes it coincides with ("Lagrange", 1, (3,)); see
# _assert_p1_coordinate_element.
_GEOM_SPACE = ("Lagrange", 1, (3,))


def _assert_p1_coordinate_element(mesh):
    """``Geometry`` wraps a ``Field`` on ``("Lagrange", 1, (3,))`` -- the space that
    coincides with the mesh's own coordinate element only when the geometry is P1
    (straight-sided). Assert this rather than assume it, mirroring
    ``ShellDomain.__init__``'s CG1/DG0 ordering checks (a P1-only assumption is
    silently wrong on an isoparametric / curved-geometry mesh otherwise)."""
    degree = _compat.coordinate_element(mesh).degree
    if degree != 1:
        raise ValueError(
            f"hm.geometry() assumes a P1 (straight-sided) mesh coordinate element -- "
            f"this mesh has a degree-{degree} geometry, which does not coincide with "
            f"('Lagrange', 1, (3,)). An isoparametric / curved-geometry mesh needs a "
            f"different Geometry space (not implemented)."
        )


class Geometry:
    """The mesh coordinate field, in any configuration :func:`~hermit.solve` accepts.

    Build one with :func:`geometry`. Giving neither ``node_disp`` nor ``nodes``
    selects the reference configuration and no mesh-derivative forms are built, so
    leaving it out costs nothing.

    Parameters
    ----------
    domain : ShellDomain
    node_disp : csdl.Variable or array_like, optional
        ``(n_nodes, 3)`` displacement added to the reference coordinates, in
        **file** vertex order. The usual shape design variable.
    nodes : csdl.Variable or array_like, optional
        ``(n_nodes, 3)`` absolute coordinates, in **file** vertex order.

    Attributes
    ----------
    field : Field
        The coordinates as a ``("Lagrange", 1, (3,))`` field, for uniformity with
        every other input.
    nodes : csdl.Variable
        ``(n_nodes, 3)`` file-order coordinates, the layout the FEniCSx operations
        consume.
    is_differentiable : bool
        True when ``node_disp`` or ``nodes`` was supplied.

    Raises
    ------
    ValueError
        If both ``node_disp`` and ``nodes`` are given, or if the mesh has a
        higher-order (curved) coordinate element.
    """

    def __init__(self, domain, *, node_disp=None, nodes=None):
        if node_disp is not None and nodes is not None:
            raise ValueError(
                "hm.geometry(): give either node_disp= or nodes=, not both -- "
                "matching hermit.csdl_helpers.resolve_mesh_nodes's rule."
            )
        _assert_p1_coordinate_element(domain.mesh)
        self.domain = domain

        ref = csdl.Variable(value=np.asarray(domain.node_coords, dtype=float))
        if nodes is not None:
            self.nodes = (nodes if isinstance(nodes, csdl.Variable)
                         else csdl.Variable(value=np.asarray(nodes, dtype=float)))
            self.is_differentiable = True
        elif node_disp is not None:
            disp = (node_disp if isinstance(node_disp, csdl.Variable)
                   else csdl.Variable(value=np.asarray(node_disp, dtype=float)))
            self.nodes = ref + disp
            self.is_differentiable = True
        else:
            self.nodes = ref
            self.is_differentiable = False

        # file -> FE (== local, for CG1) permutation, the same gather from_nodal uses.
        fe_coeffs = self.nodes[list(domain.node_input_idx)]
        self.field = Field(domain, _GEOM_SPACE, fe_coeffs, kind="vector3", global_frame=True)


def geometry(domain, *, node_disp=None, nodes=None) -> Geometry:
    """Build the geometry input for a solve.

    Parameters
    ----------
    domain : ShellDomain
    node_disp : csdl.Variable or array_like, optional
        ``(n_nodes, 3)`` displacement from the reference coordinates, file order.
    nodes : csdl.Variable or array_like, optional
        ``(n_nodes, 3)`` absolute coordinates, file order.

    Returns
    -------
    Geometry
        The reference configuration when neither argument is given.

    Examples
    --------
    >>> nd = csdl.Variable(value=np.zeros((domain.n_nodes, 3)), name="node_disp")
    >>> state = hm.solve(domain, mat, loads, bcs,
    ...                  geometry=hm.geometry(domain, node_disp=nd))
    """
    return Geometry(domain, node_disp=node_disp, nodes=nodes)
