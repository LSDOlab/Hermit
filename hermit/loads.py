"""``Loads`` -- distributed, edge and point loads; ``a + b`` composes them.

Each builder returns a :class:`Loads` carrying one term. ``pressure`` / ``traction``
/ ``moment`` are distributed over the shell surface; ``edge_pressure`` /
``edge_traction`` / ``edge_moment`` act on selected exterior facets; ``point_load``
is a consistent (weak Dirac) point load; ``load_vector`` is a direct generalised RHS
in ``domain.W`` dof ordering.

Every term reaches :func:`hermit.solve` as its own residual and compliance
contribution, on its own space, with no interpolation between terms. ``pressure``
follows the live, shape-differentiable ``ufl.CellNormal``.

``combined_traction`` / ``combined_moment`` reduce several terms onto one shared
space. They are not part of the solve path -- they exist for callers that drive the
raw FEniCSx operations directly.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import csdl_alpha as csdl
from dataclasses import dataclass

from dolfinx import mesh as dmesh

from .fenics.spaces import normalize_space
from ._field import Field, as_field, as_field_user_order, constant, field_fn
from .transfer import interpolate
from .bcs import _compiled_measure

_VEC3 = (3,)


@dataclass(frozen=True)
class _EdgeLoadTerm:
    """A coefficient and the exterior-facet measure on which it acts.

    Keeping the compiled measure (and therefore its MeshTags) alive on the load
    object is important: UFL forms retain the measure, but do not own the Python
    object which supplies its subdomain data.
    """
    field: Field
    ds: object
    facets: np.ndarray


def _edge_term(domain, value, space, *, vector, where, name):
    """Build one selected-exterior-facet load term using BC's measure machinery."""
    if vector:
        sp_ = (*normalize_space(space)[:2], _VEC3)
        field = value if isinstance(value, Field) else _as_vector_field(domain, value, sp_)
        if field.value_shape != _VEC3:
            raise ValueError(f"{name}: value_shape must be (3,), got {field.value_shape}")
    else:
        sp_ = normalize_space(space)
        if sp_[2] != ():
            raise ValueError(f"{name}: space must be scalar, got value_shape {sp_[2]}")
        field = value if isinstance(value, Field) else as_field_user_order(domain, value, sp_)

    fdim = domain.mesh.topology.dim - 1
    facets = dmesh.locate_entities_boundary(domain.mesh, fdim, where)
    ds = _compiled_measure(domain.mesh, fdim, facets, "ds", domain.quadrature_degree)
    return _EdgeLoadTerm(field, ds, np.asarray(facets, dtype=np.int32))


def _as_vector_field(domain, value, default_space) -> Field:
    """Coerce a bare ``(3,)`` vector (broadcast to every dof) or anything
    ``as_field`` resolves to a ``Field`` -- mirrors
    ``hermit.material._as_vector_field`` (a length-3 value means "broadcast", not
    "one value per dof", which ``as_field``'s own resolution order does not
    special-case)."""
    is_var = isinstance(value, csdl.Variable)
    shape = tuple(value.shape) if is_var else np.asarray(value).shape
    total = int(np.prod(shape)) if shape else 1
    if total == 3:
        return constant(domain, default_space, value)
    return as_field_user_order(domain, value, default_space)


def _combine(domain, fields, space):
    """Sum a list of ``Field`` objects onto one common ``space`` -- ``interpolate`` is the
    identity when a field is already on ``space``, so a single-term list already on
    ``space`` is combined exactly, with no approximation."""
    if not fields:
        return None
    space = normalize_space(space)
    out = None
    for f in fields:
        f2 = f if f.space == space else interpolate(f, space)
        out = f2 if out is None else field_fn(out, f2, fn=lambda a, b: a + b)
    return out


class Loads:
    """A composable bundle of load terms.

    Build one with :func:`pressure`, :func:`traction`, :func:`moment`,
    :func:`edge_pressure`, :func:`edge_traction`, :func:`edge_moment`,
    :func:`point_load` or :func:`load_vector` rather than calling this constructor
    directly. ``a + b`` merges two ``Loads`` on the same domain, concatenating the
    term lists and summing the direct right-hand sides.

    Parameters
    ----------
    domain : ShellDomain
    traction_terms, moment_terms : sequence of Field, optional
        Distributed ``(3,)`` vector fields.
    pressure_terms : sequence of Field, optional
        Distributed scalar fields, acting along the shell normal.
    edge_traction_terms, edge_moment_terms, edge_pressure_terms : sequence, optional
        Edge terms, each pairing a field with the exterior-facet measure it acts on.
    direct : csdl.Variable, optional
        A generalised right-hand side in ``domain.W`` dof ordering.

    Examples
    --------
    >>> loads = hm.pressure(domain, 2.0) + hm.point_load(domain, at=tip, force=[0, 0, -1])
    """

    def __init__(self, domain, *, traction_terms=(), moment_terms=(), pressure_terms=(),
                edge_traction_terms=(), edge_moment_terms=(), edge_pressure_terms=(), direct=None):
        self.domain = domain
        self.traction_terms = list(traction_terms)   # list[Field], each (.., (3,))
        self.moment_terms = list(moment_terms)        # list[Field], each (.., (3,))
        self.pressure_terms = list(pressure_terms)     # list[Field], each scalar
        self.edge_traction_terms = list(edge_traction_terms)  # list[_EdgeLoadTerm]
        self.edge_moment_terms = list(edge_moment_terms)      # list[_EdgeLoadTerm]
        self.edge_pressure_terms = list(edge_pressure_terms)  # list[_EdgeLoadTerm]
        self.direct = direct                           # csdl.Variable (ndof_W,) | None

    def __add__(self, other):
        if not isinstance(other, Loads):
            return NotImplemented
        if other.domain is not self.domain:
            raise ValueError("Loads.__add__ needs both operands on the same ShellDomain")
        if self.direct is None:
            direct = other.direct
        elif other.direct is None:
            direct = self.direct
        else:
            direct = self.direct + other.direct
        return Loads(self.domain, traction_terms=self.traction_terms + other.traction_terms,
                    moment_terms=self.moment_terms + other.moment_terms,
                    pressure_terms=self.pressure_terms + other.pressure_terms,
                    edge_traction_terms=self.edge_traction_terms + other.edge_traction_terms,
                    edge_moment_terms=self.edge_moment_terms + other.edge_moment_terms,
                    edge_pressure_terms=self.edge_pressure_terms + other.edge_pressure_terms,
                    direct=direct)

    __radd__ = __add__

    # -- reduction (see the module docstring) --------------------------------
    def combined_traction(self, space=("Lagrange", 1, (3,))) -> Field | None:
        """Sum the traction and pressure terms onto one space.

        Pressure is converted to a traction with a numpy snapshot of the reference
        normal. :func:`~hermit.solve` does not use this -- it takes each term
        directly, with the live normal.

        Parameters
        ----------
        space : tuple, optional
            Target space. Exact when every term is already on it.

        Returns
        -------
        Field or None
            ``None`` when there are no such terms.
        """
        converted = [_pressure_to_traction(self.domain, p) for p in self.pressure_terms]
        return _combine(self.domain, self.traction_terms + converted, space)

    def combined_moment(self, space=("Lagrange", 1, (3,))) -> Field | None:
        """Sum the distributed moment terms onto one space.

        Parameters
        ----------
        space : tuple, optional
            Target space. Exact when every term is already on it.

        Returns
        -------
        Field or None
            ``None`` when there are no moment terms.
        """
        return _combine(self.domain, self.moment_terms, space)

    def direct_vector(self, ndof=None) -> csdl.Variable:
        """The combined point-load and load-vector right-hand side.

        Parameters
        ----------
        ndof : int, optional
            Length to use when there is no direct term. Defaults to the number of
            ``domain.W`` dofs.

        Returns
        -------
        csdl.Variable
            Shape ``(ndof,)`` in ``domain.W`` dof order; zeros when there is no
            point load or load vector.
        """
        if self.direct is not None:
            return self.direct
        if ndof is None:
            W = self.domain.W
            ndof = W.dofmap.index_map.size_local * W.dofmap.index_map_bs
        return csdl.Variable(value=np.zeros(ndof))


# -- distributed loads --------------------------------------------------------

def traction(domain, t, *, space=("Lagrange", 1)) -> Loads:
    """Distributed force per unit area, in global components.

    Contributes ``int t . du dx`` to the residual. Unlike :func:`pressure` this
    never touches the shell normal, so it is the right choice for a body force such
    as self weight on a curved surface.

    Parameters
    ----------
    domain : ShellDomain
    t : Field or array_like
        A ``(3,)`` global vector, or a ``Field`` of ``(3,)`` vectors.
    space : tuple, optional
        Space for a non-``Field`` value. Default ``("Lagrange", 1)``.

    Returns
    -------
    Loads

    Raises
    ------
    ValueError
        If ``t`` does not resolve to a ``(3,)`` vector field.

    See Also
    --------
    pressure : a scalar load following the shell normal.
    edge_traction : the same load on selected exterior facets.
    """
    sp_ = (*normalize_space(space)[:2], _VEC3)
    field = t if isinstance(t, Field) else _as_vector_field(domain, t, sp_)
    if field.value_shape != _VEC3:
        raise ValueError(f"traction: value_shape must be (3,), got {field.value_shape}")
    return Loads(domain, traction_terms=[field])


def moment(domain, m, *, space=("Lagrange", 1)) -> Loads:
    """Distributed moment per unit area, in global components.

    Contributes ``int m . dtheta dx`` to the residual.

    Parameters
    ----------
    domain : ShellDomain
    m : Field or array_like
        A ``(3,)`` global vector, or a ``Field`` of ``(3,)`` vectors.
    space : tuple, optional
        Space for a non-``Field`` value. Default ``("Lagrange", 1)``.

    Returns
    -------
    Loads

    Raises
    ------
    ValueError
        If ``m`` does not resolve to a ``(3,)`` vector field.
    """
    sp_ = (*normalize_space(space)[:2], _VEC3)
    field = m if isinstance(m, Field) else _as_vector_field(domain, m, sp_)
    if field.value_shape != _VEC3:
        raise ValueError(f"moment: value_shape must be (3,), got {field.value_shape}")
    return Loads(domain, moment_terms=[field])


def pressure(domain, p, *, space=("DG", 0)) -> Loads:
    """Scalar pressure acting along the shell normal.

    Positive ``p`` acts along ``+n``. The residual uses the live
    ``ufl.CellNormal``, so the load is differentiable with respect to the mesh
    coordinates.

    Parameters
    ----------
    domain : ShellDomain
    p : Field, float or array_like
        Scalar pressure.
    space : tuple, optional
        Space for a non-``Field`` value. Default ``("DG", 0)``.

    Returns
    -------
    Loads

    Raises
    ------
    ValueError
        If ``space`` is not scalar, or if the domain's cell normals are not
        consistently oriented. On a badly-wound imported mesh a uniform pressure
        would otherwise become a sign-alternating load with no other symptom.

    See Also
    --------
    traction : an explicit global vector, independent of the normal.
    """
    domain.check_cell_orientation_consistency()
    sp_ = normalize_space(space)
    if sp_[2] != ():
        raise ValueError(f"pressure: space must be scalar, got value_shape {sp_[2]}")
    p_field = p if isinstance(p, Field) else as_field_user_order(domain, p, sp_)
    return Loads(domain, pressure_terms=[p_field])


def edge_traction(domain, t, *, where, space=("Lagrange", 1)) -> Loads:
    """Force per unit length on selected exterior facets.

    Parameters
    ----------
    domain : ShellDomain
    t : Field or array_like
        A ``(3,)`` global vector, or a ``Field`` of ``(3,)`` vectors.
    where : callable
        Coordinate predicate selecting the loaded edge, with the same contract as
        :func:`~hermit.clamp`: it receives a ``(3, N)`` coordinate array and
        returns an ``(N,)`` boolean mask. Only exterior facets are selected.
    space : tuple, optional
        Space for a non-``Field`` value. Default ``("Lagrange", 1)``.

    Returns
    -------
    Loads

    Raises
    ------
    ValueError
        If ``t`` does not resolve to a ``(3,)`` vector field.

    Examples
    --------
    >>> tip = hm.edge_traction(domain, [0.0, 0.0, -1.0], where=hm.near("x", 10.0))
    """
    return Loads(domain, edge_traction_terms=[
        _edge_term(domain, t, space, vector=True, where=where, name="edge_traction")
    ])


def edge_moment(domain, m, *, where, space=("Lagrange", 1)) -> Loads:
    """Moment per unit length on selected exterior facets.

    Parameters
    ----------
    domain : ShellDomain
    m : Field or array_like
        A ``(3,)`` global vector, or a ``Field`` of ``(3,)`` vectors.
    where : callable
        Coordinate predicate selecting the loaded edge; see :func:`edge_traction`.
    space : tuple, optional
        Space for a non-``Field`` value. Default ``("Lagrange", 1)``.

    Returns
    -------
    Loads

    Raises
    ------
    ValueError
        If ``m`` does not resolve to a ``(3,)`` vector field.
    """
    return Loads(domain, edge_moment_terms=[
        _edge_term(domain, m, space, vector=True, where=where, name="edge_moment")
    ])


def edge_pressure(domain, p, *, where, space=("DG", 0)) -> Loads:
    """Pressure per unit length along the shell normal, on selected exterior facets.

    Parameters
    ----------
    domain : ShellDomain
    p : Field, float or array_like
        Scalar pressure; positive acts along ``+n``.
    where : callable
        Coordinate predicate selecting the loaded edge; see :func:`edge_traction`.
    space : tuple, optional
        Space for a non-``Field`` value. Default ``("DG", 0)``.

    Returns
    -------
    Loads

    Raises
    ------
    ValueError
        If ``space`` is not scalar, or if the domain's cell normals are not
        consistently oriented.
    """
    domain.check_cell_orientation_consistency()
    return Loads(domain, edge_pressure_terms=[
        _edge_term(domain, p, space, vector=False, where=where, name="edge_pressure")
    ])


def _pressure_to_traction(domain, p_field) -> Field:
    """The Phase-E numpy-snapshot ``p * n̂`` conversion -- ``n̂`` evaluated once, at
    the reference geometry (``domain.local_frames()``), not the live
    ``ufl.CellNormal``. Exact on a flat mesh (one constant normal); only piecewise-constant-
    accurate on a curved one built from non-planar cells. Kept for ``combined_traction()``
    (pre-Phase-F raw-ops tests); ``hm.solve`` never calls this -- see the module
    docstring."""
    p_dg0 = p_field if p_field.space[:2] == ("DG", 0) else interpolate(p_field, ("DG", 0))
    n = domain.local_frames()[:, 2, :]                       # (n_cells, 3), FE order
    n_var = csdl.Variable(value=n)
    p_expand = csdl.expand(p_dg0.coeffs, (domain.n_cells, 3), action="i->ia")
    t_coeffs = csdl.reshape(p_expand * n_var, (domain.n_cells * 3,))
    return Field(domain, ("DG", 0, _VEC3), t_coeffs, kind="vector3", frame=domain.local_frames())


# -- point / direct loads ------------------------------------------------------

def load_vector(domain, vec) -> Loads:
    """A generalised right-hand side given directly in state-space dof order.

    Parameters
    ----------
    domain : ShellDomain
    vec : csdl.Variable or array_like
        One entry per ``domain.W`` dof, in that space's dof ordering.

    Returns
    -------
    Loads

    Raises
    ------
    ValueError
        If ``vec`` does not have exactly ``domain.W``'s dof count.
    """
    W = domain.W
    ndof = W.dofmap.index_map.size_local * W.dofmap.index_map_bs
    v = vec if isinstance(vec, csdl.Variable) else csdl.Variable(value=np.asarray(vec, dtype=float))
    if int(np.prod(v.shape)) != ndof:
        raise ValueError(f"load_vector: expected {ndof} entries (domain.W dofs), got {v.shape}")
    v = v if v.shape == (ndof,) else csdl.reshape(v, (ndof,))
    return Loads(domain, direct=v)


def _locate_point(mesh, point):
    """(FE cell id, reference coords) for the physical ``point`` -- raises if it
    lies in no cell."""
    import dolfinx

    from ._compat import coordinate_element, geometry_dofmap

    pt = np.asarray(point, dtype=float).reshape(1, 3)
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    coll = dolfinx.geometry.compute_collisions_points(tree, pt)
    cells = dolfinx.geometry.compute_colliding_cells(mesh, coll, pt).links(0)
    if len(cells) == 0:
        raise ValueError(f"point_load: at={point!r} lies in no cell of this mesh")
    cell = int(cells[0])
    cmap = coordinate_element(mesh)
    geom_dofmap = geometry_dofmap(mesh)
    cell_geom = mesh.geometry.x[geom_dofmap[cell]]
    ref = cmap.pull_back(pt, cell_geom)
    return cell, ref[0]


def _point_scatter(W, sub_index, cell, ref):
    """``(bs, ndof_W)`` sparse matrix -- the sub-space (``W.sub(sub_index)``,
    collapsed) basis functions evaluated at ``(cell, ref)``, scattered into the
    full mixed-``W`` dof numbering. Reuses ``hermit._field._Tabulation``'s point
    tabulation (built for exactly this: basis values at an arbitrary reference
    point in a given cell) on the collapsed sub-space, then remaps its columns
    through the sub-dof -> W-dof map (the same map ``hermit.fenics.maps.
    OrderingMaps._nodal_selection_matrix`` uses for nodal extraction, here at an
    arbitrary point instead of only at mesh vertices)."""
    from ._field import _Tabulation

    Vsub, sub_dofs = W.sub(sub_index).collapse()
    sub_dofs = np.asarray(sub_dofs).reshape(-1)
    tab = _Tabulation(Vsub)
    B = tab.point_matrix(np.array([cell], dtype=np.int64), np.asarray(ref).reshape(1, -1))
    B = B.tocoo()
    ndof_W = W.dofmap.index_map.size_local * W.dofmap.index_map_bs
    return sp.csr_matrix((B.data, (B.row, sub_dofs[B.col])), shape=(B.shape[0], ndof_W))


def point_load(domain, *, at, force=None, moment=None) -> Loads:
    """A consistent point force and/or moment at a physical coordinate.

    The cell containing ``at`` is located, the state-space basis is evaluated
    there, and the result is scattered into the direct right-hand side -- the weak
    form of a Dirac delta, ``int F . delta(x - at) . du dx = F . du(at)``. The
    point need not be a mesh vertex.

    Parameters
    ----------
    domain : ShellDomain
    at : array_like
        Physical coordinate, shape ``(3,)``.
    force : csdl.Variable or array_like, optional
        ``(3,)`` point force.
    moment : csdl.Variable or array_like, optional
        ``(3,)`` point moment.

    Returns
    -------
    Loads

    Raises
    ------
    ValueError
        If neither ``force`` nor ``moment`` is given, or if ``at`` lies in no cell.

    Notes
    -----
    Differentiable in ``force`` and ``moment``, but **not** in ``at`` or the mesh
    coordinates: the containing cell is fixed at construction.
    """
    if force is None and moment is None:
        raise ValueError("point_load needs force= and/or moment=")
    W = domain.W
    ndof_W = W.dofmap.index_map.size_local * W.dofmap.index_map_bs
    cell, ref = _locate_point(domain.mesh, at)
    direct = csdl.Variable(value=np.zeros(ndof_W))
    if force is not None:
        force_v = force if isinstance(force, csdl.Variable) else csdl.Variable(value=np.asarray(force, dtype=float))
        B = _point_scatter(W, 0, cell, ref)
        direct = direct + csdl.sparse.matvec(sp.csr_matrix(B.T), force_v)
    if moment is not None:
        moment_v = moment if isinstance(moment, csdl.Variable) else csdl.Variable(value=np.asarray(moment, dtype=float))
        B = _point_scatter(W, 1, cell, ref)
        direct = direct + csdl.sparse.matvec(sp.csr_matrix(B.T), moment_v)
    return Loads(domain, direct=direct)
