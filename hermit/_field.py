"""``Field`` -- the input/output primitive: CSDL coefficients on a named FE space.

Every geometry, material, load and output field is a :class:`Field`: a ``domain``, a
normalized space descriptor, and flat ``coeffs`` in FE dof order. The builders here
construct one from raw coefficients, a uniform value, a callable, or a file-ordered
per-vertex / per-cell array.

``interpolate`` / ``project`` (space-to-space transfer) live in ``hermit.transfer``;
two ``Field``\\s combine directly only when they already share a space.
"""

from __future__ import annotations

import weakref

import basix
import numpy as np
import scipy.sparse as sp
import csdl_alpha as csdl

from .fenics.spaces import normalize_space

# component layout -> how the components rotate under a change of frame
#   "scalar"  : unchanged
#   "vector3" : a 3-vector, rotated by the full 3x3 frame change
#   "strain2" : in-plane symmetric tensor in Voigt/engineering form [xx, yy, 2*xy]
#   "shear2"  : in-plane 2-vector [xz, yz]
#   "tensor3" : global symmetric 3x3 tensor, Voigt/engineering [xx, yy, zz, 2yz, 2xz, 2xy]
_KINDS = ("scalar", "vector3", "strain2", "shear2", "tensor3")

# Voigt-6 index order for "tensor3"
_V6 = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_V6_ENG = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])   # engineering-shear scaling


class _Tabulation:
    """Basis tabulation for one (scalar or blocked) FE space. One instance per
    (domain, normalized space) pair, memoized in ``_TAB_CACHE``."""

    def __init__(self, V):
        self.V = V
        self.mesh = V.mesh
        self.bs = V.dofmap.bs
        self.n_scalar_dofs = V.dofmap.index_map.size_local
        self.n_dofs = self.n_scalar_dofs * self.bs
        try:
            self._be = V.element.basix_element
        except RuntimeError:
            # a Quadrature element has no plain basix C++ element (dofs are the
            # quadrature-point values themselves, no "basis function" to tabulate at
            # an arbitrary point) -- eval() / cell_values() aren't defined on it.
            self._be = None
        self.is_dg = True if self._be is None else bool(getattr(self._be, "discontinuous", False))
        self._dofmap = np.asarray(V.dofmap.list)          # (n_cells, n_local_nodes)
        tdim = self.mesh.topology.dim
        self.n_cells = self.mesh.topology.index_map(tdim).size_local
        # FE cell owning each scalar dof (unambiguous for DG; last-writer for CG)
        self.dof_cell = np.empty(self.n_scalar_dofs, dtype=np.int64)
        self.dof_cell[self._dofmap.ravel()] = np.repeat(
            np.arange(self._dofmap.shape[0]), self._dofmap.shape[1])
        self._cache = {}

    # -- basis matrices (B[q, dof] = phi_dof(ref_point_q), block-expanded) ----
    def _matrix(self, cell_ids, ref_points):
        if self._be is None:
            raise ValueError(
                "eval() / cell_values() are not defined on a Quadrature space -- its "
                "dofs are the quadrature-point values themselves, with no basis "
                "function to tabulate at an arbitrary point. Use domain.dof_coords "
                "to see where each dof lives, or read .coeffs directly.")
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        ref_points = np.atleast_2d(np.asarray(ref_points, dtype=float))
        per_point = ref_points.shape[0] == cell_ids.shape[0] and len(ref_points) > 1
        rows, cols, data = [], [], []
        # tabulate once if every query uses the same reference point
        tab_all = None if per_point else self._be.tabulate(0, ref_points[:1])[0, 0, :, 0]
        for q, cell in enumerate(cell_ids):
            phi = self._be.tabulate(0, ref_points[q:q + 1])[0, 0, :, 0] if per_point else tab_all
            nodes = self._dofmap[cell]
            for c in range(self.bs):
                rows.extend([q * self.bs + c] * len(nodes))
                cols.extend((nodes * self.bs + c).tolist())
                data.extend(phi.tolist())
        return sp.csr_matrix((data, (rows, cols)),
                             shape=(len(cell_ids) * self.bs, self.n_dofs))

    def cell_value_matrix(self):
        """(n_cells*bs, n_dofs) -- the centroid value of every cell."""
        if "cellval" not in self._cache:
            if self._be is None:
                self._matrix(np.empty(0, dtype=np.int64), np.zeros((1, 1)))  # raises
            centroid = basix.cell.geometry(self._be.cell_type).mean(axis=0)
            self._cache["cellval"] = self._matrix(np.arange(self.n_cells), centroid)
        return self._cache["cellval"]

    def point_matrix(self, cell_ids, ref_points):
        return self._matrix(cell_ids, ref_points)


# domain -> {normalized space: _Tabulation}. Keyed on the domain (weakly, so a
# domain's cache disappears with it) rather than the dolfinx FunctionSpace directly --
# FunctionSpace is unhashable.
_TAB_CACHE = weakref.WeakKeyDictionary()


def _tabulation(domain, space) -> _Tabulation:
    per_domain = _TAB_CACHE.get(domain)
    if per_domain is None:
        per_domain = {}
        _TAB_CACHE[domain] = per_domain
    tab = per_domain.get(space)
    if tab is None:
        tab = _Tabulation(domain.function_space(space))
        per_domain[space] = tab
    return tab


def _as_coeffs(coeffs, n_scalar_dofs, bs) -> csdl.Variable:
    """Accept a flat ``(n_scalar_dofs * bs,)`` or ``(n_scalar_dofs, *value_shape)``
    array/Variable and flatten it (C order) to the interleaved block layout
    ``Function(V).x.array`` uses. Non-Variable input is wrapped."""
    if not isinstance(coeffs, csdl.Variable):
        coeffs = csdl.Variable(value=np.atleast_1d(np.asarray(coeffs, dtype=float)))
    n_dofs = n_scalar_dofs * bs
    total = int(np.prod(coeffs.shape)) if coeffs.shape else 1
    if total != n_dofs:
        raise ValueError(
            f"coeffs has {total} entries; expected {n_dofs} ({n_scalar_dofs} scalar "
            f"dofs x block size {bs})")
    return coeffs if coeffs.shape == (n_dofs,) else csdl.reshape(coeffs, (n_dofs,))


class Field:
    """A finite-element field on a :class:`~hermit.ShellDomain`.

    The input/output primitive of the API: every material, load, geometry and
    output field is a ``Field``. Build one with :func:`constant`,
    :func:`from_nodal`, :func:`from_cells`, :func:`from_function` or
    :func:`from_coeffs` rather than calling this constructor directly.

    Parameters
    ----------
    domain : ShellDomain
        The domain the field lives on.
    space : tuple
        FE space descriptor, ``(family, degree)`` or ``(family, degree,
        value_shape)``, e.g. ``("Lagrange", 1)`` or ``("DG", 0, (3,))``.
    coeffs : csdl.Variable or array_like
        Coefficients in **FE dof order**, either flat ``(n_scalar_dofs * bs,)`` or
        ``(n_scalar_dofs, *value_shape)``. Stored flat, in the interleaved block
        layout ``Function(V).x.array`` uses.
    kind : {'scalar', 'vector3', 'strain2', 'shear2', 'tensor3'}, optional
        Component layout, which fixes how the components transform under a change
        of frame. Default ``'scalar'`` (frame-invariant).
    frame : ndarray, optional
        The frame ``coeffs`` are expressed in: ``(3, 3)`` when ``global_frame`` is
        True, otherwise ``(n_cells, 3, 3)`` in FE cell order. ``None`` means
        frame-free.
    global_frame : bool, optional
        True for a single Cartesian frame shared by every dof, False for a
        per-cell local frame. Default False.

    Attributes
    ----------
    coeffs : csdl.Variable
        Flat coefficients, FE dof order.
    space : tuple
        Normalized ``(family, degree, value_shape)``.
    values : ndarray
        Centroid value per cell, file cell order (see :attr:`values`).

    See Also
    --------
    hermit.interpolate, hermit.project : move a field between spaces.

    Notes
    -----
    Two fields combine elementwise (``+ - * /``) only when they already share a
    space; a plain scalar also works, but a ``csdl.Variable`` scalar must go on
    the right (``fld * var``, not ``var * fld``).
    """

    def __init__(self, domain, space, coeffs, *, kind="scalar", frame=None,
                 global_frame=False):
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
        self.domain = domain
        self.space = normalize_space(space)
        self._tab = _tabulation(domain, self.space)
        self.coeffs = _as_coeffs(coeffs, self._tab.n_scalar_dofs, self._tab.bs)
        self.kind = kind
        self.global_frame = bool(global_frame)
        if self.global_frame:
            # a single Cartesian frame (global -> current); identity until rotated
            self.frame = np.eye(3) if frame is None else np.asarray(frame, dtype=float)
        else:
            self.frame = None if frame is None else np.asarray(frame, dtype=float)  # (n_cells,3,3) FE order

    # -- space bookkeeping -----------------------------------------------
    @property
    def V(self):
        return self._tab.V

    @property
    def block_size(self):
        return self._tab.bs

    @property
    def n_scalar_dofs(self):
        return self._tab.n_scalar_dofs

    @property
    def n_dofs(self):
        return self._tab.n_dofs

    @property
    def value_shape(self):
        return self.space[2]

    @property
    def is_dg(self):
        return self._tab.is_dg

    # -- evaluation -----------------------------------------------------
    def eval(self, cells, ref_points):
        """Evaluate the field at reference points inside given cells.

        Parameters
        ----------
        cells : array_like of int
            Cell ids in **file** order, one per query point.
        ref_points : array_like
            Reference-cell coordinates: one point broadcast to every query, or one
            point per query.

        Returns
        -------
        csdl.Variable
            Shape ``(n_query, bs)``, or ``(n_query,)`` for a scalar field.

        Raises
        ------
        ValueError
            If the field is on a Quadrature space, which has no basis to tabulate.
        """
        fe_cells = self.domain.reverse_cell_idx[np.atleast_1d(np.asarray(cells, dtype=np.int64))]
        B = self._tab.point_matrix(fe_cells, ref_points)
        flat = csdl.sparse.matvec(B, self.coeffs)
        return csdl.reshape(flat, (len(fe_cells), self._tab.bs)) if self._tab.bs > 1 else flat

    def cell_values(self):
        """Centroid value of every cell.

        Returns
        -------
        csdl.Variable
            Shape ``(n_cells, bs)``, in **file** cell order.
        """
        flat = csdl.sparse.matvec(self._tab.cell_value_matrix(), self.coeffs)
        m = csdl.reshape(flat, (self._tab.n_cells, self._tab.bs))
        return m[list(self.domain.reverse_cell_idx)]

    @property
    def values(self) -> np.ndarray:
        """ndarray: :meth:`cell_values` as plain numpy -- for inspection and plotting."""
        return np.asarray(self.cell_values().value)

    # -- orientation --------------------------------------------------
    def to_frame(self, target) -> "Field":
        """Re-express a frame-relative field in another frame.

        Parameters
        ----------
        target : array_like
            For a global-frame field, a full ``(3, 3)`` Cartesian frame whose rows
            are the new basis axes in world coordinates; every dof is rotated by
            the same transform.

            For a per-cell field, the direction the in-plane ``e0`` is aligned to:
            a ``(3,)`` global direction, a ``(3, 3)`` frame (row 0 is the
            direction), or a per-cell ``(n_cells, 3)`` / ``(n_cells, 3, 3)`` array
            in **file** cell order. The tangent plane (``e2``) is unchanged.

        Returns
        -------
        Field
            The same field in the new frame. A scalar or frame-free field is
            returned unchanged.

        Raises
        ------
        ValueError
            If a per-cell field is on a continuous (CG) space, where per-cell
            frames are ambiguous at shared nodes, or if a global-frame field is
            given anything but a full ``(3, 3)`` frame.

        See Also
        --------
        to_global : re-express in world x/y/z.
        """
        if self.kind == "scalar":
            return self
        if self.global_frame:
            return self._to_frame_global(target)
        if self.frame is None:
            return self
        if not self.is_dg:
            raise ValueError(
                "to_frame on a per-cell (local-frame) field needs a discontinuous "
                "(DG) space -- a CG field has ambiguous per-cell frames at shared "
                "nodes. Build a global-frame field (global_frame=True) for a "
                "CG-safe Cartesian field.")
        nc = self._tab.n_cells
        d = np.asarray(target, dtype=float)
        if d.ndim == 1:                       # global direction
            dir0 = np.broadcast_to(d, (nc, 3))
        elif d.shape == (3, 3):               # global frame
            dir0 = np.broadcast_to(d[0], (nc, 3))
        else:                                 # per-cell, FILE cell order -> FE order
            d = d[self.domain.cell_input_idx]
            dir0 = d[:, 0] if d.ndim == 3 else d
        new_frame = _reframe_in_plane(self.frame, dir0)               # rotate e0, keep e2
        T = _component_transform(self.kind, self.frame, new_frame)    # (n_cells, bs, bs)
        # one T block per scalar dof (DG0: dof d <-> cell d, unchanged)
        blocks = [sp.csr_matrix(T[c]) for c in self._tab.dof_cell]
        big = sp.block_diag(blocks, format="csr")
        return Field(self.domain, self.space, csdl.sparse.matvec(big, self.coeffs),
                    kind=self.kind, frame=new_frame)

    def _to_frame_global(self, target) -> "Field":
        R = np.asarray(target, dtype=float)
        if R.shape != (3, 3):
            raise ValueError(
                "to_frame on a global-frame field takes a full (3, 3) Cartesian "
                "frame (a bare direction does not define a 3D rotation); use "
                "to_global() for world x/y/z.")
        rel = R @ self.frame.T                    # current -> target
        M = _global_component_transform(self.kind, rel)          # (bs, bs)
        big = sp.kron(sp.eye(self._tab.n_scalar_dofs, format="csr"),
                      sp.csr_matrix(M), format="csr")
        return Field(self.domain, self.space, csdl.sparse.matvec(big, self.coeffs),
                    kind=self.kind, frame=R, global_frame=True)

    def to_global(self) -> "Field":
        """Re-express the field in the world x/y/z frame.

        Returns
        -------
        Field
            Equivalent to ``to_frame(np.eye(3))``. A scalar or frame-free field is
            returned unchanged.
        """
        return self.to_frame(np.eye(3))

    # -- elementwise arithmetic -----------------------------------------
    # A ``Field`` combines with another ``Field`` on the same space, or with a plain
    # scalar (python / numpy / a size-1 ``csdl.Variable``). Note that a ``csdl.Variable``
    # must go on the **right**: ``fld * var`` works, ``var * fld`` raises from inside
    # csdl_alpha, whose ``__mul__`` raises ``TypeError`` on an unknown operand instead
    # of returning ``NotImplemented``, so Python never falls back to ``__rmul__`` here.
    # (Plain floats are fine either way -- ``2.0 * fld`` does reach ``__rmul__``.)
    def __add__(self, other):
        if isinstance(other, Field):
            return field_fn(self, other, fn=lambda a, b: a + b)
        return self._scalar_op(other, lambda a, s: a + s)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Field):
            return field_fn(self, other, fn=lambda a, b: a - b)
        return self._scalar_op(other, lambda a, s: a - s)

    def __rsub__(self, other):
        return self._scalar_op(other, lambda a, s: s - a)

    def __mul__(self, other):
        if isinstance(other, Field):
            return field_fn(self, other, fn=lambda a, b: a * b)
        return self._scalar_op(other, lambda a, s: a * s)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Field):
            return field_fn(self, other, fn=lambda a, b: a / b)
        return self._scalar_op(other, lambda a, s: a / s)

    def __rtruediv__(self, other):
        return self._scalar_op(other, lambda a, s: s / a)

    def __neg__(self):
        return Field(self.domain, self.space, -self.coeffs, kind=self.kind,
                    frame=self.frame, global_frame=self.global_frame)

    def _scalar_op(self, other, op):
        if isinstance(other, csdl.Variable):
            if int(np.prod(other.shape)) != 1:
                return NotImplemented
        elif isinstance(other, np.ndarray):
            if other.size != 1:
                return NotImplemented
            other = float(other.reshape(-1)[0])
        elif isinstance(other, (int, float, np.floating, np.integer)):
            other = float(other)
        else:
            return NotImplemented
        coeffs = op(self.coeffs, other)
        return Field(self.domain, self.space, coeffs, kind=self.kind,
                    frame=self.frame, global_frame=self.global_frame)


def field_fn(*args, fn=None) -> Field:
    """Apply a function to the coefficients of one or more fields.

    Parameters
    ----------
    *args : Field
        Operands, all on the same domain and the same space. ``fn`` may be passed
        as the last positional argument instead of as a keyword.
    fn : callable
        Applied to the operands' flat ``coeffs``, one argument per field.

    Returns
    -------
    Field
        On the shared space. ``kind`` / ``frame`` / ``global_frame`` carry through
        only when every operand agrees, otherwise the result is scalar and
        frame-free.

    Raises
    ------
    ValueError
        If no operand is given, or if the operands are on different domains or
        different spaces. Use :func:`~hermit.interpolate` or
        :func:`~hermit.project` to reconcile spaces first; there is no implicit
        coercion.
    TypeError
        If an operand is not a ``Field``.

    Examples
    --------
    >>> total = field_fn(a, b, fn=lambda x, y: x + y)
    """
    if fn is None:
        *fields, fn = args
    else:
        fields = list(args)
    if not fields:
        raise ValueError("field_fn needs at least one Field")
    for f in fields:
        if not isinstance(f, Field):
            raise TypeError(f"field_fn operands must be Field, got {type(f)!r}")
    base = fields[0]
    for f in fields[1:]:
        if f.domain is not base.domain:
            raise ValueError("field_fn operands must be on the same ShellDomain")
        if f.space != base.space:
            raise ValueError(
                f"field_fn operands are on different spaces ({base.space} vs "
                f"{f.space}) -- interpolate() / project() them onto a common space "
                f"first; no implicit coercion.")
    coeffs = fn(*[f.coeffs for f in fields])
    kind = base.kind if all(f.kind == base.kind for f in fields) else "scalar"
    same_frame = all(_same_frame(f, base) for f in fields)
    frame = base.frame if same_frame else None
    global_frame = base.global_frame if same_frame else False
    return Field(base.domain, base.space, coeffs, kind=kind, frame=frame,
                global_frame=global_frame)


def _same_frame(a, b):
    if a.global_frame != b.global_frame:
        return False
    if a.frame is None and b.frame is None:
        return True
    if a.frame is None or b.frame is None:
        return False
    return a.frame.shape == b.frame.shape and np.array_equal(a.frame, b.frame)


# -- builders ---------------------------------------------------------------

def from_coeffs(domain, space, coeffs, *, kind="scalar", frame=None, global_frame=False) -> Field:
    """Build a field from raw coefficients in FE dof order.

    The explicit way to say "these are FE dof coefficients"; pair it with
    :meth:`~hermit.ShellDomain.dof_coords` to know where each dof lives.

    Parameters
    ----------
    domain : ShellDomain
    space : tuple
        ``(family, degree)`` or ``(family, degree, value_shape)``.
    coeffs : csdl.Variable or array_like
        FE dof order, flat or ``(n_scalar_dofs, *value_shape)``.
    kind, frame, global_frame
        Orientation bookkeeping; see :class:`Field`.

    Returns
    -------
    Field
    """
    return Field(domain, space, coeffs, kind=kind, frame=frame, global_frame=global_frame)


def _broadcast_value_to_dofs(value, n, value_shape):
    """Broadcast a scalar, or a ``value_shape``-shaped value, to ``(n, *value_shape)``
    (flattened by the caller) -- differentiable when ``value`` is a ``csdl.Variable``."""
    var = value if isinstance(value, csdl.Variable) else csdl.Variable(value=np.atleast_1d(np.asarray(value, dtype=float)))
    size = int(np.prod(var.shape)) if var.shape else 1
    bs = int(np.prod(value_shape)) if value_shape else 1
    out_shape = (n, *value_shape) if value_shape else (n,)
    if size == 1:
        return csdl.expand(var, out_shape)
    if size == bs:
        if not value_shape:
            return csdl.expand(var, out_shape)
        block = csdl.reshape(var, value_shape)
        src = "".join(chr(ord("i") + k) for k in range(len(value_shape)))
        return csdl.expand(block, out_shape, action=f"{src}->a{src}")
    raise ValueError(
        f"constant() value has {size} entries; expected 1 (broadcast) or {bs} "
        f"(the space's block size for value_shape {value_shape})")


def constant(domain, space, value) -> Field:
    """Build a uniform field by broadcasting one value to every dof.

    Parameters
    ----------
    domain : ShellDomain
    space : tuple
        ``(family, degree)`` or ``(family, degree, value_shape)``.
    value : float, array_like or csdl.Variable
        A scalar, or a value of the space's ``value_shape``. Stays differentiable
        when it is a ``csdl.Variable``.

    Returns
    -------
    Field
    """
    sp_ = normalize_space(space)
    tab = _tabulation(domain, sp_)
    block = _broadcast_value_to_dofs(value, tab.n_scalar_dofs, sp_[2])
    coeffs = csdl.reshape(block, (tab.n_dofs,))
    return Field(domain, sp_, coeffs)


def from_function(domain, space, fn) -> Field:
    """Build a field by evaluating a function at the space's dof coordinates.

    Parameters
    ----------
    domain : ShellDomain
    space : tuple
        ``(family, degree)`` or ``(family, degree, value_shape)``.
    fn : callable
        Called once with the ``(n_scalar_dofs, 3)`` dof coordinates and returning
        the coefficients. Returning numpy gives a constant field; returning a
        ``csdl.Variable`` (e.g. closing over a design variable) is passed through
        undisturbed and stays differentiable.

    Returns
    -------
    Field

    Notes
    -----
    The sample points are the *reference* dof coordinates, so the result is not
    differentiable with respect to the geometry.

    Examples
    --------
    >>> taper = from_function(domain, ("Lagrange", 1), lambda x: 0.2 - 0.01 * x[:, 0])
    """
    sp_ = normalize_space(space)
    xyz = domain.dof_coords(sp_)
    return Field(domain, sp_, fn(xyz))


def from_nodal(domain, values) -> Field:
    """Build a CG1 field from per-vertex values in mesh-file order.

    Parameters
    ----------
    domain : ShellDomain
    values : csdl.Variable or array_like
        Shape ``(n_nodes,)`` or ``(n_nodes, k)``, in **file** vertex order (the
        ordering an external geometry or CADDEE-side array uses).

    Returns
    -------
    Field
        On ``("Lagrange", 1)`` or ``("Lagrange", 1, (k,))``. The file-to-FE
        permutation is applied as a CSDL gather, so a ``csdl.Variable`` input stays
        differentiable.

    Raises
    ------
    ValueError
        If the leading axis is not ``domain.n_nodes``.

    See Also
    --------
    from_cells : the per-cell (DG0) counterpart.
    from_coeffs : when the array is already in FE dof order.
    """
    is_var = isinstance(values, csdl.Variable)
    shape = tuple(values.shape) if is_var else np.asarray(values).shape
    if not shape or shape[0] != domain.n_nodes:
        raise ValueError(
            f"from_nodal expects a leading axis of length domain.n_nodes "
            f"({domain.n_nodes}), got shape {tuple(shape)}")
    value_shape = tuple(shape[1:])
    space = ("Lagrange", 1, value_shape) if value_shape else ("Lagrange", 1)
    fe_order = values[list(domain.node_input_idx)] if is_var else np.asarray(values)[domain.node_input_idx]
    return Field(domain, space, fe_order)


def from_cells(domain, values) -> Field:
    """Build a DG0 field from per-cell values in mesh-file order.

    Parameters
    ----------
    domain : ShellDomain
    values : csdl.Variable or array_like
        Shape ``(n_cells,)`` or ``(n_cells, k)``, in **file** cell order.

    Returns
    -------
    Field
        On ``("DG", 0)`` or ``("DG", 0, (k,))``. The file-to-FE permutation is
        applied as a CSDL gather, so a ``csdl.Variable`` input stays
        differentiable.

    Raises
    ------
    ValueError
        If the leading axis is not ``domain.n_cells``.

    See Also
    --------
    from_nodal : the per-vertex (CG1) counterpart.
    """
    is_var = isinstance(values, csdl.Variable)
    shape = tuple(values.shape) if is_var else np.asarray(values).shape
    if not shape or shape[0] != domain.n_cells:
        raise ValueError(
            f"from_cells expects a leading axis of length domain.n_cells "
            f"({domain.n_cells}), got shape {tuple(shape)}")
    value_shape = tuple(shape[1:])
    space = ("DG", 0, value_shape) if value_shape else ("DG", 0)
    fe_order = values[list(domain.cell_input_idx)] if is_var else np.asarray(values)[domain.cell_input_idx]
    return Field(domain, space, fe_order)


def as_field_user_order(domain, value, default_space) -> Field:
    """``as_field`` for an API that has a *default* space but must not let that space
    silently change the caller's **ordering** convention.

    ``as_field(..., space=S)`` reads an array whose leading axis matches ``S``'s
    scalar-dof count as **FE dof order** (see its resolution order). But a bare
    per-vertex / per-cell array handed to ``hm.traction`` / ``hm.pressure`` /
    ``hm.fiber_direction`` is the CADDEE case: **file order**, which is what
    ``from_nodal`` / ``from_cells`` apply the file->FE permutation for. Passing a
    default space unconditionally selects the
    FE-order branch and skips that permutation -- silently, since the two orders
    coincide only when the mesh permutation is the identity.

    So: resolve by length first (file order) whenever the leading axis is
    ``n_nodes`` or ``n_cells``, and fall back to the explicit default space
    otherwise (e.g. a CG2-length array, which has no per-vertex meaning and *is*
    FE dof order). A caller who genuinely wants FE dof order builds the ``Field``
    with ``hm.from_coeffs(domain, space, coeffs)``, which is the documented way.
    """
    if isinstance(value, Field):
        return value
    shape = tuple(value.shape) if isinstance(value, csdl.Variable) else np.asarray(value).shape
    leading = shape[0] if shape else 0
    if leading in (domain.n_nodes, domain.n_cells):
        return as_field(domain, value)
    return as_field(domain, value, space=default_space)


def as_field(domain, value, *, space=None) -> Field:
    """Coerce a scalar, array or field to a :class:`Field`.

    This is how the rest of the API accepts a bare number or array wherever a
    ``Field`` is expected.

    Parameters
    ----------
    domain : ShellDomain
    value : Field, float, array_like or csdl.Variable
        The value to resolve.
    space : tuple, optional
        Space to use for a broadcast scalar, and to test an array's leading axis
        against before falling back to the per-vertex / per-cell orderings.

    Returns
    -------
    Field

    Raises
    ------
    ValueError
        If an array's leading axis matches neither ``domain.n_nodes``,
        ``domain.n_cells``, nor the given space's scalar-dof count.

    Notes
    -----
    Resolution order:

    1. A ``Field`` is returned unchanged (``space`` is ignored -- reconcile a
       mismatch yourself with :func:`~hermit.interpolate` or
       :func:`~hermit.project`).
    2. A scalar, or a length-1 array / ``csdl.Variable``, is broadcast with
       :func:`constant` on ``space`` (default ``("Lagrange", 1)``).
    3. An array with a leading axis longer than 1 is read as FE dof order when it
       matches ``space``'s scalar-dof count, else as file order via
       :func:`from_nodal` (``n_nodes``) or :func:`from_cells` (``n_cells``).
    """
    if isinstance(value, Field):
        return value
    is_var = isinstance(value, csdl.Variable)
    shape = tuple(value.shape) if is_var else np.asarray(value).shape
    total = int(np.prod(shape)) if shape else 1

    if total == 1:
        return constant(domain, space if space is not None else ("Lagrange", 1), value)

    leading = shape[0]
    if space is not None:
        sp_ = normalize_space(space)
        n_scalar = _tabulation(domain, sp_).n_scalar_dofs
        if leading == n_scalar:
            return from_coeffs(domain, sp_, value)

    if leading == domain.n_nodes:
        return from_nodal(domain, value)
    if leading == domain.n_cells:
        return from_cells(domain, value)

    raise ValueError(
        f"as_field: cannot resolve a value with leading axis {leading} to a Field -- "
        f"it matches neither domain.n_nodes ({domain.n_nodes}), domain.n_cells "
        f"({domain.n_cells}), nor the given space's scalar-dof count. Pass an "
        f"explicit Field, or an explicit space= whose scalar-dof count matches.")


# -- frame-change component transforms -----------------------------------------

def _reframe_in_plane(frame, direction):
    """A frame with the same normal e2 as ``frame`` but e0 aligned with the tangent-plane
    projection of ``direction``. ``frame``: (n, 3, 3); ``direction``: (n, 3)."""
    e2 = frame[:, 2]
    d = np.asarray(direction, dtype=float)
    dt = d - np.einsum("ij,ij->i", d, e2)[:, None] * e2
    e0 = dt / np.linalg.norm(dt, axis=1, keepdims=True)
    e1 = np.cross(e2, e0)
    return np.stack([e0, e1, e2], axis=1)


def _inplane_angle(source, target):
    """Rotation angle (per cell) from ``source`` in-plane axes to ``target`` -- both
    frames share e2, so this is an exact planar rotation. (n, 3, 3) each."""
    e0, e1 = source[:, 0], source[:, 1]
    g0 = target[:, 0]
    return np.arctan2(np.einsum("ij,ij->i", g0, e1), np.einsum("ij,ij->i", g0, e0))


def _global_component_transform(kind, R):
    """Component transform for a *global* (Cartesian) frame-relative field under the
    rotation ``R`` (current -> target, a 3x3 orthogonal matrix)."""
    if kind == "vector3":
        return np.asarray(R, dtype=float)
    if kind == "tensor3":  # engineering-Voigt symmetric-tensor rotation
        M = np.empty((6, 6))
        for a, (i, j) in enumerate(_V6):
            for b, (k, ll) in enumerate(_V6):
                v = R[i, k] * R[j, ll]
                if k != ll:
                    v += R[i, ll] * R[j, k]
                M[a, b] = v
        return M * (_V6_ENG[:, None] / _V6_ENG[None, :])
    raise ValueError(f"global frame change not defined for kind {kind!r}")


def _component_transform(kind, local, target):
    theta = _inplane_angle(local, target)
    c, s = np.cos(theta), np.sin(theta)
    n = theta.shape[0]
    if kind == "shear2":
        T = np.zeros((n, 2, 2))
        T[:, 0, 0] = c; T[:, 0, 1] = s
        T[:, 1, 0] = -s; T[:, 1, 1] = c
        return T
    if kind == "strain2":  # engineering-strain Voigt transform (== lamad.calc_t_eps)
        T = np.zeros((n, 3, 3))
        T[:, 0, 0] = c**2; T[:, 0, 1] = s**2; T[:, 0, 2] = s * c
        T[:, 1, 0] = s**2; T[:, 1, 1] = c**2; T[:, 1, 2] = -s * c
        T[:, 2, 0] = -2 * s * c; T[:, 2, 1] = 2 * s * c; T[:, 2, 2] = c**2 - s**2
        return T
    if kind == "vector3":  # full 3x3 rotation local -> target
        return np.einsum("nik,njk->nij", target, local)
    raise ValueError(kind)  # pragma: no cover
