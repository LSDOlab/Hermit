"""``interpolate`` / ``project`` -- transfer a :class:`hermit.Field` between spaces
on the same mesh.

``interpolate`` is collocation: the source basis evaluated at the target dofs'
*reference* coordinates, a constant sparse matrix and therefore a plain
``csdl.sparse.matvec``. It is fully differentiable in the coefficients and
independent of the geometry. This holds for identity-mapped, point-evaluation
elements -- Lagrange, DG, DQ and Quadrature, which covers every space these
builders produce. Piola-mapped families such as RT and Nedelec are rejected, as is
the Crouzeix-Raviart rotation space of the ``CG2CR1`` element.

``project`` is the L2 alternative, ``M c = int(phi_target . src) dx``. Both sides
carry ``dx``, so it depends on the geometry: it is a custom operation with the
mass-matrix quotient-rule VJP, reverse mode only. Its optional ``geometry=``
argument defaults to the domain's reference coordinates, which keeps the mass
factorization cached; passing a live geometry enables the shape-derivative branch.
"""

from __future__ import annotations

import weakref

import numpy as np
import scipy.sparse as sp
import ufl
import csdl_alpha as csdl
from dolfinx.fem import Expression, Function, functionspace

from . import _compat
from .fenics import assembly as fa
from .fenics.spaces import normalize_space
from ._field import Field

# families whose interpolation is a plain point evaluation (dofs are values of the
# basis, unaffected by the Piola map that RT / Nedelec elements need)
_POINT_EVAL_FAMILIES = {"Lagrange", "DG", "DQ", "Quadrature"}


def _check_point_eval_family(space, which):
    family = space[0]
    if family not in _POINT_EVAL_FAMILIES:
        raise ValueError(
            f"interpolate(): the {which} space family {family!r} is not "
            f"identity-mapped / point-evaluation -- this holds for Lagrange / DG / "
            f"Quadrature elements (everything Hermit uses) and not for Piola-mapped "
            f"families (RT, Nedelec, ...), whose interpolation drags in the "
            f"geometric Jacobian.")


def _basix_element_or_none(V):
    """``None`` for a Quadrature space -- it has no plain basix C++ element (dofs are
    the quadrature-point values themselves, no basis function to tabulate). Mirrors
    ``hermit._field._Tabulation.__init__``."""
    try:
        return V.element.basix_element
    except RuntimeError:
        return None


# -- interpolate: constant sparse matrix, memoized per (domain, src space, tgt space) --

_INTERP_CACHE = weakref.WeakKeyDictionary()   # domain -> {(src_space, tgt_space): csr}


def _interp_matrix(domain, src_space, tgt_space):
    per_domain = _INTERP_CACHE.get(domain)
    if per_domain is None:
        per_domain = {}
        _INTERP_CACHE[domain] = per_domain
    key = (src_space, tgt_space)
    P = per_domain.get(key)
    if P is None:
        P = _build_interp_matrix(domain, src_space, tgt_space)
        per_domain[key] = P
    return P


def _build_interp_matrix(domain, src_space, tgt_space):
    _check_point_eval_family(src_space, "source")
    _check_point_eval_family(tgt_space, "target")

    V_src = domain.function_space(src_space)
    if _basix_element_or_none(V_src) is None:
        raise ValueError(
            "interpolate(): the source space is Quadrature -- its dofs are the "
            "quadrature-point values themselves, with no basis function to "
            "tabulate at an arbitrary point, so there is no interpolation matrix. "
            "Use project() instead (it consumes the source as an ordinary UFL "
            "coefficient, which quadrature elements support).")
    V_tgt = domain.function_space(tgt_space)

    # phi_source, evaluated at the target's reference interpolation points: exactly
    # ShellFieldFormsOp._interp_jac's construction, with the arbitrary "field
    # expression" specialised to the source Function itself (so its derivative is
    # the source basis).
    src_fn = Function(V_src)
    pts = _compat.interpolation_points(V_tgt)
    trial = ufl.TrialFunction(V_src)
    expr = Expression(ufl.derivative(src_fn, src_fn, trial), pts)

    nel = domain.n_cells
    vj = np.asarray(expr.eval(domain.mesh, np.arange(nel, dtype=np.int32)))

    nn = np.asarray(V_tgt.dofmap.list)                     # (nel, n_local_tgt_nodes)
    npts = nn.shape[1]
    nc = V_tgt.dofmap.bs
    vj = vj.reshape(nel, npts, nc, -1)                      # (nel, npts, nc, n_src_local)

    acols = np.asarray(V_src.dofmap.list)
    abs_ = V_src.dofmap.bs
    if abs_ > 1:
        acols = (acols[:, :, None] * abs_ + np.arange(abs_)).reshape(nel, -1)

    rows = np.broadcast_to(nn[:, :, None, None] * nc + np.arange(nc)[:, None], vj.shape)
    cols = np.broadcast_to(acols[:, None, None, :], vj.shape)

    n_tgt_dofs = V_tgt.dofmap.index_map.size_local * nc
    n_src_dofs = V_src.dofmap.index_map.size_local * abs_
    # only the owning cell's row per target dof -- see fa.owned_entry_mask
    keep = np.broadcast_to(
        fa.owned_entry_mask(nn, V_tgt.dofmap.index_map.size_local)[:, :, None, None],
        vj.shape)
    return sp.coo_matrix((vj[keep], (rows[keep], cols[keep])),
                         shape=(n_tgt_dofs, n_src_dofs)).tocsr()


def interpolate(src_field, space) -> Field:
    """Collocate a field at another space's dof points -- the default transfer.

    Parameters
    ----------
    src_field : Field
    space : tuple
        Target space; its value shape must match the source's.

    Returns
    -------
    Field
        On the target space, or ``src_field`` itself if already there.

    Raises
    ------
    ValueError
        If the value shapes differ, or if either space is not point-evaluation
        (Lagrange, DG, DQ or Quadrature). Piola-mapped families such as RT and
        Nedelec drag in the geometric Jacobian and are rejected, as does
        Crouzeix-Raviart. A Quadrature *source* has no basis to tabulate -- use
        :func:`project`.

    Notes
    -----
    A constant sparse matrix, memoized per source/target pair, so this is a plain
    ``csdl.sparse.matvec``: fully differentiable in the coefficients and
    independent of the geometry, because the target dofs' reference coordinates
    never move.

    See Also
    --------
    project : the geometry-dependent L2 alternative.
    """
    domain = src_field.domain
    tgt_space = normalize_space(space)
    src_space = src_field.space
    if src_space[2] != tgt_space[2]:
        raise ValueError(
            f"interpolate(): source value shape {src_space[2]} != target value "
            f"shape {tgt_space[2]}")
    if src_space == tgt_space:
        return src_field   # P would be the identity
    P = _interp_matrix(domain, src_space, tgt_space)
    coeffs = csdl.sparse.matvec(P, src_field.coeffs)
    return Field(domain, tgt_space, coeffs, kind=src_field.kind, frame=src_field.frame,
                global_frame=src_field.global_frame)


# -- project: L2 projection, geometry-dependent -----------------------------------

def _resolve_geometry(domain, geometry):
    """``geometry=None`` -> the domain's own reference node coordinates (file
    order), not a live CSDL input -- the mass factorization is cached across calls.
    Otherwise a structural stand-in for the real ``Geometry``: any object with a
    ``.nodes`` attribute holding an ``(n_nodes, 3)`` file-order
    ``csdl.Variable``, or a bare ``(n_nodes, 3)`` array / ``csdl.Variable`` -- always
    treated as a live (differentiable) input, the same "explicit geometry ->
    declared mesh_nodes input" convention ``hermit.csdl_helpers.resolve_mesh_nodes``
    already uses.
    """
    if geometry is None:
        return csdl.Variable(value=domain.node_coords), False
    geom_domain = getattr(geometry, "domain", None)
    if geom_domain is not None and geom_domain is not domain:
        raise ValueError(
            "project(): geometry belongs to a different ShellDomain than "
            "src_field -- interpolate() / project() are strictly same-mesh.")
    nodes = getattr(geometry, "nodes", geometry)
    if not isinstance(nodes, csdl.Variable):
        nodes = csdl.Variable(value=np.asarray(nodes, dtype=float))
    return nodes, True


def _quadrature_degree(src_space, tgt_space):
    """The shared ``quadrature_degree`` metadata every measure touching a Quadrature
    coefficient must carry (dolfinx errors otherwise) -- ``None`` if neither space is
    Quadrature; raises if both are Quadrature at different degrees."""
    degs = {sp[1] for sp in (src_space, tgt_space) if sp[0] == "Quadrature"}
    if not degs:
        return None
    if len(degs) > 1:
        raise ValueError(
            f"project(): source and target ask for different quadrature degrees "
            f"{sorted(degs)} -- pick one space's degree for both, or project "
            f"through an intermediate space.")
    return degs.pop()


def _mat_tvec(A, x):
    """``A^T @ x`` for an assembled PETSc Mat and a numpy vector."""
    v, y = A.createVecLeft(), A.createVecRight()
    v.setArray(np.asarray(x)); v.assemble()
    A.multTranspose(v, y)
    return y.getArray().copy()


class _ProjectOp(csdl.CustomExplicitOperation):
    """``M c = int(phi_target @ src) dx`` -- L2 projection of one field onto another
    space, on the same domain. Reverse-mode only (one cotangent solve per call, never
    a dense Jacobian); port of ``ShellFieldFormsOp``'s ``project`` branch (forward
    solve, mass-matrix quotient-rule VJP), run on the input side. Kept private --
    ``project()`` is the public entry point.
    """

    def __init__(self, domain, src_space, tgt_space, differentiable_geometry):
        super().__init__()
        self.domain = domain
        self.mesh = domain.mesh
        self.gdim = domain.mesh.geometry.dim
        self._diff_geom = differentiable_geometry
        self._const_mesh_nodes = None
        # mesh.geometry.x is a scratch buffer -- callers must restore it (see
        # ShellPDE.set_geometry / restore_geometry, which this mirrors without
        # depending on ShellPDE).
        self._x_ref = domain.mesh.geometry.x.copy()

        V_src = domain.function_space(src_space)
        V_tgt = domain.function_space(tgt_space)
        self.n_tgt_dofs = V_tgt.dofmap.index_map.size_local * V_tgt.dofmap.bs

        self._src_fn = Function(V_src)
        self._c_fn = Function(V_tgt)   # holder for the projected coeffs (VJP quotient rule)

        qdeg = _quadrature_degree(src_space, tgt_space)
        dxm = ufl.dx if qdeg is None else ufl.dx(metadata={"quadrature_degree": qdeg})
        v = ufl.TestFunction(V_tgt)
        self._L = ufl.inner(self._src_fn, v) * dxm
        self._Mform = ufl.inner(ufl.TrialFunction(V_tgt), v) * dxm

        if differentiable_geometry:
            self.X = ufl.SpatialCoordinate(domain.mesh)
            self.Vc = functionspace(domain.mesh, domain.mesh.ufl_domain().ufl_coordinate_element())

        self._ksp = None   # cached mass factorization (constant geometry only)

    # -- graph wiring --------------------------------------------------
    def evaluate(self, src_coeffs, mesh_nodes):
        self.declare_input("src", src_coeffs)
        if self._diff_geom:
            self.declare_input("mesh_nodes", mesh_nodes)
        else:
            self._const_mesh_nodes = np.asarray(mesh_nodes.value)
        out = self.create_output("coeffs", (self.n_tgt_dofs,))
        self.declare_derivative_parameters("coeffs", "src", dependent=True)
        if self._diff_geom:
            self.declare_derivative_parameters("coeffs", "mesh_nodes", dependent=True)
        return out

    # -- geometry scratch buffer ----------------------------------------
    def _set_geometry(self, mesh_nodes):
        self.mesh.geometry.x[:] = (
            np.asarray(mesh_nodes).reshape(-1, self.gdim)[self.domain.node_input_idx])

    def _restore_geometry(self):
        self.mesh.geometry.x[:] = self._x_ref

    def _push(self, input_vals):
        mn = input_vals["mesh_nodes"] if self._diff_geom else self._const_mesh_nodes
        self._set_geometry(mn)
        fa.set_array(self._src_fn, input_vals["src"])

    def _mass_ksp(self):
        if not self._diff_geom and self._ksp is not None:
            return self._ksp
        ksp = fa.ksp_mumps(fa.assemble_matrix(self._Mform))
        if not self._diff_geom:
            self._ksp = ksp
        return ksp

    @staticmethod
    def _msolve(ksp, rhs):
        M = ksp.getOperators()[0]
        b, x = M.createVecRight(), M.createVecRight()
        b.setArray(np.asarray(rhs)); b.assemble()
        ksp.solve(b, x)
        return x.getArray().copy()

    # -- forward ---------------------------------------------------------
    def compute(self, input_vals, output_vals):
        self._push(input_vals)
        try:
            b = fa.assemble_vector(self._L)
            output_vals["coeffs"] = self._msolve(self._mass_ksp(), b)
        finally:
            self._restore_geometry()

    # -- reverse (VJP) -----------------------------------------------
    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        assert mode == "rev"
        self._push(inputs)
        try:
            if "coeffs" not in d_outputs:
                return
            lam = np.asarray(d_outputs["coeffs"]).reshape(-1)
            if not lam.any():
                return
            lam_t = self._msolve(self._mass_ksp(), lam)   # M^-1 lambda (M symmetric)

            if "src" in d_inputs:
                dLda = fa.assemble_matrix(ufl.derivative(self._L, self._src_fn))
                d_inputs["src"] += _mat_tvec(dLda, lam_t).reshape(d_inputs["src"].shape)

            if self._diff_geom and "mesh_nodes" in d_inputs:
                tc = ufl.TrialFunction(self.Vc)
                g = _mat_tvec(fa.assemble_matrix(ufl.derivative(self._L, self.X, tc)), lam_t)
                fa.set_array(self._c_fn, np.asarray(outputs["coeffs"]).reshape(-1))
                dmc = fa.assemble_matrix(
                    ufl.derivative(ufl.action(self._Mform, self._c_fn), self.X, tc))
                g = g - _mat_tvec(dmc, lam_t)               # - d(M c)/dX^T lam_t
                ext = np.zeros((self.domain.node_input_idx.size, self.gdim))
                np.add.at(ext, self.domain.node_input_idx, g.reshape(-1, self.gdim))
                d_inputs["mesh_nodes"] += ext.reshape(d_inputs["mesh_nodes"].shape)
        finally:
            self._restore_geometry()


def project(src_field, space, *, geometry=None) -> Field:
    """L2-project a field onto another space, ``M c = int(phi_target . src) dx``.

    Parameters
    ----------
    src_field : Field
    space : tuple
        Target space; its value shape must match the source's.
    geometry : Geometry, ndarray or csdl.Variable, optional
        Defaults to the domain's reference coordinates, which lets the mass-matrix
        factorization be cached across calls. Passing a live geometry enables the
        shape-derivative branch.

    Returns
    -------
    Field
        On the target space. Projecting onto the same space is the identity up to
        solver round-off.

    Raises
    ------
    ValueError
        If the value shapes differ, if ``geometry`` belongs to another domain, or
        if the source and target Quadrature degrees disagree.

    Notes
    -----
    Both sides carry ``dx``, so unlike :func:`interpolate` this depends on the
    geometry. It is a custom operation with a mass-matrix quotient-rule VJP,
    reverse mode only: one cotangent solve per call, never a dense Jacobian.
    """
    domain = src_field.domain
    src_space = src_field.space
    tgt_space = normalize_space(space)
    if src_space[2] != tgt_space[2]:
        raise ValueError(
            f"project(): source value shape {src_space[2]} != target value shape "
            f"{tgt_space[2]}")
    mesh_nodes, diff_geom = _resolve_geometry(domain, geometry)
    op = _ProjectOp(domain, src_space, tgt_space, diff_geom)
    coeffs = op.evaluate(src_field.coeffs, mesh_nodes)
    return Field(domain, tgt_space, coeffs, kind=src_field.kind, frame=src_field.frame,
                global_frame=src_field.global_frame)
