"""The FEniCSx custom operations. Called only from ``hermit._solve`` /
``hermit.outputs`` / ``hermit.transfer``.

``ShellSolveOp``  -- implicit; assemble R/J, linear (or Newton) solve, adjoint.
``ShellScalarFormsOp`` (scalar output forms) -- explicit, dense Jacobians.
``ShellFieldFormsOp`` (field outputs -- one UFL expression per :class:`FieldSpec`
into its target space, by ``project`` / ``interpolate`` / ``average`` / ``midpoint``)
-- explicit, reverse-mode only (one cotangent solve or matvec per field, never a
dense Jacobian). ``project`` / ``average`` / ``midpoint`` keep mesh-coordinate
derivatives; ``interpolate`` does not.

Port of ``StateOperation`` / ``OutputOperation`` / ``OutputFieldOperation`` (femo
dev_coupling), self-contained (no ``FEA``). The shell tangent is symmetric, so one MUMPS
factorization of ``dR/dw`` serves both the forward solve and the adjoint.
"""

from dataclasses import dataclass, field as _dcfield, replace

import numpy as np
import scipy.sparse as sp
import ufl
import csdl_alpha as csdl
from dolfinx.fem import Expression, Function, functionspace
from dolfinx import mesh as dmesh

from .. import _compat
from . import assembly as fa
from .solvers import linear_solve
from .spaces import make_space, normalize_space

_FUNCTION_ARGS = ("A", "B", "D", "As", "thickness", "f", "m")
_PDE_ATTR = {"thickness": "h"}  # FE arg name -> ShellPDE attribute
_ORIENTATION_NAMES = ("fiber_angle", "fiber_direction")


def _share_edge_and_penalty_ds(pde, bc, loads):
    """Return form-local measures sharing one exterior-facet ``MeshTags`` object.

    DOLFINx rejects a form containing independently-created tagged ``ds`` measures.
    Boundary conditions already have one; edge loads add more.  A facet may belong
    to several terms, so tag each distinct membership set and make a term's measure
    the sum of all tags containing it.  This preserves overlapping load/penalty
    regions instead of silently choosing one owner.
    """
    edge_indices = [i for i, item in enumerate(loads or ()) if len(item) == 5]
    if not edge_indices:
        return bc, loads
    penalty_facets = list(bc.penalty_entities) if bc.penalty else []
    if bc.penalty and not penalty_facets:  # defensive for BCData made by an older/raw caller
        return bc, loads
    groups = [np.asarray(f, dtype=np.int32) for f in penalty_facets] + [
        np.asarray(loads[i][4], dtype=np.int32) for i in edge_indices
    ]
    membership = {}
    for group, facets in enumerate(groups):
        for facet in facets:
            membership.setdefault(int(facet), set()).add(group)
    by_membership = {}
    for facet, members in membership.items():
        by_membership.setdefault(tuple(sorted(members)), []).append(facet)
    entities, values, group_tags = [], [], [[] for _ in groups]
    for tag, (members, facets) in enumerate(by_membership.items(), start=1):
        entities.extend(facets); values.extend([tag] * len(facets))
        for group in members:
            group_tags[group].append(tag)
    order = np.argsort(entities)
    mt = dmesh.meshtags(pde.mesh, pde.mesh.topology.dim - 1,
                        np.asarray(entities, dtype=np.int32)[order],
                        np.asarray(values, dtype=np.int32)[order])
    base = ufl.Measure("ds", domain=pde.mesh, subdomain_data=mt,
                       metadata={"quadrature_degree": pde.quadrature_degree})
    def measure(tags):
        if not tags:
            # A locator is allowed to select no exterior facets; retain its
            # well-defined zero contribution instead of indexing an empty tag list.
            return base(-1)
        out = base(tags[0])
        for tag in tags[1:]:
            out = out + base(tag)
        return out
    shared = [measure(tags) for tags in group_tags]
    if bc.penalty_terms:
        terms = [(shared[i], dSS, mask) for i, (_, dSS, mask) in enumerate(bc.penalty_terms)]
        bc = replace(bc, penalty_terms=terms)
    elif bc.penalty:
        bc = replace(bc, dss=shared[0])
    loads = list(loads)
    for group, idx in enumerate(edge_indices, start=len(penalty_facets)):
        item = loads[idx]
        loads[idx] = (*item[:3], shared[group], item[4])
    return bc, loads


def _bc_structural_key(bc):
    """Structural signature of a ``BCData``'s penalty composition -- distinguishes "no
    penalty", a single ``(dss, dSS, mask)`` triple, and a multi-region
    ``penalty_terms`` list, by measure-object identity. A ``BoundaryConditions``
    compiles fresh ``dss``/``dSS`` measures per region, so this also separates two
    differently-located BCs on the same domain -- not just presence/kind."""
    if not bc.penalty:
        return (False,)
    if bc.penalty_terms:
        return (True, "multi") + tuple((id(d), id(D), m) for d, D, m in bc.penalty_terms)
    return (True, "single", id(bc.dss), id(bc.dSS), bc.bc_dof_mask)


def _orientation_key(orientation):
    """``None``, or ``(name, normalized_space)`` -- part of the structural form-cache
    key (see ``ShellSolveOp``)."""
    if orientation is None:
        return None
    name, space = orientation
    return (name, normalize_space(space))


def _material_key(material):
    """``None`` (fixed-space A/B/D/As), or a sorted ``(name, normalized_space)``
    tuple per coefficient -- part of the structural form-cache key. Two ``Material``\\s
    with the same kind/orientation but different ``constitutive_space`` values must never
    share a compiled residual."""
    if material is None:
        return None
    return tuple(sorted((n, normalize_space(sp)) for n, sp in material.items()))


def _loads_key(loads):
    """``None`` selects the fixed-``f``/``m`` residual path; otherwise an
    ordered tuple of ``(kind, normalized_space)`` per distributed term (``()`` for a
    point-load-only ``Loads`` with none) -- part of the structural form-cache key.
    ``None`` and ``()`` must stay distinct keys: they build genuinely different
    residuals (the always-present ``f``/``m`` VF coefficients vs. truly zero
    distributed terms), so collapsing them could reuse the wrong compiled form. The
    load *composition* (which kinds are present, and each one's space) changes the
    residual's structure, not just its coefficient values -- reusing a form across
    two different compositions would silently apply the wrong physics."""
    if loads is None:
        return None
    return tuple((item[1], normalize_space(item[2]), id(item[3]) if len(item) == 5 else None)
                 for item in loads)


class ShellSolveOp(csdl.experimental.CustomImplicitOperation):
    def __init__(self, pde, bc, arg_names, options, form_cache=None, orientation=None,
                material=None, loads=None):
        """``orientation``, given, is ``(name, space)`` with ``name`` in
        ``("fiber_angle", "fiber_direction")`` -- the in-form ``Teps(theta)`` rotation
        (see ``hermit.fenics.elastic_model.ElasticModel``). ``name`` must also appear
        in ``arg_names`` for it to be a declared (differentiable) CSDL input.

        ``material``, given, is a dict ``{"A": space, "B": space, "D": space, "As":
        space}`` (all four): each coefficient is routed through ``pde.coefficient(name,
        space)`` instead of the PDE's fixed ``VABD``/``VAs`` space, so an arbitrary
        caller-chosen ``constitutive_space`` reaches the solve. ``None`` (default)
        keeps the fixed-space Functions.

        ``loads``, given, is a list of ``(argname, kind, space)`` triples: ``kind`` in
        ``"traction"`` / ``"moment"`` / ``"pressure"``, each routed through
        ``pde.coefficient(argname, space)`` and contributing its own
        residual/compliance term (see ``ElasticModel.weak_residual`` /
        ``ShellPDE.compliance_form``'s ``load_terms``), rather than being reduced onto
        the fixed ``f``/``m`` VF coefficients. Every ``argname`` must also appear in
        ``arg_names``. ``None`` (default) keeps the fixed-VF ``f``/``m`` mechanism; an
        empty list selects the per-term path with zero distributed terms (e.g. a
        point-load-only ``Loads``).

        The residual is not structurally fixed once orientation, material spaces, load
        composition, and multi-region penalty terms can all vary -- ``form_cache`` is
        keyed by a structural signature of all four, not by a single fixed string, so
        two structurally different compositions on the same ``ShellDomain`` never
        share a (wrong) compiled residual.
        """
        super().__init__()
        self.pde = pde
        self.bc = bc
        self.options = options
        self.arg_names = tuple(arg_names)
        self.state_name = "disp_solid"
        self.ndof = pde.w.x.array.size
        self.orientation = orientation
        bc, loads = _share_edge_and_penalty_ds(pde, bc, loads)
        self.bc = bc

        # The common penalty form references ShellPDE.g.  BCData owns the static
        # prescribed Function, so copy it before each assembly (the PDE is cached and
        # may be shared by solves with different BC objects).
        self._penalty_target = bc.g

        # persistent argument Functions, by FE arg name
        self._funcs = {n: getattr(pde, _PDE_ATTR.get(n, n)) for n in _FUNCTION_ARGS}
        if material is not None:
            for n, sp in material.items():
                self._funcs[n] = pde.coefficient(n, sp)
        if orientation is not None:
            oname, ospace = orientation
            self._funcs[oname] = pde.coefficient(oname, ospace)
        for item in (loads or ()):
            argname, kind, space = item[:3]
            self._funcs[argname] = pde.coefficient(argname, space)

        # residual / tangent forms + coordinate space are arg-independent (for a given
        # structural key) and built on the PDE's persistent Functions -> memoize
        # across op instances
        cache = form_cache if form_cache is not None else {}
        key = (_bc_structural_key(bc), _orientation_key(orientation),
              _material_key(material), _loads_key(loads))
        if key not in cache:
            mat_funcs = None
            if material is not None:
                mat_funcs = tuple(self._funcs[n] for n in ("A", "B", "D", "As"))
            # loads=None keeps ShellPDE.residual_form's fixed f/m path (the default,
            # for raw-ops callers that never pass loads=); loads=[] or a populated
            # list selects the per-term path with that many distributed terms
            # (possibly zero -- a point-load-only Loads has none).
            load_terms = None if loads is None else [
                (item[1], self._funcs[item[0]]) if len(item) == 3
                else (item[1], self._funcs[item[0]], item[3])
                for item in loads
            ]
            residual = pde.residual_form(
                penalty=bc.penalty, dss=bc.dss, dSS=bc.dSS, bc_dof_mask=bc.bc_dof_mask,
                penalty_terms=bc.penalty_terms or None,
                penalty_targets=bc.penalty_targets or None,
                orientation=None if orientation is None else (orientation[0], self._funcs[orientation[0]]),
                material=mat_funcs, loads=load_terms,
            )
            cache[key] = dict(
                residual=residual,
                dR_dw=ufl.derivative(residual, pde.w),
                Vc=functionspace(pde.mesh, pde.mesh.ufl_domain().ufl_coordinate_element()),
                dR_dp={},
            )
        self._forms = cache[key]
        self.residual = self._forms["residual"]
        self.dR_dw = self._forms["dR_dw"]

        # dR/darg forms -- built lazily (the mesh-coord form is expensive and only
        # needed when someone differentiates w.r.t. mesh_nodes)
        self._kind = {}
        self._dR_form = self._forms["dR_dp"]  # lazy, shared: name -> UFL dR/d(arg) form
        for n in self.arg_names:
            if n in self._funcs:
                self._kind[n] = "function"
            elif n == "mesh_nodes":
                self._kind[n] = "mesh"
            elif n == "load_vector":
                self._kind[n] = "direct"
            else:  # pragma: no cover
                raise ValueError(f"unknown solve arg {n!r}")
        self.Vc = self._forms["Vc"]

        self._node_idx = np.asarray(pde.mesh.geometry.input_global_indices, dtype=np.int64)
        self.gdim = pde.mesh.geometry.dim
        self._diff_geometry = "mesh_nodes" in self.arg_names
        self._const_mesh_nodes = None  # captured in evaluate() when geometry isn't a live arg

        self._A = None       # assembled dR/dw (PETSc Mat)
        self._ksp = None
        self._dRdf = {}      # assembled dR/darg mats

    # -- graph wiring --------------------------------------------------
    def evaluate(self, fe):
        for n in self.arg_names:
            self.declare_input(n, getattr(fe, n))
        if not self._diff_geometry:
            self._const_mesh_nodes = np.asarray(fe.mesh_nodes.value)
        state = self.create_output(self.state_name, (self.ndof,))
        state.add_name(self.state_name)
        self.declare_derivative_parameters(self.state_name, "*", dependent=True)
        return state

    # -- write the FE-side inputs ------------------------------------
    def _push_inputs(self, inputs):
        """Set mesh geometry + argument Functions from ``inputs``; return direct-RHS list.

        ``mesh.geometry.x`` is a scratch buffer -- callers must ``restore_geometry()``
        (via ``_done()``) once they have finished assembling.
        """
        direct = []
        mn = inputs["mesh_nodes"] if self._diff_geometry else self._const_mesh_nodes
        self.pde.set_geometry(mn)
        if self._penalty_target is not None:
            fa.set_array(self.pde.g, fa.get_array(self._penalty_target))
        for n in self.arg_names:
            val = inputs[n]
            if n in self._funcs:   # A/B/D/As/thickness/f/m, or the orientation coefficient
                fa.set_array(self._funcs[n], val)
            elif n == "load_vector":
                direct.append({"value": np.asarray(val).reshape(-1), "sign": -1.0})
        return direct

    def _done(self):
        self.pde.restore_geometry()

    def solve_residual_equations(self, inputs, outputs):
        direct = self._push_inputs(inputs)
        try:
            self.pde.w.x.array[:] = 0.0
            linear_solve(self.residual, self.pde.w, bcs=self.bc.strong,
                         direct_residual_inputs=direct)
            outputs[self.state_name] = fa.get_array(self.pde.w).copy()
            self._assemble_derivatives(inputs, outputs)
        finally:
            self._done()

    def _dR_dp(self, name):
        """UFL form dR/d(name); built + cached on first use."""
        if name not in self._dR_form:
            kind = self._kind[name]
            if kind == "function":
                self._dR_form[name] = ufl.derivative(self.residual, self._funcs[name])
            elif kind == "mesh":
                trial = ufl.TrialFunction(self.Vc)
                self._dR_form[name] = ufl.derivative(self.residual, self.pde.X, trial)
            else:  # pragma: no cover
                raise ValueError(kind)
        return self._dR_form[name]

    def _assemble_dRdp(self, name):
        """Assemble dR/d(name); zero the strong-BC rows (residual there is u - g,
        which has no dependence on the parameter)."""
        M = fa.assemble_matrix(self._dR_dp(name))
        rows = self.bc.strong_dofs
        if rows.size:
            M.zeroRows(rows.astype("int32"), diag=0.0)
        return M

    # -- derivative assembly (called with geometry already set) -----------
    def _assemble_derivatives(self, inputs, outputs):
        fa.set_array(self.pde.w, outputs[self.state_name])
        self._A = fa.assemble_matrix(self.dR_dw, bcs=self.bc.strong)
        self._ksp = fa.ksp_mumps(self._A)
        self._dRdf = {n: self._assemble_dRdp(n)
                      for n, k in self._kind.items() if k in ("function", "mesh")}

    # -- adjoint / tangent solves (tangent is symmetric -> A^-1 == A^-T) --
    def apply_inverse_jacobian(self, inputs, outputs, d_outputs, d_residuals, mode):
        s = self.state_name
        src = d_residuals[s] if mode == "fwd" else d_outputs[s]
        rhs = self._A.createVecRight()
        rhs.setArray(np.asarray(src).reshape(-1))
        rhs.assemble()
        sol = self._A.createVecLeft()
        self._ksp.solve(rhs, sol)
        out = sol.getArray().copy()
        if mode == "fwd":
            d_outputs[s] = out
        else:
            d_residuals[s] = out

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        s = self.state_name
        self._push_inputs(inputs)
        fa.set_array(self.pde.w, outputs[self.state_name])
        try:
            if mode == "fwd":
                if s in d_outputs:
                    d_residuals[s] += self._matvec(self._A, d_outputs[s])
                for n in self.arg_names:
                    if n not in d_inputs:
                        continue
                    d_residuals[s] += self._dRdp_matvec(n, d_inputs[n], transpose=False)
            elif mode == "rev":
                lam = np.asarray(d_residuals[s])
                for n in self.arg_names:
                    if n not in d_inputs:
                        continue
                    d_inputs[n] += self._dRdp_matvec(n, lam, transpose=True).reshape(d_inputs[n].shape)
            else:  # pragma: no cover
                raise ValueError(mode)
        finally:
            self._done()

    # -- helpers ----------------------------------------------------
    @staticmethod
    def _matvec(A, x):
        v = A.createVecRight()
        v.setArray(np.asarray(x))
        y = A.createVecLeft()
        A.mult(v, y)
        return y.getArray().copy()

    def _dRdp_matvec(self, name, vec, transpose):
        kind = self._kind[name]
        if kind == "direct":  # dR/d(load_vector) = -I
            return -np.asarray(vec).reshape(-1)
        if name not in self._dRdf:
            self._dRdf[name] = self._assemble_dRdp(name)
        mat = self._dRdf[name]
        if kind == "mesh":
            if transpose:
                y = mat.createVecRight()
                mat.multTranspose(_petsc_vec(mat.createVecLeft(), vec), y)
                local = y.getArray().reshape(-1, self.gdim)
                ext = np.zeros((self._node_idx.size, self.gdim))
                np.add.at(ext, self._node_idx, local)
                return ext
            gathered = np.asarray(vec).reshape(-1, self.gdim)[self._node_idx].reshape(-1)
            y = mat.createVecLeft()
            mat.mult(_petsc_vec(mat.createVecRight(), gathered), y)
            return y.getArray().copy()
        # plain function arg
        if transpose:
            y = mat.createVecRight()
            mat.multTranspose(_petsc_vec(mat.createVecLeft(), vec), y)
        else:
            y = mat.createVecLeft()
            mat.mult(_petsc_vec(mat.createVecRight(), vec), y)
        return y.getArray().copy()


def _petsc_vec(v, values):
    v.setArray(np.asarray(values).reshape(-1))
    v.assemble()
    return v


_ALL_FORM_ARGS = _FUNCTION_ARGS + ("density", "E", "nu", "disp_solid")
_ARG_TO_PDE = {**{n: _PDE_ATTR.get(n, n) for n in _FUNCTION_ARGS},
               "density": "density", "E": "E", "nu": "nu", "disp_solid": "w"}


class ShellScalarFormsOp(csdl.CustomExplicitOperation):
    """Assemble several scalar UFL output forms + their partials in one op.

    Port of femo's ``OutputOperation`` (multi-form). ``forms`` maps output name ->
    (ufl_form, tuple of arg names). Function args are written onto the persistent
    ``ShellPDE`` functions; ``mesh_nodes`` writes the mesh geometry (scratch buffer,
    restored after each call) and, when differentiable, contributes a
    ``d(form)/d(SpatialCoordinate)`` partial scattered back to user node ordering.

    ``orientation``, given, is ``(name, space)`` (``name`` in ``("fiber_angle",
    "fiber_direction")``) -- needed whenever one of ``forms``' arg-name tuples
    includes it (``elastic_energy`` depends on theta for an oriented composite; a
    strain-field form does not). Unlike the fixed-space ``_ARG_TO_PDE`` args, an
    orientation coefficient's space is caller-chosen, so it is resolved through
    ``pde.coefficient(name, space)`` (see that method) rather than a fixed ``pde``
    attribute -- the same seam ``ShellSolveOp`` uses, so both ops reference the exact
    same persistent ``Function`` the form the caller built was differentiated against.

    ``coefficients``, given, is a dict ``{name: space}`` generalizing that same seam
    to any other arg name a caller-built form needs on a caller-chosen space (Phase
    F's per-load-term / arbitrary-``constitutive_space`` forms -- e.g. ``hm.solve``'s
    ``ShellPDE.compliance_form(loads=...)``, whose args are per-term names like
    ``"traction_0"`` with no fixed ``pde`` attribute at all). ``None`` (default,
    every pre-Phase-F call site) leaves every non-orientation arg on the fixed
    ``_ARG_TO_PDE`` route, unchanged.
    """

    def __init__(self, pde, forms: dict, differentiable_geometry: bool = False, orientation=None,
                coefficients=None):
        super().__init__()
        self.pde = pde
        self.forms = dict(forms)
        self._diff_geom = differentiable_geometry
        func_args = sorted({a for _, args in self.forms.values() for a in args})
        self.arg_names = tuple(func_args + (["mesh_nodes"] if differentiable_geometry else []))
        oname = orientation[0] if orientation is not None else None
        coefficients = coefficients or {}
        self._func = {}
        for n in func_args:
            if n == oname:
                self._func[n] = pde.coefficient(oname, orientation[1])
            elif n in coefficients:
                self._func[n] = pde.coefficient(n, coefficients[n])
            else:
                self._func[n] = getattr(pde, _ARG_TO_PDE[n])
        self._const_mesh_nodes = None
        self._node_idx = np.asarray(pde.mesh.geometry.input_global_indices, dtype=np.int64)
        self.gdim = pde.mesh.geometry.dim
        if differentiable_geometry:
            self.Vc = functionspace(pde.mesh, pde.mesh.ufl_domain().ufl_coordinate_element())

    def evaluate(self, fe):
        for n in self.arg_names:
            self.declare_input(n, getattr(fe, n))
        if not self._diff_geom:
            self._const_mesh_nodes = np.asarray(fe.mesh_nodes.value)
        out = csdl.VariableGroup()
        for name in self.forms:
            v = self.create_output(name, (1,))
            v.add_name(name)
            setattr(out, name, v)
        for name, (_, args) in self.forms.items():
            for a in args:
                self.declare_derivative_parameters(name, a, dependent=True)
            if self._diff_geom:
                self.declare_derivative_parameters(name, "mesh_nodes", dependent=True)
        return out

    def _push(self, input_vals):
        mn = input_vals["mesh_nodes"] if self._diff_geom else self._const_mesh_nodes
        self.pde.set_geometry(mn)
        for n, fn in self._func.items():
            fa.set_array(fn, input_vals[n])

    def compute(self, input_vals, output_vals):
        self._push(input_vals)
        try:
            for name, (form, _) in self.forms.items():
                output_vals[name] = np.array([fa.assemble_scalar(form)])
        finally:
            self.pde.restore_geometry()

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        self._push(input_vals)
        try:
            for name, (form, args) in self.forms.items():
                for a in args:
                    derivatives[name, a] = fa.assemble_vector(
                        ufl.derivative(form, self._func[a])
                    ).reshape(1, -1)
                if self._diff_geom:
                    g = fa.assemble_vector(
                        ufl.derivative(form, self.pde.X, ufl.TestFunction(self.Vc))
                    ).reshape(-1, self.gdim)
                    ext = np.zeros((self._node_idx.size, self.gdim))
                    np.add.at(ext, self._node_idx, g)
                    derivatives[name, "mesh_nodes"] = ext.reshape(1, -1)
        finally:
            self.pde.restore_geometry()


@dataclass
class FieldSpec:
    """One field output: a UFL expression + how to represent it as FE coefficients.

    ``space`` is ``(family, degree)`` (blocked by ``n_components`` when the op builds
    the FunctionSpace). ``method``:

    * ``"project"`` (default) -- L2 projection ``M c = b``. Any space; one mass
      factorization; full derivatives including mesh coordinates.
    * ``"interpolate"`` -- collocate ``expr`` at the space's nodal points. Any
      Lagrange space; cheaper (no solve); exact where ``expr`` already lies in the
      space. Has **no** mesh-coordinate (shape) derivative -- use it only when the
      geometry is fixed.
    * ``"average"`` / ``"midpoint"`` -- ``(1/vol_K) * integral_K(expr)``, i.e. L2
      projection onto ``("DG", 0)`` (diagonal mass -> a divide, no solve); ``"average"``
      uses full quadrature, ``"midpoint"`` a single centroid point. DG0 only; full
      derivatives including mesh coordinates.
    """

    expr: object
    n_components: int
    args: tuple = ("disp_solid",)
    kind: str = "scalar"          # hermit._field.Field orientation kind
    space: tuple = ("DG", 2)
    method: str = "project"
    frame: str = "local"          # "local" (element in-plane frame) | "global" (Cartesian)


def _space_is_dg(space) -> bool:
    from .spaces import is_discontinuous
    return is_discontinuous(space)


def strain_fields(pde, space=None, method=None, frame=None):
    """``name -> FieldSpec`` for the membrane strain / bending curvature / transverse
    shear (Voigt / engineering convention). Natural space ``("DG", 2)``, natural method
    ``"project"``; override with ``space`` / ``method``.

    ``frame`` selects the components:

    * ``"local"`` -- the element in-plane frame: strain / curvature as a length-3
      in-plane Voigt vector, shear as a length-2 vector. Only well defined on a
      **discontinuous** space (adjacent elements have different in-plane frames, so a
      continuous space would average frame-inconsistent components at shared nodes).
    * ``"global"`` -- global Cartesian: strain / curvature as a length-6 symmetric
      tensor ``[xx, yy, zz, 2yz, 2xz, 2xy]`` (engineering shear), shear as a length-3
      vector. Frame-consistent -> works on CG spaces.

    ``frame=None`` picks ``"local"`` for a DG space and ``"global"`` for a CG space.
    """
    from .kinematics import (vec2D_local_to_global, voigt2D, voigt3D,
                             strain2D_local_to_global as _l2g)

    sp_ = tuple(space) if space is not None else ("DG", 2)
    is_dg = _space_is_dg(sp_)
    fr = frame or ("local" if is_dg else "global")
    if fr not in ("local", "global"):
        raise ValueError(f"frame must be 'local' or 'global', got {fr!r}")
    if fr == "local" and not is_dg:
        raise ValueError(
            f"strain field space {sp_} is continuous (CG); element-local strain "
            f"components are frame-inconsistent at shared nodes. Pass "
            f"strain_field_frame='global' (global Cartesian components), or use a "
            f"discontinuous space such as ('DG', 2).")

    em = pde.elastic_model()
    T = em.E01
    if fr == "local":
        natural = {
            "mid_strain": (voigt2D(em.eps), 3, "strain2"),
            "curvature": (voigt2D(em.kappa), 3, "strain2"),
            "shear_strain": (em.gamma, 2, "shear2"),
        }
    else:
        natural = {
            "mid_strain": (voigt3D(_l2g(em.eps, T)), 6, "tensor3"),
            "curvature": (voigt3D(_l2g(em.kappa, T)), 6, "tensor3"),
            "shear_strain": (vec2D_local_to_global(em.gamma, T), 3, "vector3"),
        }
    return {
        name: FieldSpec(expr, nc, ("disp_solid",), kind=kind, space=sp_,
                        method=method or "project", frame=fr)
        for name, (expr, nc, kind) in natural.items()
    }


def orientation_cos_sin_spec(pde, orientation_name, coeff_space, target_space=("DG", 0),
                             method="average"):
    """``FieldSpec`` for ``(cos theta, sin theta)`` of the fibre orientation angle.

    The angle between a global ``fiber_direction`` and each element's in-plane frame
    is a function of the mesh: moving nodes rotates the frame. Building the cos/sin in
    UFL -- the same expression the shell form itself uses -- keeps that dependence in
    the differentiated path, so the geometry derivative reaches the Tsai-Wu outputs.
    Evaluating the angle in numpy instead freezes it at the reference geometry and
    silently drops a term from the shape gradient.

    ``fiber_angle`` is element-relative and carries no geometry dependence, but it is
    accepted here too so both orientation kinds take one path.
    """
    from .kinematics import local_basis_inplane, orientation_cos_sin

    E0, E1, E2 = local_basis_inplane(pde.mesh)
    coeff = pde.coefficient(orientation_name, coeff_space)
    if orientation_name == "fiber_direction":
        cs = orientation_cos_sin(None, coeff, E0, E1, E2)
    elif orientation_name == "fiber_angle":
        cs = orientation_cos_sin(coeff, None, E0, E1, E2)
    else:
        raise ValueError(
            f"orientation_name must be 'fiber_direction' or 'fiber_angle', "
            f"got {orientation_name!r}")
    return FieldSpec(ufl.as_vector(cs), 2, (orientation_name,), kind="vector2",
                     space=target_space, method=method, frame="local")


class ShellFieldFormsOp(csdl.CustomExplicitOperation):
    """Field outputs -- several UFL expressions -> FE coefficients, in one op.

    ``fields`` maps output name -> :class:`FieldSpec`. Each output is
    ``(n_scalar_dofs, n_components)`` in that field's target space, FE dof order
    (wrap in :class:`hermit._field.Field` for user ordering, point evaluation
    and frame changes).

    Reverse-mode only (``compute_jacvec_product``), like femo's
    ``OutputFieldOperation``: one cotangent solve / matvec per field, never a dense
    Jacobian. ``method="interpolate"`` has no ``mesh_nodes`` derivative -- the op
    raises if ``differentiable_geometry`` is set with an interpolated field.
    """

    # "midpoint" = 0 means "one centroid point" (the definition of the midpoint method),
    # so it stays at 0 regardless of the setting. Only "average" follows the setting.
    _METHODS = ("average", "midpoint", "interpolate", "project")

    def __init__(self, pde, fields: dict, differentiable_geometry: bool = False,
                 coefficients=None, quadrature_degree=4):
        super().__init__()
        self.pde = pde
        self.fields = dict(fields)
        bad = {s.method for s in self.fields.values()} - set(self._METHODS)
        if bad:
            raise ValueError(f"unknown field method(s) {sorted(bad)}; choose from {self._METHODS}")
        self._diff_geom = differentiable_geometry
        if differentiable_geometry and any(s.method == "interpolate" for s in self.fields.values()):
            raise NotImplementedError(
                "field method 'interpolate' has no mesh-coordinate derivative; "
                "use method='project' or method='average'")

        self.arg_names = tuple(sorted({a for s in self.fields.values() for a in s.args}))
        coefficients = coefficients or {}
        self._func = {n: pde.coefficient(n, coefficients[n]) if n in coefficients
                      else getattr(pde, _ARG_TO_PDE[n]) for n in self.arg_names}
        self.gdim = pde.mesh.geometry.dim
        self.tdim = pde.mesh.topology.dim
        self.nel = pde.mesh.topology.index_map(self.tdim).size_local
        self._node_idx = np.asarray(pde.mesh.geometry.input_global_indices, dtype=np.int64)
        self._const_mesh_nodes = None
        if differentiable_geometry:
            self.Vc = functionspace(pde.mesh, pde.mesh.ufl_domain().ufl_coordinate_element())

        self.space, self._nsd = {}, {}
        self._L, self._vol, self._Mform = {}, {}, {}
        self._ksp = {}                       # cached scalar mass factorization (constant geometry only)
        self._holder = {}                    # a Function(V) per field, reused
        self._ipts = {}                      # interpolation points per field
        self._iexpr, self._ijexpr = {}, {}   # compiled interpolate value / jacobian Expressions
        for name, s in self.fields.items():
            # A 1-component field is a *scalar* space, not a block-size-1 vector
            # space: DOLFINx would give the latter a shape-(1,) TestFunction, which
            # ufl.inner cannot pair with a scalar expression (von Mises stress). The
            # strain outputs are all n_components > 1, so this path was unexercised
            # until stress_field.
            V = functionspace(pde.mesh, s.space if s.n_components == 1
                              else (*s.space, (s.n_components,)))
            self.space[name] = V
            self._nsd[name] = V.dofmap.index_map.size_local
            self._holder[name] = Function(V)
            tv = ufl.TestFunction(V)
            if s.method in ("average", "midpoint"):
                if tuple(s.space) != ("DG", 0):
                    raise ValueError(f"field {name!r}: method={s.method!r} requires space=('DG', 0)")
                qdeg = 0 if s.method == "midpoint" else quadrature_degree
                dxm = ufl.dx(domain=pde.mesh,
                             metadata={"quadrature_degree": qdeg})
                self._L[name] = ufl.inner(s.expr, tv) * dxm
                one = 1.0 if s.n_components == 1 else ufl.as_vector([1.0] * s.n_components)
                self._vol[name] = ufl.inner(one, tv) * dxm
            elif s.method == "project":
                self._L[name] = ufl.inner(s.expr, tv) * ufl.dx
                self._Mform[name] = ufl.inner(ufl.TrialFunction(V), tv) * ufl.dx
            else:  # interpolate
                pts = _compat.interpolation_points(V)
                self._ipts[name] = pts
                self._iexpr[name] = Expression(s.expr, pts)
                self._ijexpr[name] = {
                    a: Expression(ufl.derivative(s.expr, self._func[a],
                                                 ufl.TrialFunction(self._func[a].function_space)), pts)
                    for a in s.args
                }

    # -- graph wiring ------------------------------------------------------
    def evaluate(self, fe):
        for a in self.arg_names:
            self.declare_input(a, getattr(fe, a))
        if self._diff_geom:
            self.declare_input("mesh_nodes", fe.mesh_nodes)
        else:
            self._const_mesh_nodes = np.asarray(fe.mesh_nodes.value)
        out = csdl.VariableGroup()
        for name, s in self.fields.items():
            v = self.create_output(name, (self._nsd[name], s.n_components))
            v.add_name(name)
            setattr(out, name, v)
            for a in s.args:
                self.declare_derivative_parameters(name, a, dependent=True)
            if self._diff_geom and s.method != "interpolate":
                self.declare_derivative_parameters(name, "mesh_nodes", dependent=True)
        return out

    # -- FE-side push ----------------------------------------------------
    def _push(self, input_vals):
        mn = input_vals["mesh_nodes"] if self._diff_geom else self._const_mesh_nodes
        self.pde.set_geometry(mn)
        for a in self.arg_names:
            fa.set_array(self._func[a], input_vals[a])

    def _mass_ksp(self, name):
        """MUMPS factorization of the scalar target-space mass matrix.

        A blocked vector mass matrix is this scalar matrix repeated once per
        component.  Factor the scalar matrix once per target space, so the three
        strain fields share the one factorization their common target space permits.
        """
        space = self.fields[name].space
        if not self._diff_geom and space in self._ksp:
            return self._ksp[space]
        V = make_space(self.pde.mesh, space)
        v = ufl.TestFunction(V)
        ksp = fa.ksp_mumps(fa.assemble_matrix(ufl.TrialFunction(V) * v * ufl.dx))
        if not self._diff_geom:
            self._ksp[space] = ksp
        return ksp

    @staticmethod
    def _msolve(ksp, rhs, n_components):
        M = ksp.getOperators()[0]
        rhs = np.asarray(rhs).reshape(-1, n_components)
        out = np.empty_like(rhs)
        for c in range(n_components):
            b, x = M.createVecRight(), M.createVecRight()
            b.setArray(rhs[:, c]); b.assemble()
            ksp.solve(b, x)
            out[:, c] = x.getArray()
        return out.reshape(-1)

    # -- forward -------------------------------------------------------
    def compute(self, input_vals, output_vals):
        self._push(input_vals)
        try:
            for name, s in self.fields.items():
                nsd, nc = self._nsd[name], s.n_components
                if s.method in ("average", "midpoint"):
                    arr = fa.assemble_vector(self._L[name]) / fa.assemble_vector(self._vol[name])
                elif s.method == "project":
                    arr = self._msolve(self._mass_ksp(name), fa.assemble_vector(self._L[name]), nc)
                else:  # interpolate
                    fh = self._holder[name]
                    fh.interpolate(self._iexpr[name])
                    arr = fh.x.array
                output_vals[name] = np.asarray(arr[: nsd * nc]).reshape(nsd, nc)
        finally:
            self.pde.restore_geometry()

    # -- reverse (VJP) -----------------------------------------------
    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        assert mode == "rev"
        self._push(inputs)
        try:
            tc = ufl.TrialFunction(self.Vc) if self._diff_geom else None
            geom = np.zeros((self._node_idx.size, self.gdim))
            for name, s in self.fields.items():
                if name not in d_outputs:
                    continue
                lam = np.asarray(d_outputs[name]).reshape(-1)          # cotangent, flat
                if not lam.any():
                    continue
                self._vjp_field(name, s, lam, outputs[name], d_inputs, tc, geom)
            if self._diff_geom and "mesh_nodes" in d_inputs and geom.any():
                d_inputs["mesh_nodes"] = geom.reshape(d_inputs["mesh_nodes"].shape)
        finally:
            self.pde.restore_geometry()

    def _vjp_field(self, name, s, lam, cval, d_inputs, tc, geom):
        if s.method == "interpolate":
            for a in s.args:
                if a in d_inputs:
                    d_inputs[a] += (self._interp_jac(name, a).T @ lam).reshape(d_inputs[a].shape)
            return

        L = self._L[name]
        avg = s.method in ("average", "midpoint")
        if avg:
            vol = fa.assemble_vector(self._vol[name])
            lam_t = lam / vol                                          # M^-1 lambda (diagonal)
        else:  # project
            lam_t = self._msolve(self._mass_ksp(name), lam, s.n_components)

        for a in s.args:
            if a not in d_inputs:
                continue
            dLda = fa.assemble_matrix(ufl.derivative(L, self._func[a]))
            d_inputs[a] += _mat_tvec(dLda, lam_t).reshape(d_inputs[a].shape)

        if not self._diff_geom or "mesh_nodes" not in d_inputs:
            return
        g = _mat_tvec(fa.assemble_matrix(ufl.derivative(L, self.pde.X, tc)), lam_t)
        if avg:
            b = fa.assemble_vector(L)
            dvol = fa.assemble_matrix(ufl.derivative(self._vol[name], self.pde.X, tc))
            g = g - _mat_tvec(dvol, lam * b / vol**2)                  # - d(vol)/dX^T (lam c / vol)
        else:  # project:  - d(M c)/dX^T lam_t
            c = self._holder[name]
            fa.set_array(c, np.asarray(cval).reshape(-1))
            dmc = fa.assemble_matrix(ufl.derivative(ufl.action(self._Mform[name], c), self.pde.X, tc))
            g = g - _mat_tvec(dmc, lam_t)
        np.add.at(geom, self._node_idx, g.reshape(-1, self.gdim))

    def _interp_jac(self, name, arg):
        """Sparse d(interpolated coeffs)/d(arg dofs)  ((nsd*nc) x n_arg_dofs), FE order."""
        V = self.space[name]
        nc = self.fields[name].n_components
        nn = np.asarray(V.dofmap.list)                                # (nel, n_nodes)
        npts = nn.shape[1]
        vj = np.asarray(self._ijexpr[name][arg].eval(
            self.pde.mesh, np.arange(self.nel, dtype=np.int32)))
        vj = vj.reshape(self.nel, npts, nc, -1)                       # (nel, npts, nc, n_a_local)
        fn = self._func[arg]
        acols = np.asarray(fn.function_space.dofmap.list)
        abs_ = fn.function_space.dofmap.bs
        if abs_ > 1:
            acols = (acols[:, :, None] * abs_ + np.arange(abs_)).reshape(self.nel, -1)
        rows = np.broadcast_to(nn[:, :, None, None] * nc + np.arange(nc)[:, None], vj.shape)
        cols = np.broadcast_to(acols[:, None, None, :], vj.shape)
        # Keep only the owning cell's row for each target dof: interpolation is a
        # per-cell point evaluation with last-writer-wins, so the union of every
        # adjacent cell's contribution (what a plain COO scatter builds, since the
        # cells contribute on different source columns) over-counts a shared CG dof.
        # No-op on DG. See fa.owned_entry_mask.
        keep = np.broadcast_to(
            fa.owned_entry_mask(nn, self._nsd[name])[:, :, None, None], vj.shape)
        return sp.coo_matrix((vj[keep], (rows[keep], cols[keep])),
                             shape=(self._nsd[name] * nc, fn.x.array.size)).tocsr()


def _mat_tvec(A, x):
    """A^T @ x for an assembled PETSc Mat and a numpy vector."""
    v, y = A.createVecLeft(), A.createVecRight()
    v.setArray(np.asarray(x)); v.assemble()
    A.multTranspose(v, y)
    return y.getArray().copy()
