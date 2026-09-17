"""``ShellPDE`` -- function spaces + UFL form factories for the R-M shell.

Pure FEniCSx (DOLFINx 0.9 / 0.11), no CSDL. Port of ``rm_shell_pde_composite.py`` (femo
dev_coupling), unified ABD backend. The argument functions live on the object so the
custom operations can update them by array between assemblies.
"""

import scipy.sparse as sp
import ufl
from dolfinx.fem import Constant, Function, assemble_scalar, form, functionspace
from dolfinx.fem.petsc import assemble_matrix

from .elastic_model import ElasticModel
from .material import ABDHolder
from .spaces import make_space, normalize_space, state_space
from .stress import ShellStressRM


class ShellPDE:
    def __init__(self, mesh, element="CG2CG1", element_wise_material=False,
                 elementwise_pressure=False, W=None, quadrature_degree=4):
        """``W=``, given, is an existing mixed state space to reuse instead of
        building a fresh ``state_space(mesh, element)`` -- ``hm.solve`` (via
        ``hermit._solve._pde_for``) passes ``domain.W`` here so the residual / BCs /
        ops it builds all share the *one* ``FunctionSpace`` object a
        ``BoundaryConditions`` was compiled against (a strong ``DirichletBC`` located
        against a structurally identical but distinct ``FunctionSpace`` is silently
        dropped, not an error)."""
        self.mesh = mesh
        self.gdim = mesh.geometry.dim
        self.W = W if W is not None else state_space(mesh, element)
        self.quadrature_degree = quadrature_degree

        mat_family = ("DG", 0) if element_wise_material else ("Lagrange", 1)
        self.VT = functionspace(mesh, mat_family)
        self.VABD = functionspace(mesh, (*mat_family, (3, 3)))
        self.VAs = functionspace(mesh, (*mat_family, (2, 2)))
        pf = ("DG", 0) if elementwise_pressure else ("Lagrange", 1)
        self.VF = functionspace(mesh, (*pf, (3,)))

        self.X = ufl.SpatialCoordinate(mesh)  # geometry-sensitivity handle

        # pristine reference node coordinates. The custom ops treat mesh.geometry.x
        # as a scratch buffer: set it from their mesh_nodes input, assemble, restore.
        self.x_ref = mesh.geometry.x.copy()
        self._node_input_idx = mesh.geometry.input_global_indices

        # persistent argument functions
        self.w = Function(self.W)
        self.A, self.B, self.D = (Function(self.VABD) for _ in range(3))
        self.As = Function(self.VAs)
        self.h = Function(self.VT)
        self.E = Function(self.VT)
        self.nu = Function(self.VT)
        self.density = Function(self.VT)
        self.f = Function(self.VF)
        self.m = Function(self.VF)
        self.g = Function(self.W)  # penalty target (0)

        self._elastic = None
        self._coeff_cache = {}

    # -- geometry scratch-buffer management ----------------------------------
    def set_geometry(self, mesh_nodes) -> None:
        """Write ``mesh.geometry.x`` from user-ordered node coords (n_nodes, gdim)."""
        import numpy as np
        self.mesh.geometry.x[:] = np.asarray(mesh_nodes).reshape(-1, self.gdim)[self._node_input_idx]

    def restore_geometry(self) -> None:
        self.mesh.geometry.x[:] = self.x_ref

    # -- general per-(name, space) coefficient seam ---------------------------
    def coefficient(self, name, space) -> Function:
        """Persistent ``Function`` for ``name`` on ``space``, memoised by ``(name,
        normalized space)`` -- repeated calls with the same arguments return the exact
        same object. That matters: a residual form built against one ``Function``
        instance is wrong (or raises, for a strong BC -- see ``ShellDomain``'s class
        docstring) if later evaluated with a different instance, and the custom ops write
        into the coefficient by array every solve, so it must be the one the compiled
        form actually references.

        General seam for any per-mesh input field a form needs as a UFL coefficient on
        a space the caller picks (materials, loads, orientation, ...) rather than one
        of the PDE's fixed-space attributes (``self.A`` / ``self.h`` / ...).
        """
        norm = normalize_space(space)
        key = (name, norm)
        if key not in self._coeff_cache:
            self._coeff_cache[key] = Function(make_space(self.mesh, norm))
        return self._coeff_cache[key]

    # -- elastic model (rebuilt when the state Function identity changes) ----
    def elastic_model(self, w=None, orientation=None, material=None):
        """``orientation`` is ``(name, Function)`` with ``name`` in ``("fiber_angle",
        "fiber_direction")``, or ``None``. The no-orientation path reuses the
        memoised ``ElasticModel`` (bit-identical to the unoriented form); an oriented
        call always builds a fresh one -- it is only built once per
        ``ShellSolveOp`` / ``ShellScalarFormsOp`` construction (themselves memoised by
        the structural form cache), so there is nothing to gain by caching it too.

        ``material``, given, is a ``(A, B, D, As)`` tuple of ``Function``\\s
        overriding the PDE's own fixed-space ``self.A``/``self.B``/``self.D``/
        ``self.As``: a caller-chosen ``constitutive_space`` (any space, via
        :meth:`coefficient`) reaches the solve instead of only the fixed
        ``VABD``/``VAs`` space ``element_wise_material`` picks. ``None`` (default)
        keeps the fixed-space attrs -- also why the ``material=None`` unoriented path
        is the only one eligible for the ``self._elastic`` memo (a ``material=``
        override is only ever built once per op anyway, same as an oriented call).
        """
        w = self.w if w is None else w
        clt = ABDHolder(*material).CLT if material is not None \
            else ABDHolder(self.A, self.B, self.D, self.As).CLT
        if orientation is not None:
            oname, ofunc = orientation
            return ElasticModel(self.mesh, w, clt, quadrature_degree=self.quadrature_degree, **{oname: ofunc})
        if material is not None:
            return ElasticModel(self.mesh, w, clt, quadrature_degree=self.quadrature_degree)
        if self._elastic is None or self._elastic.w is not w:
            self._elastic = ElasticModel(self.mesh, w, clt, quadrature_degree=self.quadrature_degree)
        return self._elastic

    # -- forms -------------------------------------------------------------
    def residual_form(self, penalty=True, dss=None, dSS=None, bc_dof_mask=None,
                      penalty_terms=None, penalty_targets=None, orientation=None, material=None, loads=None):
        """``penalty_terms``, given, is a list of ``(dss, dSS, bc_dof_mask)`` triples
        whose penalty contributions are summed -- multi-region penalty
        (``hm.clamp(...) + hm.symmetry(...)``); with it omitted (the common case) this
        is exactly the single ``(dss, dSS, bc_dof_mask)`` triple. ``orientation`` is
        ``(name, Function)``, ``material`` is ``(A, B, D, As)`` Functions -- see
        :meth:`elastic_model`.

        ``loads``, given, is a list of ``(kind, Function)`` pairs (``kind`` in
        ``"traction"``/``"moment"``/``"pressure"``): each term gets its own residual
        contribution on its own space, via ``ElasticModel.weak_residual``'s
        ``load_terms`` (instead of the fixed ``self.f``/``self.m`` VF-space
        coefficients this method uses with ``loads=None``, the default). An empty list
        is valid (e.g. a point-load-only ``Loads`` has no distributed term at all) --
        only ``None`` selects the fixed ``f``/``m`` path.
        """
        em = self.elastic_model(orientation=orientation, material=material)
        energy = em.elastic_energy()
        if loads is None:
            return em.weak_residual(
                energy, f=self.f, m=self.m, penalty=penalty, g=self.g,
                dss=dss, dSS=dSS, bc_dof_mask=bc_dof_mask, penalty_terms=penalty_terms,
                penalty_targets=penalty_targets,
            )
        return em.weak_residual(
            energy, load_terms=loads, penalty=penalty, g=self.g,
            dss=dss, dSS=dSS, bc_dof_mask=bc_dof_mask, penalty_terms=penalty_terms,
            penalty_targets=penalty_targets,
        )

    def compliance_form(self, loads=None):
        """``loads=None`` (default) -- the work conjugate of the fixed
        ``self.f``/``self.m`` VF-space coefficients. ``loads``, given, is the same
        ``(kind, Function)`` list ``residual_form`` takes -- the matching compliance
        term per load kind (``inner(u, f)`` / ``inner(theta, m)`` / ``p * dot(n,
        u)``), so this stays the exact work conjugate of ``residual_form``'s
        distributed terms (every load kind must contribute both a residual term and a
        compliance term, or the two drift apart silently). The
        ``point_load``/``load_vector`` direct-RHS contribution is *not* included here
        (it has no distributed form) -- callers add ``csdl.vdot(direct, w)``.
        """
        u_mid, theta = ufl.split(self.w)
        if loads is None:
            return ufl.inner(u_mid, self.f) * ufl.dx + ufl.inner(theta, self.m) * ufl.dx
        total = None
        E2 = None
        for item in loads:
            kind, func = item[:2]
            measure = ufl.dx if len(item) == 2 else item[2]
            if kind == "traction":
                term = ufl.inner(u_mid, func) * measure
            elif kind == "moment":
                term = ufl.inner(theta, func) * measure
            elif kind == "pressure":
                if E2 is None:
                    from .kinematics import local_basis_inplane
                    E2 = local_basis_inplane(self.mesh)[2]
                term = func * ufl.dot(E2, u_mid) * measure
            else:  # pragma: no cover
                raise ValueError(f"unknown load kind {kind!r}")
            total = term if total is None else total + term
        return total if total is not None else Constant(self.mesh, 0.0) * ufl.dx

    def mass_form(self, thickness=None, density=None):
        """``thickness=None``/``density=None`` (default) use the fixed-space
        ``self.h``/``self.density``. Given, they override with an arbitrary-space
        ``Function`` (from :meth:`coefficient`)."""
        t = self.h if thickness is None else thickness
        rho = self.density if density is None else density
        return rho * t * ufl.dx

    def cg_forms(self, thickness=None, density=None):
        """Mass moments, optionally from arbitrary-space coefficients (else the fixed
        ``density`` / ``h`` forms)."""
        t = self.h if thickness is None else thickness
        rho = self.density if density is None else density
        me = rho * t
        return (
            self.X[0] * me * ufl.dx,
            self.X[1] * me * ufl.dx,
            self.X[2] * me * ufl.dx,
            me * ufl.dx,
        )

    def elastic_energy_form(self, orientation=None, material=None):
        return self.elastic_model(orientation=orientation, material=material).elastic_energy()

    def area_form(self, measure=ufl.dx):
        return Constant(self.mesh, 1.0) * measure

    def _stress(self, surface, thickness=None, E=None, nu=None):
        t = self.h if thickness is None else thickness
        E = self.E if E is None else E
        nu = self.nu if nu is None else nu
        try:
            xi2 = {"top": t / 2, "mid": 0.0, "bot": -t / 2}[surface]
        except KeyError:
            raise ValueError("surface must be 'top', 'mid', or 'bot'") from None
        em = self.elastic_model()
        s = ShellStressRM(self.mesh, self.w, t, E, nu,
                          strains=(em.eps, em.kappa, em.gamma))
        return s, xi2

    def von_mises_form(self, surface="top", thickness=None, E=None, nu=None):
        s, xi2 = self._stress(surface, thickness=thickness, E=E, nu=nu)
        return s.von_mises(xi2)

    def pnorm_stress_form(self, dx_measure, m=1e-6, rho=100.0, surface="top",
                           thickness=None, E=None, nu=None):
        s, xi2 = self._stress(surface, thickness=thickness, E=E, nu=nu)
        vm = s.von_mises(xi2)
        alpha = float(assemble_scalar(form(self.area_form(dx_measure))))
        return 1.0 / alpha * (m * vm) ** rho * dx_measure

    # -- setup-time maps -------------------------------------------------
    def force_to_pressure_matrix(self):
        """VF consistent mass matrix M such that M @ pressure = consistent nodal force."""
        v, Pv = ufl.TestFunction(self.VF), ufl.TrialFunction(self.VF)
        A = assemble_matrix(form(ufl.inner(Pv, v) * ufl.dx))
        A.assemble()
        ai, aj, av = A.getValuesCSR()
        M = sp.csr_matrix((av, aj, ai), shape=A.getSize())
        M.eliminate_zeros()
        return M
