"""Reissner-Mindlin shell elastic model: strains, stress resultants, energy, residual.

Port of ``ElasticModelShapeOpt`` from femo dev_coupling. Geometry sensitivity is via
``SpatialCoordinate`` differentiation, so all gradients are plain ``ufl.grad`` on the
(possibly moved) mesh -- no ``gradx`` / ``J(uhat)``.
"""

from dolfinx.fem import Constant, Function, functionspace
from ufl import (
    CellDiameter,
    TestFunction,
    as_tensor,
    as_vector,
    cross,
    derivative,
    dot,
    grad,
    indices,
    inner,
    split,
    dx,
)

from .kinematics import (
    congruent_transform,
    constant_normal_bending_curvature,
    global_to_local_inplane,
    gradv_local,
    local_basis_inplane,
    membrane_strain,
    orientation_cos_sin,
    shear_rotation,
    strain_rotation,
    voigt2D,
)


# ``fiber_direction``'s cos/sin come from a tangent-plane projection (``unit()``,
# i.e. a ``sqrt`` and a division -- see ``kinematics.orientation_cos_sin``), which
# is not polynomial. UFL's automatic quadrature-degree estimator does not know that
# a sqrt/division node is "morally the same order" as the plain ``cos``/``sin`` of a
# ``fiber_angle`` coefficient -- it estimates a much higher degree for the direction
# form (e.g. 46-86 vs 18-30 per energy term, measured on this module's own test
# fixture) purely from the expression tree's shape. The shell energy integrand is not
# exactly polynomial anyway (CellDiameter-based drilling stabilization, a bilinear
# quad's non-affine terms), so two different auto-picked quadrature rules integrate it
# to two different (both "reasonable", neither exact) answers -- a ~1e-6 relative
# compliance gap between the fiber_angle and fiber_direction forms that has nothing to
# do with the rotation math (verified: forcing any *explicit* common degree from 2 to
# 24 makes the two branches agree to ~1e-9, an order of magnitude tighter than the
# default-degree gap, at every degree tested from 4 up).
#
# Fix: give every *oriented* energy integral an explicit, deliberately chosen degree
# so the answer no longer depends on this heuristic, and so adding an orientation
# cannot silently change the integration accuracy of the whole elastic energy
# underneath it. 4 matches the quadrature degree already used everywhere else a
# measure is pinned explicitly in this codebase (BC penalty terms, the p-norm stress
# measure, DG0 field-average methods) and is already deep in the converged regime for
# this integrand (2 is measurably too coarse -- ~1% off the degree>=4 plateau; 4..24
# agree to ~1e-8). The *unoriented* path keeps plain ``ufl.dx`` (whatever degree UFL
# already auto-picks for it today) -- untouched, so it stays bit-identical to the
# pre-orientation form.
#
# That degree is ``ShellDomain(quadrature_degree=...)``, threaded here via
# ``ShellPDE``; it defaults to 4, so everything above describes the default. It is a
# setting rather than a constant because the right value is a property of the
# *problem* (curvature, element order, how peaked the stress is), not of this module.


class ElasticModel:
    def __init__(self, mesh, w, clt_matrices, fiber_angle=None, fiber_direction=None, quadrature_degree=4):
        """``fiber_angle`` / ``fiber_direction`` (at most one) rotate ``clt_matrices``
        (laminate axes) into the element frame *inside the form*. With neither given,
        ``A/B/D/A_s`` are used exactly as passed (no identity rotation multiplied in),
        so the unoriented form is bit-identical to the form without orientation.
        """
        self.mesh = mesh
        self.w = w
        self.u_mid, self.theta = split(w)
        self.W = w.function_space
        self.quadrature_degree = quadrature_degree

        E0, E1, self.E2 = local_basis_inplane(mesh)
        self.E01 = global_to_local_inplane(E0, E1)

        # mid-surface offset (0 => reference plane at mid-surface)
        self.offset = Function(functionspace(mesh, ("DG", 0)))

        self.gradu = grad(self.u_mid)
        self.t_gu = gradv_local(self.gradu, self.E01)
        A, B, D, A_s = clt_matrices

        self._oriented = fiber_angle is not None or fiber_direction is not None
        if self._oriented:
            c, s = orientation_cos_sin(fiber_angle, fiber_direction, E0, E1, self.E2)
            Teps, R = strain_rotation(c, s), shear_rotation(c, s)
            # Rotating A/B/D/A_s here (rather than rotating the strain and contracting
            # in laminate axes) is exactly energy-equivalent -- see the kinematics
            # module docstring -- and means everything downstream (stress resultants,
            # energies, *and* the drilling stabilization's ``D[0, 0]``) automatically
            # sees the element-frame matrices with no further changes.
            A, B, D = (congruent_transform(Teps, X) for X in (A, B, D))
            A_s = congruent_transform(R, A_s)
        self.A, self.B, self.D, self.A_s = A, B, D, A_s

        self.kappa = self._bending_curvature()
        self.eps = self._membrane_strains()
        self.gamma = self._shear_strains()
        self.N, self.M, self.Q = self._stress_resultants()

    # --- strains ---------------------------------------------------------
    def _membrane_strains(self):
        return membrane_strain(self.u_mid, self.E01) - self.offset * self.kappa

    def _bending_curvature(self):
        """Constant-normal (shallow-shell) curvature.

        The cell normal is held constant within the bending measure, so curvature
        comes from ``grad(theta)`` alone. This makes rigid-body objectivity
        structural on warped cells. See ``docs/src/background.md`` for the
        formulation choice, evidence, and range of applicability.
        """
        return constant_normal_bending_curvature(self.theta, self.E01)

    def _shear_strains(self):
        # transverse shear strains in the local in-plane frame, as a length-2 vector
        dudxi2_local = _contract1(-cross(self.E2, self.theta), self.E01)
        gradu2_local = _contract1(dot(self.E2, self.gradu), self.E01)
        return dudxi2_local + gradu2_local

    # --- stress resultants ----------------------------------------------
    def _stress_resultants(self):
        """``B`` enters with a **minus** sign: this model's kinematics are
        ``eps(xi) = eps_mid - xi * kappa`` (``ShellStressRM.u`` is
        ``u_mid - xi * (E2 x theta)``, and ``_membrane_strains`` agrees), whereas
        ``A/B/D`` come from classical lamination theory, which is written for
        ``eps(z) = eps_0 + z * kappa_clt``. The two curvatures are opposite:
        ``kappa_clt = -kappa``.

        With ``A = int Q dxi``, ``B = int Q xi dxi``, ``D = int Q xi^2 dxi``, the
        physical energy density is

            1/2 int (eps - xi*kappa)^T Q (eps - xi*kappa) dxi
              = 1/2 (eps^T A eps  -  2 eps^T B kappa  +  kappa^T D kappa)

        so only the *cross* term flips -- ``A`` and ``D`` are even in ``xi`` and are
        used as-is. Writing ``N = A eps - B kappa`` and ``M = -B eps + D kappa`` makes
        ``1/2 (N.eps + M.kappa)`` reproduce exactly that.

        This sign is easy to get wrong (``+B`` in both) and hard to catch: ``B = 0``
        for an isotropic single layer *and* for every symmetric layup, which is all
        the tests and the femo reference contain.

        ``M`` is therefore the conjugate of *this* model's ``kappa``, i.e. the negative
        of the CLT moment resultant; ``M @ kappa`` is invariant under that pair of sign
        flips, which is all the energy uses it for.
        """
        N = self.A * voigt2D(self.eps) - self.B * voigt2D(self.kappa)
        M = -self.B * voigt2D(self.eps) + self.D * voigt2D(self.kappa)
        Q = self.A_s * self.gamma
        return N, M, Q

    # --- energies -------------------------------------------------------
    def shear_energy(self, dx_shear=dx):
        return 0.5 * dot(self.Q, self.gamma) * dx_shear

    def membrane_energy(self, dx_inplane=dx):
        return 0.5 * dot(self.N, voigt2D(self.eps)) * dx_inplane

    def bending_energy(self, dx_inplane=dx):
        return 0.5 * dot(self.M, voigt2D(self.kappa)) * dx_inplane

    def drilling_energy(self, dx_drilling=dx):
        h_mesh = CellDiameter(self.mesh)
        drilling_strain = (self.t_gu[0, 1] - self.t_gu[1, 0]) / 2 + dot(self.theta, self.E2)
        # 12 * D[0,0] recovers E*h**3 for a single isotropic layer and is well-defined
        # for a composite ABD too (unlike max(D.array), which needs an integration
        # domain). self.D is already the *element*-frame D (see __init__) when an
        # orientation is set, so this is automatically the element-frame D11 -- using
        # the un-rotated laminate-axes D11 here would silently change the
        # stabilization coefficient for any oriented composite.
        alpha = 12 * self.D[0, 0]
        drilling_stress = alpha * drilling_strain / h_mesh**2
        return 0.5 * drilling_stress * drilling_strain * dx_drilling

    def _default_energy_measure(self):
        """Plain ``ufl.dx`` (today's implicit, auto-estimated-degree rule) when
        unoriented -- bit-identical to the pre-orientation form; an explicit
        degree when oriented -- see the module-level comment above."""
        if not self._oriented:
            return dx
        return dx(domain=self.mesh, metadata={"quadrature_degree": self.quadrature_degree})

    def elastic_energy(self, dx_inplane=None, dx_shear=None, dx_drilling=None):
        if dx_inplane is None:
            dx_inplane = self._default_energy_measure()
        if dx_shear is None:
            dx_shear = self._default_energy_measure()
        if dx_drilling is None:
            dx_drilling = self._default_energy_measure()
        return (
            self.shear_energy(dx_shear)
            + self.membrane_energy(dx_inplane)
            + self.bending_energy(dx_inplane)
            + self.drilling_energy(dx_drilling)
        )

    # --- weak residual ------------------------------------------------
    def weak_residual(self, elastic_energy, f=None, m=None, load_terms=None, penalty=False,
                      g=None, dss=None, dSS=None, bc_dof_mask=None, penalty_terms=None,
                      penalty_targets=None):
        """``penalty_terms``, given, is a list of ``(dss, dSS, bc_dof_mask)`` triples
        whose contributions are summed (multi-region penalty, e.g.
        ``hm.clamp(...) + hm.symmetry(...)``); with it omitted, the single
        ``(dss, dSS, bc_dof_mask)`` triple is used.

        ``load_terms``, given, is a list of ``(kind, Function)`` pairs: each
        contributes its own residual term, on its own space, in place of the
        fixed-space ``f``/``m`` coefficients ``load_terms=None`` (the default) uses.
        An empty list is valid (no distributed term -- e.g. a point-load-only
        ``Loads``, whose contribution is a direct RHS entry handled entirely outside
        this form); only ``f``/``m`` both ``None`` *and* ``load_terms`` ``None`` is
        the error case (nothing at all to apply)."""
        dw = TestFunction(self.W)
        self.du_mid, self.dtheta = split(dw)
        res = derivative(elastic_energy, self.w, dw)
        if penalty:
            terms = penalty_terms if penalty_terms is not None else [(dss, dSS, bc_dof_mask)]
            res += self._penalty_residual(self.w, dw, g, terms, penalty_targets)
        if load_terms is not None:
            for term in load_terms:
                kind, func = term[:2]
                measure = dx if len(term) == 2 else term[2]
                res -= self._load_term_residual(kind, func, measure)
            return res
        if f is not None:
            res -= inner(f, self.du_mid) * dx
        if m is not None:
            res -= inner(m, self.dtheta) * dx
        if f is None and m is None:
            raise ValueError("weak_residual needs at least one of f, m")
        return res

    def _load_term_residual(self, kind, func, measure=dx):
        """One load term's residual contribution -- the work conjugate
        ``ShellPDE.compliance_form``'s ``loads`` branch also emits, so the two never
        drift apart (see that method's docstring)."""
        if kind == "traction":
            return inner(func, self.du_mid) * measure
        if kind == "moment":
            return inner(func, self.dtheta) * measure
        if kind == "pressure":
            return func * dot(self.E2, self.du_mid) * measure
        raise ValueError(f"unknown load kind {kind!r}")  # pragma: no cover

    def _penalty_residual(self, u, v, g, terms, targets=None):
        """Sum the penalty contribution of each ``(dss, dSS, bc_dof_mask)`` in
        ``terms``. For a single term this reduces to exactly the old expression (no
        ``0 + form`` -- summing UFL forms is fine, but this keeps the one-term path
        bit-identical rather than merely equivalent)."""
        beta = Constant(self.mesh, 1.0e15)
        h_E = CellDiameter(self.mesh)
        pieces = []
        if targets is None:
            targets = [g] * len(terms)
        if len(targets) != len(terms):
            raise ValueError("penalty_targets must have one target per penalty term")
        for term, target in zip(terms, targets):
            dss, dSS, bc_dof_mask = term
            if bc_dof_mask is None:
                integrand = beta / h_E * inner(u - target, v)
            else:
                u_mid, theta = split(u)
                v_mid, v_theta = split(v)
                g_mid, g_theta = split(target)
                mm, mt = bc_dof_mask[:3], bc_dof_mask[3:]
                masked = as_vector(
                    [mm[i] * (u_mid[i] - g_mid[i]) for i in range(3)]
                    + [mt[i] * (theta[i] - g_theta[i]) for i in range(3)]
                )
                v_full = as_vector([v_mid[i] for i in range(3)] + [v_theta[i] for i in range(3)])
                integrand = beta / h_E * inner(masked, v_full)
            pieces.append(integrand * dss + integrand("+") * dSS + integrand("-") * dSS)
        res = pieces[0]
        for p in pieces[1:]:
            res = res + p
        return res


def _contract1(vec3, E01):
    """as_tensor(vec3[j] * E01[i, j], (i,)) -> length-2 local vector."""
    i, j = indices(2)
    return as_tensor(vec3[j] * E01[i, j], (i,))
