"""``inertia_relief`` -- the inertial load that balances a free structure.

The six rigid-body modes of the shell are ``u = e_k`` (translations) and
``u = e_k x (x - about)``, ``theta = e_k`` (rotations); all six lie in the state
space exactly. The applied loads' virtual work on them is their force and moment
resultant ``R``; the structure's translational mass on them is the 6x6 rigid-body
mass matrix ``M``. The inertial load with rigid acceleration ``g = -M^{-1} R`` then
does exactly the opposite work on every mode, so ``loads + relief`` is self-
equilibrated.

Everything is assembled through ``ShellScalarFormsOp`` or plain CSDL, so the relief
acceleration is differentiable in the load coefficients, thickness, density and --
with a live ``geometry`` -- the mesh coordinates, including the moment arms of a
point load or load vector.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import csdl_alpha as csdl
import ufl

from .fenics.ops import ShellScalarFormsOp
from ._geometry import geometry as _geometry
from ._solve import _load_work_terms, _pde_for
from .loads import inertial_load
from .transfer import interpolation_matrix

_GEOM_SPACE = ("Lagrange", 1, (3,))


def _rigid_modes(mesh, about):
    """``[(u, theta), ...]`` -- the six rigid-body modes as UFL expressions,
    translations first; ``theta`` is ``None`` for a translation."""
    X = ufl.SpatialCoordinate(mesh)
    r = X - ufl.as_vector([float(c) for c in about])
    axes = [ufl.as_vector([1.0 if i == k else 0.0 for i in range(3)]) for k in range(3)]
    return [(e, None) for e in axes] + [(ufl.cross(e, r), e) for e in axes]


def _direct_resultant(domain, geometry, direct, about):
    """``(F, M)`` of a direct right-hand side about ``about``, as a ``(6,)`` Variable.

    The moment arms are the displacement dofs' coordinates, interpolated from the
    live geometry nodes, so a point load's moment follows the material point it is
    attached to."""
    W = domain.W
    ndof = W.dofmap.index_map.size_local * W.dofmap.index_map_bs
    Vu, u_dofs = W.sub(0).collapse()
    _, t_dofs = W.sub(1).collapse()
    u_dofs, t_dofs = np.asarray(u_dofs).reshape(-1), np.asarray(t_dofs).reshape(-1)

    def select(dofs):
        return sp.csr_matrix((np.ones(dofs.size), (np.arange(dofs.size), dofs)),
                             shape=(dofs.size, ndof))

    n_u = u_dofs.size // 3
    d_u = csdl.reshape(csdl.sparse.matvec(select(u_dofs), direct), (n_u, 3))
    d_t = csdl.reshape(csdl.sparse.matvec(select(t_dofs), direct), (t_dofs.size // 3, 3))
    P = interpolation_matrix(domain.mesh, domain.function_space(_GEOM_SPACE), Vu)
    x_u = csdl.sparse.matvec(P, csdl.reshape(geometry.field.coeffs, (P.shape[1],)))
    r_u = csdl.reshape(x_u - np.tile(np.asarray(about, dtype=float), n_u), (n_u, 3))
    force = csdl.sum(d_u, axes=(0,))
    moment = csdl.sum(csdl.cross(r_u, d_u, axis=1), axes=(0,)) + csdl.sum(d_t, axes=(0,))
    return csdl.concatenate((force, moment))


def inertia_relief(domain, material, loads, *, geometry=None, about=None):
    """The inertial load that puts a free structure in equilibrium with ``loads``.

    Finds the rigid-body acceleration at which the structure's own inertia exactly
    balances the force and moment resultants of ``loads``, and returns that
    inertial load. ``loads + relief`` is self-equilibrated: solve it with
    :func:`~hermit.gauge` holding all six dofs at one point, and the gauge carries no
    reaction, so where it sits changes the displacement only by a rigid-body motion.

    Parameters
    ----------
    domain : ShellDomain
    material : Material
        Supplies ``thickness`` and ``density``; use the material passed to
        :func:`~hermit.solve`.
    loads : Loads
        The applied loads to balance, including any inertial terms such as
        gravity. Read, not modified, and not included in the result.
    geometry : Geometry, optional
        Pass the same geometry as to :func:`~hermit.solve` for the resultants to be
        differentiable in the mesh coordinates. Defaults to the reference
        configuration.
    about : array_like, optional
        ``(3,)`` point the linear part of the acceleration refers to. Default the
        global origin; pass the centre of gravity to read the linear part as the
        centre-of-gravity acceleration. Fixed, not differentiable. The balancing
        load itself does not depend on it.

    Returns
    -------
    relief : Loads
        A single :func:`~hermit.inertial_load` term.
    acceleration : csdl.Variable
        Shape ``(6,)``: the load per unit mass at ``about``, then the angular
        acceleration, both in :func:`~hermit.inertial_load`'s gravity-sign
        convention. The rigid body's own acceleration is the negative.

    Raises
    ------
    ValueError
        If ``material`` or ``loads`` was built on a different domain.

    Notes
    -----
    The mass is translational only, as in :func:`~hermit.inertial_load` and
    :func:`~hermit.mass`; since the relief load uses the same mass model, the
    balance is exact to round-off on affine cells. On non-affine (warped) cells the
    resultant and residual integrands are integrated with different automatically
    chosen quadrature, and the balance holds to quadrature accuracy.

    Examples
    --------
    >>> relief, accel = hm.inertia_relief(domain, material, loads)
    >>> support = hm.gauge(domain, at=p, dofs=("ux", "uy", "uz", "rx", "ry", "rz"))
    >>> state = hm.solve(domain, material, loads + relief, support)
    """
    if material.domain is not domain:
        raise ValueError("inertia_relief: material must be built against this exact ShellDomain")
    if loads.domain is not domain:
        raise ValueError("inertia_relief: loads must be built against this exact ShellDomain")
    about = np.zeros(3) if about is None else np.asarray(about, dtype=float).reshape(3)
    geometry = _geometry(domain) if geometry is None else geometry
    pde = _pde_for(domain)
    modes = _rigid_modes(domain.mesh, about)

    terms, term_args, values, coefficients = _load_work_terms(pde, loads)
    forms = {}
    zero = ufl.as_vector([0.0, 0.0, 0.0])
    for i, (u, theta) in enumerate(modes):
        # a distributed moment does no work on a translation
        keep = [k for k, term in enumerate(terms) if theta is not None or term[0] != "moment"]
        if keep:
            form = pde.load_work_form([terms[k] for k in keep], u, zero if theta is None else theta)
            forms[f"resultant_{i}"] = (form, tuple(sorted({a for k in keep for a in term_args[k]})))

    names = ("relief_thickness", "relief_density")
    for name, field in zip(names, (material.thickness, material.density)):
        values[name] = field.coeffs
        coefficients[name] = field.space
    t, rho = (pde.coefficient(n, coefficients[n]) for n in names)
    for i in range(6):
        for j in range(i, 6):
            forms[f"mass_{i}_{j}"] = (rho * t * ufl.inner(modes[i][0], modes[j][0]) * ufl.dx, names)

    op = ShellScalarFormsOp(pde, forms, differentiable_geometry=geometry.is_differentiable,
                            coefficients=coefficients)
    out = op.evaluate(SimpleNamespace(**values, mesh_nodes=geometry.nodes))

    resultant = csdl.concatenate(tuple(
        getattr(out, f"resultant_{i}") if f"resultant_{i}" in forms
        else csdl.Variable(value=np.zeros(1)) for i in range(6)))
    if loads.direct is not None:
        resultant = resultant + _direct_resultant(domain, geometry, loads.direct, about)
    mass = csdl.reshape(csdl.concatenate(tuple(
        getattr(out, f"mass_{min(i, j)}_{max(i, j)}") for i in range(6) for j in range(6))), (6, 6))

    acceleration = -csdl.solve_linear(mass, resultant)
    relief = inertial_load(domain, material, acceleration=acceleration[0:3],
                           angular_acceleration=acceleration[3:6], about=about)
    return relief, acceleration
