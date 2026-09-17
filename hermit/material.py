"""``Material`` + ``Orientation`` -- the constitutive input bundle.

A ``Material`` bundles the ``Field`` objects the residual and the mass / stress /
failure outputs consume. The constitutive tensors are stored **in the laminate's own
material axes, unrotated**: the rotation into each element's frame happens inside the
shell form, so there is no space restriction on any of these fields --
``A``/``B``/``D``/``As`` and ``orientation`` may each live on their own space.
"""

from __future__ import annotations

import numpy as np
import csdl_alpha as csdl

from .csdl_helpers import assemble_abd_isotropic
from .fenics.spaces import normalize_space
from ._field import Field, as_field, as_field_user_order, constant
from .transfer import interpolate


class Orientation:
    """The laminate fibre orientation carried on a :class:`Material`.

    Build one with :func:`fiber_angle` or :func:`fiber_direction`. It does not
    rotate anything at construction time: the rotation into each element's frame
    happens inside the shell form during :func:`~hermit.solve`, and again in the
    Tsai-Wu failure recovery.

    Parameters
    ----------
    domain : ShellDomain
    kind : {'angle', 'direction'}
        Whether ``value`` is an in-plane angle or a global direction vector.
    value : Field
        Scalar field of angles (rad) for ``kind='angle'``, or a ``(3,)`` vector
        field for ``kind='direction'``.
    """

    def __init__(self, domain, *, kind, value):
        if kind not in ("angle", "direction"):
            raise ValueError(f"Orientation kind must be 'angle' or 'direction', got {kind!r}")
        self.domain = domain
        self.kind = kind
        self.value = value   # Field: scalar (kind="angle") or (3,) vector (kind="direction")


def fiber_angle(domain, angle) -> Orientation:
    """Orientation given as an angle from each element's own in-plane axis.

    Parameters
    ----------
    domain : ShellDomain
    angle : Field, float or array_like
        Angle in radians from the element's ``e0`` to the fibre direction, on any
        space. A ``csdl.Variable`` makes it a differentiable design field.

    Returns
    -------
    Orientation

    Raises
    ------
    ValueError
        If ``angle`` is not a scalar field.

    Notes
    -----
    The angle is measured from ``e0``, which follows the mesh parametrisation. On a
    mesh where ``e0`` is not uniform, a continuous (CG) angle field makes the
    physical fibre direction kink slightly at element interfaces -- fine on a flat
    or structured mesh, otherwise prefer :func:`fiber_direction`.

    See Also
    --------
    fiber_direction : orientation given as a global direction.
    """
    field = angle if isinstance(angle, Field) else as_field(domain, angle)
    if field.value_shape != ():
        raise ValueError(f"fiber_angle: angle must be scalar, got value_shape {field.value_shape}")
    return Orientation(domain, kind="angle", value=field)


def fiber_direction(domain, direction) -> Orientation:
    """Orientation given as a global fibre direction.

    Parameters
    ----------
    domain : ShellDomain
    direction : Field or array_like
        A ``(3,)`` global vector, or a ``Field`` of ``(3,)`` vectors on any space
        (e.g. a per-cell curvilinear fibre path).

    Returns
    -------
    Orientation

    Raises
    ------
    ValueError
        If ``direction`` is not a ``(3,)`` vector field.

    Notes
    -----
    The shell form projects the direction into each element's tangent plane and
    never forms the angle explicitly, so there is no branch cut. The projection is
    degenerate only where ``direction`` is parallel to the shell normal, which is
    detected when concrete values reach the form.

    See Also
    --------
    fiber_angle : orientation given as an element-relative angle.
    """
    field = direction if isinstance(direction, Field) else _as_vector_field(domain, direction, ("DG", 0, (3,)))
    if field.value_shape != (3,):
        raise ValueError(f"fiber_direction: direction must be a (3,) vector field, got value_shape {field.value_shape}")
    return Orientation(domain, kind="direction", value=field)


def _as_vector_field(domain, value, default_space) -> Field:
    """Coerce a bare ``(3,)`` vector (broadcast to every dof), or anything
    ``as_field`` already resolves (a ``Field``, a per-node/per-cell array, an
    explicit-space FE-dof-order array), to a ``Field``. ``as_field``'s own
    resolution order treats only a *length-1* value as a broadcast constant, so a
    length-3 vector -- the common case for a load or a fibre direction -- needs this
    extra step first, or it would be misread as "one value per dof" on a
    coincidentally-3-node/3-cell mesh."""
    is_var = isinstance(value, csdl.Variable)
    shape = tuple(value.shape) if is_var else np.asarray(value).shape
    total = int(np.prod(shape)) if shape else 1
    if total == 3:
        return constant(domain, default_space, value)
    return as_field_user_order(domain, value, default_space)


class Material:
    """The constitutive input bundle consumed by the solve and the outputs.

    Build one with :func:`isotropic`, :func:`laminate`, :func:`composite` or
    :func:`thickness_only` rather than calling this constructor directly.

    Parameters
    ----------
    A, B, D : Field or None
        Membrane, coupling and bending stiffness, each a ``(3, 3)`` field in the
        laminate's own material axes, **unrotated**. ``None`` only for a
        :func:`thickness_only` material.
    As : Field or None
        Transverse-shear stiffness, a ``(2, 2)`` field in the ``(xz, yz)``
        ordering the shell form uses.
    thickness, density : Field
        Scalar fields, on any space.
    orientation : Orientation, optional
        Applied to ``A``/``B``/``D``/``As`` inside the shell form. ``None`` means
        no rotation, which is correct for an isotropic material.
    E, nu : Field, optional
        Isotropic constants, needed by the von Mises stress outputs.
    layup : Layup, optional
        The ply stack, needed by the Tsai-Wu failure outputs.

    Attributes
    ----------
    domain : ShellDomain
        Taken from ``thickness``.
    """

    def __init__(self, A, B, D, As, thickness, density, *, orientation=None,
                 E=None, nu=None, layup=None):
        self.A, self.B, self.D, self.As = A, B, D, As
        self.thickness = thickness
        self.density = density
        self.orientation = orientation
        self.E = E
        self.nu = nu
        self.layup = layup

    @property
    def domain(self):
        return self.thickness.domain


# -- constitutive-space default ("DG", d), d = max input degree ------------------

def _default_constitutive_space(*fields):
    d = max((f.space[1] for f in fields if f is not None), default=0)
    return ("DG", d)


def _to_scalar_space(field: Field, scalar_space) -> Field:
    """``field``, brought onto ``scalar_space`` (a ``(family, degree)`` pair) if it
    is not already there -- via ``interpolate``, a plain collocation."""
    if field.space[:2] == scalar_space:
        return field
    return interpolate(field, scalar_space)


# -- constructors -----------------------------------------------------------

def isotropic(domain, *, E, nu, thickness, density, constitutive_space=None,
             orientation=None) -> Material:
    """Isotropic single-layer material.

    ``A``/``B``/``D``/``As`` are assembled from the closed form in plain CSDL, with a
    0.833 shear-correction factor.

    Parameters
    ----------
    domain : ShellDomain
    E, nu, thickness, density : Field, float or array_like
        Young's modulus, Poisson's ratio, shell thickness and mass density. Each
        may be a scalar (broadcast), a per-vertex / per-cell array, or a ``Field``
        on its own space. A ``csdl.Variable`` makes it a design variable.
    constitutive_space : tuple, optional
        Space the ABD fields are evaluated on. Defaults to ``("DG", d)`` where
        ``d`` is the highest degree among ``E``, ``nu`` and ``thickness``.
    orientation : Orientation, optional
        Accepted and ignored -- an isotropic in-plane stiffness has no preferred
        axis, so rotating it is a no-op. The parameter exists so that one
        orientation object can be passed uniformly to any material constructor.

    Returns
    -------
    Material

    Notes
    -----
    The ABD fields are an *interpolant* of the closed form at the
    ``constitutive_space`` dofs, not an exact representation of it: ``E*t`` and
    ``t**3`` have higher polynomial degree than any fixed space carries. Raise
    ``constitutive_space`` if that matters for a strongly graded thickness.

    Examples
    --------
    >>> mat = hm.isotropic(domain, E=4.32e8, nu=0.0, thickness=0.2, density=1.0)
    """
    E_f, nu_f = as_field(domain, E), as_field(domain, nu)
    t_f, rho_f = as_field(domain, thickness), as_field(domain, density)
    cspace = normalize_space(constitutive_space) if constitutive_space is not None \
        else _default_constitutive_space(E_f, nu_f, t_f)
    scalar_space = cspace[:2]

    E_c = _to_scalar_space(E_f, scalar_space)
    nu_c = _to_scalar_space(nu_f, scalar_space)
    t_c = _to_scalar_space(t_f, scalar_space)
    A_v, B_v, D_v, As_v = assemble_abd_isotropic(E_c.coeffs, nu_c.coeffs, t_c.coeffs)

    A = Field(domain, (*scalar_space, (3, 3)), A_v)
    B = Field(domain, (*scalar_space, (3, 3)), B_v)
    D = Field(domain, (*scalar_space, (3, 3)), D_v)
    As = Field(domain, (*scalar_space, (2, 2)), As_v)
    return Material(A, B, D, As, thickness=t_f, density=rho_f, orientation=None,
                    E=E_f, nu=nu_f)


def laminate(domain, *, layup, density, orientation=None, shear_correction=0.833,
            constitutive_space=None) -> Material:
    """Composite material from a ply stack, via classical lamination theory.

    Parameters
    ----------
    domain : ShellDomain
    layup : Layup
        The ply stack (materials, angles, heights). Its angles and heights may be
        ``csdl.Variable``\\s, making the ABD differentiable in the layup.
    density : Field, float or array_like
        Mass density.
    orientation : Orientation, optional
        Rotates the laminate axes into each element's frame inside the shell form.
        Without it the laminate 0-degree axis is taken to be the element ``e0``.
    shear_correction : float, optional
        Multiplies the CLT transverse-shear stiffness. Default 0.833, the same
        factor :func:`isotropic` applies.
    constitutive_space : tuple, optional
        Space the ABD fields are broadcast onto. Defaults to ``("DG", d)`` where
        ``d`` is the highest degree among ``density`` and ``orientation``.

    Returns
    -------
    Material
        Carrying ``layup``, so :func:`~hermit.failure_index` and
        :func:`~hermit.failure_field` work on the resulting state.

    Notes
    -----
    One layup is broadcast uniformly to every cell; spatial variation of the stack
    itself is not part of this API. Use :func:`composite` with per-point ABD fields
    for that.

    Examples
    --------
    >>> layup = hm.Layup(ud, np.radians([0.0, 90.0, 0.0]), np.full(3, 0.002))
    >>> mat = hm.laminate(domain, layup=layup, density=1.6e3,
    ...                   orientation=hm.fiber_direction(domain, [1.0, 0.0, 0.0]))
    """
    from ._laminate import compute_clt

    rho_f = as_field(domain, density)
    fields_for_degree = [rho_f] + ([orientation.value] if orientation is not None else [])
    cspace = normalize_space(constitutive_space) if constitutive_space is not None \
        else _default_constitutive_space(*fields_for_degree)
    scalar_space = cspace[:2]
    n = domain.function_space(scalar_space).dofmap.index_map.size_local

    A, B, D, A_star = compute_clt(layup)
    # compute_clt (vendored verbatim from LamAD) returns A_star in the FSDT Voigt
    # (4, 5) = (yz, xz) ordering; the shell form's transverse-shear vector is
    # (xz, yz). Permute here, at the CLT seam, rather than touching compute_clt.
    P = csdl.Variable(value=np.array([[0.0, 1.0], [1.0, 0.0]]))
    A_star = P @ A_star @ P

    tile2 = lambda M, s: csdl.expand(M, (n, *s), action="ij->kij")
    thick = csdl.expand(csdl.reshape(layup.h, (1,)), (n,))

    A_f = Field(domain, (*scalar_space, (3, 3)), tile2(A, (3, 3)))
    B_f = Field(domain, (*scalar_space, (3, 3)), tile2(B, (3, 3)))
    D_f = Field(domain, (*scalar_space, (3, 3)), tile2(D, (3, 3)))
    As_f = Field(domain, (*scalar_space, (2, 2)), tile2(A_star * shear_correction, (2, 2)))
    t_f = Field(domain, scalar_space, thick)
    return Material(A_f, B_f, D_f, As_f, thickness=t_f, density=rho_f,
                    orientation=orientation, layup=layup)


def composite(domain, *, A, B, D, As, thickness, density, orientation=None,
             E=None, nu=None) -> Material:
    """Material from pre-computed ABD stiffness fields.

    Nothing is assembled here, so there is no ``constitutive_space`` to choose:
    each field keeps whatever space it was built on.

    Parameters
    ----------
    domain : ShellDomain
    A, B, D : Field or array_like
        ``(3, 3)`` membrane, coupling and bending stiffness, in laminate axes.
    As : Field or array_like
        ``(2, 2)`` transverse-shear stiffness, ``(xz, yz)`` ordering.
    thickness, density : Field, float or array_like
    orientation : Orientation, optional
        Rotates the supplied ABD into each element's frame inside the shell form.
    E, nu : Field, float or array_like, optional
        Supply these to enable the isotropic von Mises stress outputs.

    Returns
    -------
    Material

    Notes
    -----
    The result carries no ``layup``, so the Tsai-Wu outputs
    (:func:`~hermit.failure_index`, :func:`~hermit.failure_field`) do not work on it.
    Those need :func:`laminate`, which knows the ply stack the criterion is
    evaluated over.
    """
    A_f, B_f = as_field(domain, A), as_field(domain, B)
    D_f, As_f = as_field(domain, D), as_field(domain, As)
    t_f, rho_f = as_field(domain, thickness), as_field(domain, density)
    E_f = as_field(domain, E) if E is not None else None
    nu_f = as_field(domain, nu) if nu is not None else None
    return Material(A_f, B_f, D_f, As_f, thickness=t_f, density=rho_f,
                    orientation=orientation, E=E_f, nu=nu_f)


def thickness_only(domain, *, thickness, density) -> Material:
    """Thickness and density only, for a surrogate solve.

    ``A`` through ``As`` stay ``None``, so :func:`~hermit.solve` rejects this
    material: it is for a surrogate that carries its own constitutive model but
    still needs :func:`~hermit.mass` and :func:`~hermit.center_of_gravity`.

    Parameters
    ----------
    domain : ShellDomain
    thickness, density : Field, float or array_like

    Returns
    -------
    Material
    """
    return Material(None, None, None, None, thickness=as_field(domain, thickness),
                    density=as_field(domain, density))
