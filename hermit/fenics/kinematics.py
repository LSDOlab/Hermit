"""Basis transformations and Voigt notation for the R-M shell.

Ported from ``femo_alpha/rm_shell/linear_shell_fenicsx/kinematics.py`` (dev_coupling).
Unchanged across DOLFINx 0.9 / 0.11 -- these are pure UFL. Note there is *no* ``gradx`` / ``J`` /
``F``: geometry sensitivity comes from differentiating w.r.t. ``SpatialCoordinate``
directly, so all gradients here are plain ``ufl.grad`` on the (possibly moved) mesh.
"""

from ufl import (
    CellNormal,
    Jacobian,
    as_matrix,
    as_tensor,
    as_vector,
    cos,
    cross,
    dot,
    grad,
    indices,
    sin,
    sqrt,
    sym,
)


def unit(v):
    """Normalize the vector ``v``."""
    return v / sqrt(dot(v, v))


def local_basis_inplane(mesh):
    """Local orthonormal shell basis (E0, E1 in-plane; E2 = element normal).

    E0 is aligned with the 0-th parametric coordinate direction.
    """
    E2 = CellNormal(mesh)
    A0 = as_vector([Jacobian(mesh)[j, 0] for j in range(3)])
    E0 = unit(A0)
    E1 = cross(E2, E0)
    return E0, E1, E2


def global_to_local_inplane(E0, E1):
    """2x3 change-of-basis matrix T; T[i, j] is the j-th component of the i-th basis vector."""
    return as_matrix([[E0[i] for i in range(3)], [E1[i] for i in range(3)]])


def gradv_local(gradv_global, T):
    """In-plane components of a rank-2 tensor in the local orthonormal frame."""
    i, j, k, ll = indices(4)
    return as_tensor(T[i, k] * gradv_global[k, ll] * T[j, ll], (i, j))


def membrane_strain(u_mid, E01):
    """Mid-surface membrane strain in the local in-plane frame."""
    return sym(gradv_local(grad(u_mid), E01))


def constant_normal_bending_curvature(theta, E01):
    """Shallow-shell curvature with the cell normal held constant.

    Only the rotation gradient enters. In particular, a rigid rotation is a
    constant ``theta`` and therefore has zero curvature structurally, including on
    a warped quadrilateral cell.
    """
    tg = gradv_local(grad(theta), E01)
    off = 0.5 * (tg[0, 0] - tg[1, 1])
    return as_tensor([[-tg[1, 0], off], [off, tg[0, 1]]])


def voigt2D(T, strain=True):
    """2x2 symmetric tensor -> Voigt vector. ``strain`` doubles the shear component."""
    fac = 2.0 if strain else 1.0
    return as_vector([T[0, 0], T[1, 1], fac * T[0, 1]])


def voigt3D(T, strain=True):
    """3x3 symmetric tensor -> length-6 Voigt vector ``[xx, yy, zz, yz, xz, xy]``.
    ``strain`` doubles the three shear components (engineering convention, matching
    :func:`voigt2D`)."""
    fac = 2.0 if strain else 1.0
    return as_vector([T[0, 0], T[1, 1], T[2, 2],
                      fac * T[1, 2], fac * T[0, 2], fac * T[0, 1]])


def strain2D_local_to_global(strain_local, T):
    """2x2 local in-plane strain -> 3x3 global Cartesian tensor."""
    i, j, k, ll = indices(4)
    return as_tensor(T[k, i] * strain_local[k, ll] * T[ll, j], (i, j))


def vec2D_local_to_global(v_local, T):
    """length-2 local in-plane vector -> length-3 global Cartesian vector."""
    i, j = indices(2)
    return as_tensor(v_local[i] * T[i, j], (j,))


# -- fibre orientation: element frame -> laminate frame -----------------------
# A laminate's A/B/D/As come out of CLT in the laminate's own material axes; the
# shell form works in each element's local in-plane frame (E0, E1, E2 above). The two
# differ by an in-plane angle theta. Rather than rotate the ABD upstream, the strain
# is rotated into laminate axes inside the form and contracted there -- equivalent
# (bit-for-bit, modulo float-op ordering) to rotating the ABD into the element frame by the
# congruence transform below and using it unchanged (``elastic_model.ElasticModel``
# does the latter, since it lets every other form -- energy, drilling stabilization --
# stay untouched).

def orientation_cos_sin(fiber_angle, fiber_direction, E0, E1, E2):
    """``(cos theta, sin theta)`` of the in-plane angle from the element ``E0`` to the
    fibre direction -- from ``fiber_angle`` directly, or projected from a global
    ``fiber_direction`` vector (no ``atan2``: it is more expensive and puts a branch
    cut at +/-pi right in the differentiated path). Exactly one of ``fiber_angle`` /
    ``fiber_direction`` must be given.

    Degenerate only when ``fiber_direction`` is parallel to the shell normal ``E2``
    (the tangent-plane projection vanishes) -- a UFL expression cannot raise, so
    callers must guard that numerically, with concrete values, before building the
    form.
    """
    if (fiber_angle is None) == (fiber_direction is None):
        raise ValueError("give exactly one of fiber_angle, fiber_direction")
    if fiber_angle is not None:
        return cos(fiber_angle), sin(fiber_angle)
    d_t = fiber_direction - dot(fiber_direction, E2) * E2
    e_hat = unit(d_t)
    return dot(e_hat, E0), dot(e_hat, E1)


def strain_rotation(c, s):
    """Teps(theta) -- 3x3 engineering-Voigt in-plane strain/curvature rotation,
    element -> laminate axes: ``eps_L = Teps @ voigt2D(eps)``. ``Teps^T == Tsig^-1``
    for this convention (the identity ``hermit.csdl_helpers.rotate_abd`` also uses)."""
    return as_matrix([[c**2, s**2, s * c],
                      [s**2, c**2, -s * c],
                      [-2 * s * c, 2 * s * c, c**2 - s**2]])


def shear_rotation(c, s):
    """R(theta) -- 2x2 transverse-shear rotation, element -> laminate axes, standard
    FSDT Voigt (xz, yz) ordering: ``gam_L = R @ gamma``."""
    return as_matrix([[c, s], [-s, c]])


def congruent_transform(T, X):
    """``T^T X T`` for square UFL matrices (``T`` orthogonal, ``X`` symmetric) --
    rotates a laminate-axes constitutive matrix into the element frame,
    ``A_element = Teps^T A Teps`` (and ``As_element = R^T As R``). Index notation
    (not ``ufl.transpose(T) * X * T``) so it is unambiguous for any square size."""
    i, j, k, ll = indices(4)
    return as_tensor(T[k, i] * X[k, ll] * T[ll, j], (i, j))
