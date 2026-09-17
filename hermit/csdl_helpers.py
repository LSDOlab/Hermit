"""Native-CSDL preprocessing / postprocessing helpers.

Everything here is ordinary CSDL (auto-differentiated) -- gathers, reshapes, the
isotropic ABD assembly, the force->pressure L2 solve, stress aggregation, nodal
extraction. No FEniCSx, no custom operations.
"""

import warnings

import numpy as np
import csdl_alpha as csdl


def reorder(var: csdl.Variable, idx) -> csdl.Variable:
    """Gather ``var`` along axis 0 by integer index array ``idx`` (user -> FE ordering)."""
    return var[list(idx)]


def broadcast0(var: csdl.Variable, n: int) -> csdl.Variable:
    """Broadcast ``var`` along axis 0 to length ``n``.

    A scalar (``()`` / ``(1,)``) or a leading axis of length 1 is expanded to ``n``
    (differentiably); an axis already of length ``n`` is returned unchanged. Used to
    let material inputs be given as a single value instead of a per-node / per-cell
    array.
    """
    if var is None:
        return None
    nd = len(var.shape)
    tail = tuple(var.shape[1:]) if nd >= 1 else ()
    if nd >= 1 and var.shape[0] == n:
        return var
    if nd == 0 or var.shape[0] == 1:
        if tail == ():
            return csdl.expand(var.reshape((1,)), (n,))
        src = "".join(chr(ord("i") + k) for k in range(len(tail)))
        return csdl.expand(var.reshape(tail), (n, *tail), action=f"{src}->a{src}")
    raise ValueError(f"cannot broadcast leading axis {var.shape[0]} to {n}")


# -- material orientation (anisotropic ABD in the element local frame) --------

def orientation_angle(direction, frames) -> np.ndarray:
    """Per-cell angle (rad) from each cell's local e0 to the tangent-plane projection of
    ``direction`` (a global ``(3,)`` vector or a per-cell ``(n, 3)`` field). ``frames``:
    ``(n, 3, 3)`` local frames, rows e0, e1, e2. Pure numpy -- fibre layout is fixed."""
    n = len(frames)
    d = np.broadcast_to(np.asarray(direction, dtype=float), (n, 3))
    e0, e1, e2 = frames[:, 0], frames[:, 1], frames[:, 2]
    dt = d - np.einsum("ij,ij->i", d, e2)[:, None] * e2
    dt = dt / np.linalg.norm(dt, axis=1, keepdims=True)
    return np.arctan2(np.einsum("ij,ij->i", dt, e1), np.einsum("ij,ij->i", dt, e0))


def rotate_abd(A, B, D, As, theta):
    """Rotate the ABD stiffness ``(n,3,3)`` and transverse shear ``As`` ``(n,2,2)`` by a
    per-cell in-plane angle ``theta`` ``(n,)``. ``theta`` may be numpy (fixed fibre
    layout) or a ``csdl.Variable`` (differentiable, e.g. a fibre-angle design field).
    ``X_rot = Tsig_inv(theta) @ X @ Teps(theta)`` -- LamAD's CLT transform.

    ``As`` is in the shell's standard (xz, yz) transverse-shear ordering (the CLT seam
    in ``hm.laminate`` permutes it there), so its rotation uses
    ``r2t @ As @ r2 == R(theta)^T As R(theta)`` -- the ``Teps``-consistent convention
    (``r2 == R(theta)``, ``r2t == R(theta)^T``), matching
    ``hermit.fenics.kinematics.shear_rotation`` / ``congruent_transform``. (``r2 @ As
    @ r2t`` would instead be correct for a swapped ``(yz, xz)`` ordering.)"""
    is_var = isinstance(theta, csdl.Variable)
    c, s = (csdl.cos(theta), csdl.sin(theta)) if is_var else (np.cos(theta), np.sin(theta))
    n = c.shape[0]

    def mat(rows):  # nested list of (n,) -> (n, k, k)
        k = len(rows)
        if is_var:
            t = csdl.Variable(value=np.zeros((n, k, k)))
            for i in range(k):
                for j in range(k):
                    t = t.set(csdl.slice[:, i, j], rows[i][j])
            return t
        return np.stack([np.stack(r, -1) for r in rows], axis=1)

    t_eps = mat([[c**2, s**2, s * c],
                 [s**2, c**2, -s * c],
                 [-2 * s * c, 2 * s * c, c**2 - s**2]])
    t_sig_inv = mat([[c**2, s**2, -2 * s * c],
                     [s**2, c**2, 2 * s * c],
                     [s * c, -s * c, c**2 - s**2]])
    r2, r2t = mat([[c, s], [-s, c]]), mat([[c, -s], [s, c]])

    if is_var:
        e = lambda P, X: csdl.einsum(P, X, action="nij,njk->nik")
        return (e(e(t_sig_inv, A), t_eps), e(e(t_sig_inv, B), t_eps),
                e(e(t_sig_inv, D), t_eps), e(e(r2t, As), r2))
    av = lambda X: np.asarray(X.value if isinstance(X, csdl.Variable) else X)
    e = lambda P, X: np.einsum("nij,njk->nik", P, av(X))
    return tuple(csdl.Variable(value=v) for v in (
        e(e(t_sig_inv, A), t_eps), e(e(t_sig_inv, B), t_eps),
        e(e(t_sig_inv, D), t_eps), e(e(r2t, As), r2)))


def _tensor(components, shape):
    """Stack a dict ``{(i, j): scalar-or-(n,) Variable}`` into an ``(n, *shape)`` Variable."""
    n = next(iter(components.values())).shape[0]
    t = csdl.Variable(value=np.zeros((n, *shape)))
    for (i, j), comp in components.items():
        t = t.set(csdl.slice[:, i, j], comp)
    return t


def assemble_abd_isotropic(E, nu, thickness):
    """Isotropic single-layer A, B, D (n,3,3) and As (n,2,2) from E, nu, thickness (n,).

    Port of ``MaterialInputFactory.from_isotropic`` (femo dev_coupling).
    """
    E = _as_1d(E, thickness.shape)
    nu = _as_1d(nu, thickness.shape)
    pre = E / (1.0 - nu * nu)
    c11, c12, c33 = pre, pre * nu, pre * 0.5 * (1.0 - nu)
    shear = 0.833 * (E / (2.0 * (1.0 + nu))) * thickness
    z = 0.0 * thickness
    t, t3 = thickness, thickness**3 / 12.0
    A = _tensor({(0, 0): t * c11, (0, 1): t * c12, (1, 0): t * c12, (1, 1): t * c11,
                 (2, 2): t * c33, (0, 2): z, (2, 0): z, (1, 2): z, (2, 1): z}, (3, 3))
    D = _tensor({(0, 0): t3 * c11, (0, 1): t3 * c12, (1, 0): t3 * c12, (1, 1): t3 * c11,
                 (2, 2): t3 * c33, (0, 2): z, (2, 0): z, (1, 2): z, (2, 1): z}, (3, 3))
    B = csdl.Variable(value=np.zeros((thickness.shape[0], 3, 3)))
    As = _tensor({(0, 0): shear, (1, 1): shear, (0, 1): z, (1, 0): z}, (2, 2))
    return A, B, D, As


def _as_1d(v, shape):
    if isinstance(v, csdl.Variable):
        if v.shape == shape:
            return v
        if v.shape in ((), (1,)):
            return csdl.expand(v.reshape(()) if v.shape == (1,) else v, shape)
        raise ValueError(f"cannot broadcast {v.shape} -> {shape}")
    arr = np.asarray(v, dtype=float)
    if arr.shape == shape:
        return csdl.Variable(value=arr)
    return csdl.Variable(value=np.full(shape, arr.reshape(-1)[0]))


def build_pressure(loads, maps, vf_size: int):
    """Combine nodal pressure + (force -> pressure) into the VF-dof RHS vector."""
    parts = []
    if loads.nodal_pressure is not None:
        parts.append(csdl.reshape(reorder(loads.nodal_pressure, maps.pressure_idx), (-1,)))
    if loads.nodal_forces is not None:
        fvec = csdl.reshape(reorder(loads.nodal_forces, maps.pressure_idx), (-1,))
        parts.append(csdl.solve_linear(maps.force_to_pressure.toarray(), fvec))
    if not parts:
        return csdl.Variable(value=np.zeros(vf_size))
    out = parts[0]
    for extra in parts[1:]:
        out = out + extra
    return out


def build_moment(loads, maps, vf_size: int):
    if loads.nodal_moments is None:
        return csdl.Variable(value=np.zeros(vf_size))
    return csdl.reshape(reorder(loads.nodal_moments, maps.pressure_idx), (-1,))


def resolve_mesh_nodes(reference_nodes, node_disp=None, mesh_nodes=None):
    """Return (mesh_nodes_variable, geometry_is_differentiable)."""
    if node_disp is not None and mesh_nodes is not None:
        raise ValueError("give either node_disp or mesh_nodes, not both")
    ref = csdl.Variable(value=np.asarray(reference_nodes))
    if mesh_nodes is not None:
        v = mesh_nodes if isinstance(mesh_nodes, csdl.Variable) else csdl.Variable(value=np.asarray(mesh_nodes))
        return v, True
    if node_disp is None:
        return ref, False
    return ref + node_disp, True


# Additive guard on the p-norm, and nothing more: ``P**(1/rho)`` has an infinite
# slope at ``P = 0``, so a p-norm that has underflowed all the way to zero would put
# inf/NaN into the adjoint. 1e-300 keeps ``(P + g)**(1/rho - 1)`` finite for every
# rho >= 1 while sitting ~150 decades below the point where the degeneracy warning
# below fires, so it cannot quietly *become* the answer.
#
# This replaces femo's ``csdl.absolute(P, rho=50)``, which was doing exactly that.
# ``csdl.absolute`` is a *smooth* abs (log-sum-exp of +-x), so its value at x -> 0 is
# not 0 but ln(2)/50 = 1.386e-2; with the femo defaults m=1e-6, rho=100 the p-norm of
# the cantilever fixture is 1.3e-186 -- utterly negligible against that floor -- so
# the aggregate was the *constant* 1e6 * (ln2/50)**0.01 = 958117.0189 whatever the
# displacement field was. The softabs was never needed either: ``von_mises`` is a sqrt, so
# ``(m*vm)**rho`` is pointwise non-negative for m > 0 and any rho, and the area
# integral of it over positive quadrature weights is provably >= 0. It could only
# ever impose its floor.
_PNORM_GUARD = 1.0e-300

# Warn once the p-norm has spent more than half of double precision's ~308-decade
# exponent budget. That is the honest measure of a bad ``m``: the p-norm is
# ``(m*vm_ks)**rho``, so log10(P) tells us directly how close ``(m*vm)**rho`` is to
# flushing to zero, and it is rho-aware for free (at rho=100 this trips at
# m*max(vm) < 0.030, at rho=200 at m*max(vm) < 0.17).
_PNORM_DEGENERATE_LOG10 = -154.0


def _warn_if_degenerate(pnorm_stress, m, rho):
    """Warn when ``m`` is scaled so badly that the p-norm is heading for underflow.

    Deliberately cheap and non-differentiable: it reads the value of the p-norm that
    has *already* been assembled -- no second assembly, nothing added to the recorded
    graph, no effect on the derivative. Silently skipped when the value is not
    available (a non-inline recorder builds the graph without evaluating it).
    """
    value = getattr(pnorm_stress, "value", pnorm_stress)
    if value is None:
        return
    p = float(np.ravel(np.asarray(value, dtype=float))[0])
    if not np.isfinite(p):
        return
    if p > 0.0 and np.log10(p) > _PNORM_DEGENERATE_LOG10:
        return
    # P**(1/rho) *is* m * (the KS stress), so both the diagnosis and the suggested
    # rescaling come out of the number already in hand.
    scaled = p ** (1.0 / rho) if p > 0.0 else 0.0
    if scaled > 0.0:
        lead = ("more than half of double precision's exponent range below 1, so "
                "(m*vm)**rho is one modest change of load or rho away from flushing to "
                "zero and leaving the aggregate equal to the underflow guard")
        detail = f"m*max(vm) is only ~{scaled:.2e}; rescale to m ~ {m / scaled:.3e}"
    else:
        lead = ("not positive: it has flushed to zero, so this number is the underflow "
                "guard and not the stress at all")
        detail = "m*max(vm) is far below 1; rescale m"
    warnings.warn(
        f"degenerate stress-aggregate scaling: with m={m:g}, rho={rho:g} the p-norm "
        f"is {p:.3e} -- {lead}. m is a problem scaling, not a smoothing knob: the KS "
        f"aggregate is only meaningful when m*max(vm) ~ 1, and here {detail} "
        f"(hm.stress_scaling(state) computes it from one stress-field pass).",
        stacklevel=3,  # -> the hm.aggregated_stress call inside outputs.py
    )


def aggregate(pnorm_stress, m, rho):
    """Aggregated stress ``(1/m) * (pnorm + guard)**(1/rho)``.

    ``m`` is a problem-specific *scaling*, not a smoothing knob: the aggregate is
    mathematically invariant under ``m`` (it cancels), and only the finite exponent
    range of float64 breaks that invariance -- which it does violently once
    ``m*max(vm)`` strays far from 1. ``_warn_if_degenerate`` says so out loud.
    """
    _warn_if_degenerate(pnorm_stress, m, rho)
    return 1.0 / m * (pnorm_stress + _PNORM_GUARD) ** (1.0 / rho)


def extract_nodal(disp_vec, sparse_map, geom_shape, reverse_node_idx):
    """(state dof vec) -> (n_nodes, 3) nodal field, reordered back to user node ordering."""
    flat = csdl.sparse.matvec(sparse_map, disp_vec)
    mat = csdl.transpose(csdl.reshape(flat, (geom_shape[1], geom_shape[0])))
    return mat[list(reverse_node_idx), :]
