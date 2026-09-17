"""Through-thickness sign gates: ``eps(xi) = eps_mid - xi*kappa``.

The shell kinematics are ``u(xi) = u_mid - xi*(E2 x theta)`` (``hermit/fenics/stress.py``),
while ``A/B/D`` come from CLT, written for ``eps(z) = eps_0 + z*kappa_clt``. So
``kappa_clt = -kappa``: only the ``B`` cross term flips (``A``/``D`` are even in ``xi``),
and every ply sits at ``-z``, not ``+z``.

Both gates need an **unsymmetric** layup -- ``B = 0`` for an isotropic single layer and
for every symmetric layup, which is all the rest of the suite contains. The references
here are plain numpy (ply ``Q``, rotated and integrated by hand); no second hermit path.
"""

import numpy as np

import csdl_alpha as csdl

FACES = (-0.5, 0.5)
ANGLES_DEG = np.array([0.0, 45.0, 90.0, 0.0])       # unsymmetric angles ...
HEIGHTS = np.array([0.004, 0.002, 0.006, 0.003])    # ... and unequal ply thicknesses


def _ud():
    from caddee_materials import TransverseMaterial

    m = TransverseMaterial(name="ud", EA=138e9, ET=10e9, vA=0.34, vT=0.4, GA=7e9, density=1.6e3)
    m.set_strength(F1t=1500e6, F1c=1200e6, F2t=50e6, F2c=200e6, F12=70e6, F23=40e6)
    return m


def _layup(material=None):
    from hermit._laminate import Layup

    mat = material if material is not None else _ud()
    angles = csdl.Variable(value=np.radians(ANGLES_DEG), name="ply_angles")
    return Layup(mat, angles, HEIGHTS, num_plies=len(ANGLES_DEG))


def _ply_z():
    """Ply interface coordinates, bottom to top, about the mid-surface."""
    return np.concatenate([[-HEIGHTS.sum() / 2],
                           -HEIGHTS.sum() / 2 + np.cumsum(HEIGHTS)])


def _t_eps(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c * c, s * s, s * c],
                     [s * s, c * c, -s * c],
                     [-2 * s * c, 2 * s * c, c * c - s * s]])


def _q_bar(Q, theta):
    """Ply ``Q`` (material axes) rotated to laminate axes -- numpy, by hand."""
    c, s = np.cos(theta), np.sin(theta)
    t_sig_inv = np.array([[c * c, s * s, -2 * s * c],
                          [s * s, c * c, 2 * s * c],
                          [s * c, -s * c, c * c - s * s]])
    return t_sig_inv @ Q @ _t_eps(theta)


def _rotated_ply_q(layup):
    """Per-ply rotated ``Q``, from ``clt.calc_q``'s ply-level constants only."""
    from hermit._laminate.clt import calc_q, read_materials

    q = calc_q(read_materials(layup.materials))[0].value
    return [_q_bar(q[k], np.radians(ANGLES_DEG[k])) for k in range(len(ANGLES_DEG))]


# --- gate 1: the B cross-term sign in the shell form ------------------------

def test_membrane_bending_energy_matches_numpy_through_thickness(plate_mesh, recorder):
    import ufl
    from dolfinx.fem import Expression, Function, assemble_scalar, form, functionspace

    from hermit import _compat
    from hermit.domain import ShellDomain
    from hermit.fenics.kinematics import voigt2D
    from hermit.fenics.shell_pde import ShellPDE
    from hermit._laminate import compute_clt

    layup = _layup()
    A, B, D, As = (X.value for X in compute_clt(layup))
    assert np.abs(B).max() > 0, "layup is symmetric -- the gate would be vacuous"

    # independent numpy ABD from the ply Q's; pins the rotation convention so an
    # energy mismatch below can only be the form's cross-term sign.
    q_bar = _rotated_ply_q(layup)
    z = _ply_z()
    An, Bn, Dn = (np.zeros((3, 3)) for _ in range(3))
    for k, Q in enumerate(q_bar):
        An += Q * (z[k + 1] - z[k])
        Bn += Q * (z[k + 1] ** 2 - z[k] ** 2) / 2
        Dn += Q * (z[k + 1] ** 3 - z[k] ** 3) / 3
    for got, want in ((A, An), (B, Bn), (D, Dn)):
        assert np.abs(got - want).max() <= 1e-10 * np.abs(want).max()

    pde = ShellPDE(plate_mesh, element="CG2CG1", element_wise_material=True)
    for fn, M in ((pde.A, A), (pde.B, B), (pde.D, D)):
        fn.x.array[:] = np.tile(M.reshape(-1), fn.x.array.size // 9)
    pde.As.x.array[:] = np.tile(As.reshape(-1), pde.As.x.array.size // 4)

    # flat mesh in the xy-plane -> deterministic local frame (E0, E1, E2) = (x, y, z)
    frames = ShellDomain(plate_mesh, element="CG2CG1").local_frames()
    assert np.abs(frames - np.eye(3)).max() < 1e-13

    # manufactured state: u_mid and theta linear in (x, y) -> eps, kappa cell-constant
    w = Function(pde.W)
    for sub, fn in ((0, lambda x: np.vstack([1e-3 * x[0] + 5e-4 * x[1],
                                             -7e-4 * x[0] + 3e-4 * x[1],
                                             2e-4 * x[0]])),
                    (1, lambda x: np.vstack([2e-2 * x[0] + 1e-2 * x[1],
                                             -3e-2 * x[0] + 1.5e-2 * x[1],
                                             np.zeros_like(x[0])]))):
        V, dofs = pde.W.sub(sub).collapse()
        f = Function(V)
        f.interpolate(fn)
        w.x.array[dofs] = f.x.array

    em = pde.elastic_model(w=w)
    VD = functionspace(plate_mesh, ("DG", 0, (3,)))

    def cellwise(expr):
        f = Function(VD)
        f.interpolate(Expression(expr, _compat.interpolation_points(VD)))
        return f.x.array.reshape(-1, 3)

    eps_c, kappa_c = cellwise(voigt2D(em.eps)), cellwise(voigt2D(em.kappa))
    assert np.abs(eps_c - eps_c[0]).max() < 1e-14
    assert np.abs(kappa_c - kappa_c[0]).max() < 1e-14
    eps, kappa = eps_c[0], kappa_c[0]
    assert np.abs(eps).max() > 0 and np.abs(kappa).max() > 0

    # membrane + bending only: shear / drilling are irrelevant here
    got = assemble_scalar(form(em.membrane_energy() + em.bending_energy()))
    area = assemble_scalar(form(1.0 * ufl.dx(domain=plate_mesh)))

    # 0.5 * area * int (eps - xi*kappa)^T Q(xi) (eps - xi*kappa) dxi, ply by ply,
    # exact (the integrand is quadratic in xi)
    want = 0.0
    for k, Q in enumerate(q_bar):
        z0, z1 = z[k], z[k + 1]
        want += ((eps @ Q @ eps) * (z1 - z0)
                 - (eps @ Q @ kappa) * (z1 ** 2 - z0 ** 2)
                 + (kappa @ Q @ kappa) * (z1 ** 3 - z0 ** 3) / 3)
    want *= 0.5 * area

    cross_frac = abs(2 * eps @ Bn @ kappa) * 0.5 * area / abs(want)
    assert cross_frac > 0.01, f"cross term only {cross_frac:.2%} of the energy"
    assert abs(got - want) <= 1e-10 * abs(want), (
        f"membrane+bending energy {got!r} vs numpy {want!r} "
        f"(rel {abs(got - want) / abs(want):.3e}, cross term {cross_frac:.2%})")


# --- gate 2: tsai_wu_field ply position ------------------------------------

def _tsai_wu_numpy(eps_lam, shear_lam, theta, p):
    """Tsai-Wu index for one ply, numpy, from laminate-frame strains."""
    c, s = np.cos(theta), np.sin(theta)
    e1, e2, g12 = (eps_lam @ _t_eps(theta).T).T
    E1, E2, v12, G12 = p["E1"], p["E2"], p["v12"], p["G12"]
    den = 1.0 - v12 * (v12 * E2 / E1)
    q11, q12, q22 = E1 / den, v12 * E2 / den, E2 / den
    s1, s2, t12 = q11 * e1 + q12 * e2, q12 * e1 + q22 * e2, G12 * g12
    g13 = c * shear_lam[:, 0] + s * shear_lam[:, 1]
    g23 = -s * shear_lam[:, 0] + c * shear_lam[:, 1]

    Xt, Xc, Yt, Yc, S12, S13, S23 = (p[k] for k in ("Xt", "Xc", "Yt", "Yc",
                                                    "S12", "S13", "S23"))
    f1, f2 = 1.0 / Xt - 1.0 / Xc, 1.0 / Yt - 1.0 / Yc
    f11, f22, f66 = 1.0 / (Xt * Xc), 1.0 / (Yt * Yc), 1.0 / S12 ** 2
    idx = (f1 * s1 + f2 * s2 + f11 * s1 ** 2 + f22 * s2 ** 2
           - np.sqrt(f11 * f22) * s1 * s2 + f66 * t12 ** 2)
    if S13 and p["G13"]:
        idx = idx + (1.0 / S13 ** 2) * (p["G13"] * g13) ** 2
    if S23 and p["G23"]:
        idx = idx + (1.0 / S23 ** 2) * (p["G23"] * g23) ** 2
    return idx


def _tsai_wu_field_numpy(mid, curv, shear, p, sign):
    z = _ply_z()
    cols = []
    for k, theta in enumerate(np.radians(ANGLES_DEG)):
        for frac in FACES:
            zk = 0.5 * (z[k] + z[k + 1]) + frac * HEIGHTS[k]
            cols.append(_tsai_wu_numpy(mid + sign * zk * curv, shear, theta, p))
    return np.stack(cols, axis=1)


def test_tsai_wu_field_evaluates_plies_at_minus_z(recorder):
    from hermit._laminate.failure import ply_properties, tsai_wu_field

    rng = np.random.default_rng(3)
    mid = rng.normal(scale=1e-3, size=(5, 3))
    curv = rng.normal(scale=1e-1, size=(5, 3))
    shear = rng.normal(scale=1e-3, size=(5, 2))

    mat = _ud()
    got = tsai_wu_field(csdl.Variable(value=mid), csdl.Variable(value=curv),
                        csdl.Variable(value=shear), _layup(mat), faces=FACES).value
    p = ply_properties(mat)
    minus = _tsai_wu_field_numpy(mid, curv, shear, p, -1.0)
    plus = _tsai_wu_field_numpy(mid, curv, shear, p, +1.0)

    scale = np.abs(minus).max()
    assert got.shape == minus.shape
    # the layup is unsymmetric enough that mirroring the plies changes the answer
    assert np.abs(plus - minus).max() > 0.01 * scale
    assert np.abs(got - minus).max() <= 1e-12 * scale
    assert np.abs(got - plus).max() > 0.01 * scale
