"""Tsai-Wu ply failure.

Two layers:

* the per-point reference implementation vendored from ``lamad.src.failure``
  (``ply_local_stress_from_strain`` / ``tsai_wu_failure_index`` /
  ``tsai_wu_stress_ratio``) -- kept mostly for testing and single-point use;
* the vectorised-over-cells version Hermit's post-processing actually calls
  (``ply_properties`` / ``tsai_wu_field`` / ``aggregate_failure``), which mirrors
  the per-point quadratic form. ``tests/test_composite_failure.py`` pins the two
  together cell-by-cell.

Pure CSDL. Given the per-cell mid-surface strain / curvature / transverse-shear
fields (from ``ShellFieldFormsOp`` / ``strain_fields``) and a ``Layup``,
``tsai_wu_field`` evaluates the index in every ply at the ply faces;
``aggregate_failure`` reduces it to a single smooth-max constraint (``< 1`` safe).
"""

import numpy as np
import csdl_alpha as csdl

from .clt import calc_t_eps


# --------------------------------------------------------------------------
# per-point reference implementation (vendored from lamad.src.failure)
# --------------------------------------------------------------------------

def _extract_ply_property(ply_properties, key, default=None, required=False):
    if not isinstance(ply_properties, dict):
        raise TypeError("ply_properties must be a dictionary of ply constants/strengths")
    if key in ply_properties:
        return ply_properties[key]
    if required:
        raise KeyError(f"Missing required ply property: {key}")
    return default


def _in_plane_local_strain(strain_in_plane_laminate, ply_angle):
    """Transform laminate in-plane strains [ex, ey, gxy] to local [e1, e2, g12]."""
    t_eps = calc_t_eps(ply_angle)
    return t_eps @ strain_in_plane_laminate.reshape((3, 1))


def _transverse_shear_local_strain(strain_transverse_shear_laminate, ply_angle):
    """Transform laminate transverse shear [gxz, gyz] to local [g13, g23]."""
    c = csdl.cos(ply_angle)
    s = csdl.sin(ply_angle)
    gxz = strain_transverse_shear_laminate[0]
    gyz = strain_transverse_shear_laminate[1]
    g13 = c * gxz + s * gyz
    g23 = -s * gxz + c * gyz
    return csdl.blockmat([[g13.reshape((1, 1))], [g23.reshape((1, 1))]])


def ply_local_stress_from_strain(strain_in_plane_laminate,
                                 strain_transverse_shear_laminate,
                                 ply_angle, ply_properties):
    """Local ply stresses from laminate strains and ply properties.

    ``ply_properties`` needs ``E1, E2, v12, G12`` (and optionally ``G13, G23``).
    Returns ``(sigma_local (3,1), tau_transverse_local (2,1))``.
    """
    E1 = _extract_ply_property(ply_properties, "E1", required=True)
    E2 = _extract_ply_property(ply_properties, "E2", required=True)
    v12 = _extract_ply_property(ply_properties, "v12", required=True)
    G12 = _extract_ply_property(ply_properties, "G12", required=True)
    G13 = _extract_ply_property(ply_properties, "G13", default=0.0)
    G23 = _extract_ply_property(ply_properties, "G23", default=0.0)

    eps_local = _in_plane_local_strain(strain_in_plane_laminate, ply_angle)
    gamma_local = _transverse_shear_local_strain(strain_transverse_shear_laminate, ply_angle)

    v21 = v12 * E2 / E1
    denom = 1.0 - v12 * v21
    q11 = E1 / denom
    q12 = v12 * E2 / denom
    q22 = E2 / denom

    sigma_1 = q11 * eps_local[0] + q12 * eps_local[1]
    sigma_2 = q12 * eps_local[0] + q22 * eps_local[1]
    tau_12 = G12 * eps_local[2]
    tau_13 = G13 * gamma_local[0]
    tau_23 = G23 * gamma_local[1]

    sigma_local = csdl.blockmat([[sigma_1.reshape((1, 1))],
                                 [sigma_2.reshape((1, 1))],
                                 [tau_12.reshape((1, 1))]])
    tau_transverse_local = csdl.blockmat([[tau_13.reshape((1, 1))],
                                          [tau_23.reshape((1, 1))]])
    return sigma_local, tau_transverse_local


def tsai_wu_failure_index(strain_in_plane_laminate, strain_transverse_shear_laminate,
                          ply_angle, ply_properties, FoS=1.0):
    """Tsai-Wu failure index for one ply from laminate-frame strains (``>= 1`` fails).

    ``ply_properties`` needs ``E1, E2, v12, G12, Xt, Xc, Yt, Yc, S12`` and
    optionally ``G13, G23, S13, S23``.
    """
    Xt = _extract_ply_property(ply_properties, "Xt", required=True)
    Xc = _extract_ply_property(ply_properties, "Xc", required=True)
    Yt = _extract_ply_property(ply_properties, "Yt", required=True)
    Yc = _extract_ply_property(ply_properties, "Yc", required=True)
    S12 = _extract_ply_property(ply_properties, "S12", required=True)
    S13 = _extract_ply_property(ply_properties, "S13", default=None)
    S23 = _extract_ply_property(ply_properties, "S23", default=None)

    sigma_local, tau_transverse_local = ply_local_stress_from_strain(
        strain_in_plane_laminate, strain_transverse_shear_laminate, ply_angle, ply_properties)

    sigma_1 = sigma_local[0] * FoS
    sigma_2 = sigma_local[1] * FoS
    tau_12 = sigma_local[2] * FoS
    tau_13 = tau_transverse_local[0] * FoS
    tau_23 = tau_transverse_local[1] * FoS

    f1 = 1.0 / Xt - 1.0 / Xc
    f2 = 1.0 / Yt - 1.0 / Yc
    f11 = 1.0 / (Xt * Xc)
    f22 = 1.0 / (Yt * Yc)
    f66 = 1.0 / (S12**2)
    f12 = -0.5 * csdl.sqrt(f11 * f22)

    index = (f1 * sigma_1 + f2 * sigma_2
             + f11 * sigma_1**2 + f22 * sigma_2**2
             + 2.0 * f12 * sigma_1 * sigma_2 + f66 * tau_12**2)
    if S13 is not None:
        index = index + (1.0 / S13**2) * tau_13**2
    if S23 is not None:
        index = index + (1.0 / S23**2) * tau_23**2
    return index


def tsai_wu_stress_ratio(strain_in_plane_laminate, strain_transverse_shear_laminate,
                         ply_angle, ply_properties):
    """Tsai-Wu stress ratio (a safety-factor-like quantity)."""
    Xt = _extract_ply_property(ply_properties, "Xt", required=True)
    Xc = _extract_ply_property(ply_properties, "Xc", required=True)
    Yt = _extract_ply_property(ply_properties, "Yt", required=True)
    Yc = _extract_ply_property(ply_properties, "Yc", required=True)
    S12 = _extract_ply_property(ply_properties, "S12", required=True)
    S13 = _extract_ply_property(ply_properties, "S13", default=None)
    S23 = _extract_ply_property(ply_properties, "S23", default=None)

    sigma_local, tau_transverse_local = ply_local_stress_from_strain(
        strain_in_plane_laminate, strain_transverse_shear_laminate, ply_angle, ply_properties)

    sigma_1 = sigma_local[0]
    sigma_2 = sigma_local[1]
    tau_12 = sigma_local[2]
    tau_13 = tau_transverse_local[0]
    tau_23 = tau_transverse_local[1]

    f1 = 1.0 / Xt - 1.0 / Xc
    f2 = 1.0 / Yt - 1.0 / Yc
    f11 = 1.0 / (Xt * Xc)
    f22 = 1.0 / (Yt * Yc)
    f66 = 1.0 / (S12**2)
    f12 = -0.5 * csdl.sqrt(f11 * f22)

    a = f11 * sigma_1**2 + f22 * sigma_2**2 + f66 * tau_12**2 + 2.0 * f12 * sigma_1 * sigma_2
    b = f1 * sigma_1 + f2 * sigma_2
    if S13 is not None:
        a = a + (1.0 / S13**2) * tau_13**2
    if S23 is not None:
        a = a + (1.0 / S23**2) * tau_23**2
    return (-b + csdl.sqrt(b**2 + 4.0 * a)) / (2.0 * a)


# --------------------------------------------------------------------------
# vectorised-over-cells version used by Hermit's post-processing
# --------------------------------------------------------------------------

def ply_properties(material) -> dict:
    """caddee_materials ``Material`` -> the constants the Tsai-Wu index needs.

    ``E1, E2, v12, G12`` (elastic) and ``Xt, Xc, Yt, Yc, S12`` (strengths), plus the
    interlaminar pairs ``G13, S13`` and ``G23, S23``. Strengths require
    ``material.set_strength(...)`` to have been called.
    """
    from caddee_materials import IsotropicMaterial, TransverseMaterial

    strength = getattr(material, "strength", None)
    if strength is None:
        raise ValueError(
            f"ply material {getattr(material, 'name', material)!r} has no strength data; "
            f"call material.set_strength(F1t, F1c, F2t, F2c, F12, F23) before building the layup"
        )
    # caddee_materials' set_strength(F1t, F1c, F2t, F2c, F12, F23) stores the shear row
    # as [F12, F12, F23] -- F12 twice, because for a transversely isotropic ply the 1-3
    # plane contains the fibre exactly as the 1-2 plane does, so S13 == S12. s[2, 1] is
    # that S13; skipping it would silently drop the tau_13 term from the index.
    s = np.asarray(strength, dtype=float)   # [[Xt,Yt,Yt],[Xc,Yc,Yc],[S12,S13,S23]]

    if isinstance(material, TransverseMaterial):
        E1, E2, v12, v23, G12 = (float(x) for x in material.get_constants())
        # The same transverse isotropy on the stiffness side: GA is the shear modulus of
        # any plane containing the fibre, so G13 == G12 == GA, while 2-3 is the isotropic
        # plane and gets G23 = E2 / 2(1 + v23). clt.calc_q makes exactly this split for
        # the FSDT out-of-plane block, and tests/test_composite.py pins it.
        G13 = G12
        G23 = E2 / (2.0 * (1.0 + v23))
    elif isinstance(material, IsotropicMaterial):
        E, nu, G = (float(x) for x in material.get_constants())
        E1 = E2 = E
        v12 = nu
        G12 = G13 = G23 = G       # one shear modulus, every plane
    else:  # pragma: no cover
        raise TypeError(f"unsupported ply material type {type(material).__name__}")

    return dict(E1=E1, E2=E2, v12=v12, G12=G12, G13=G13, G23=G23,
                Xt=s[0, 0], Xc=s[1, 0], Yt=s[0, 1], Yc=s[1, 1],
                S12=s[2, 0], S13=s[2, 1], S23=s[2, 2])


def _tsai_wu_index(eps_lam, shear_lam, angle, p):
    """Vectorised Tsai-Wu index. ``eps_lam`` (nel,3) [ex,ey,gxy], ``shear_lam`` (nel,2)
    [gxz,gyz] in the laminate local frame; ``angle`` a scalar Variable (rad); ``p`` a
    ``ply_properties`` dict. Returns (nel,)."""
    c0, s0 = csdl.cos(angle), csdl.sin(angle)
    c, s = c0.reshape((1, 1)), s0.reshape((1, 1))
    # strain transform laminate -> ply, engineering shear (== calc_t_eps)
    cc, ss = c * c, s * s
    t_eps = csdl.blockmat([
        [cc, ss, s * c],
        [ss, cc, -s * c],
        [-2 * s * c, 2 * s * c, cc - ss],
    ])
    eps_loc = csdl.matmat(eps_lam, csdl.transpose(t_eps))          # (nel,3): [e1,e2,g12]
    e1, e2, g12 = eps_loc[:, 0], eps_loc[:, 1], eps_loc[:, 2]

    E1, E2, v12, G12 = p["E1"], p["E2"], p["v12"], p["G12"]
    v21 = v12 * E2 / E1
    den = 1.0 - v12 * v21
    q11, q12, q22 = E1 / den, v12 * E2 / den, E2 / den
    s1 = q11 * e1 + q12 * e2
    s2 = q12 * e1 + q22 * e2
    t12 = G12 * g12

    # laminate (xz, yz) -> ply (13, 23) by the same 2-D rotation the in-plane transform
    # uses: the 1-3 plane rides with the fibre. Both components matter -- keeping tau_23
    # and dropping tau_13 under-predicts failure for out-of-plane load aligned with the
    # fibres, which is the non-conservative direction for a failure criterion.
    g13 = c0 * shear_lam[:, 0] + s0 * shear_lam[:, 1]
    g23 = -s0 * shear_lam[:, 0] + c0 * shear_lam[:, 1]

    Xt, Xc, Yt, Yc, S12 = (p[k] for k in ("Xt", "Xc", "Yt", "Yc", "S12"))
    S13, S23 = p.get("S13"), p.get("S23")
    f1, f2 = 1.0 / Xt - 1.0 / Xc, 1.0 / Yt - 1.0 / Yc
    f11, f22, f66 = 1.0 / (Xt * Xc), 1.0 / (Yt * Yc), 1.0 / S12**2
    f12 = -0.5 * np.sqrt(f11 * f22)

    idx = (f1 * s1 + f2 * s2
           + f11 * s1**2 + f22 * s2**2 + 2.0 * f12 * s1 * s2
           + f66 * t12**2)
    # guarded so a material carrying no interlaminar data degrades to the in-plane
    # criterion rather than raising -- exactly how the tau_23 term already behaved
    if S13 and p.get("G13"):
        idx = idx + (1.0 / S13**2) * (p["G13"] * g13) ** 2
    if S23 and p.get("G23"):
        idx = idx + (1.0 / S23**2) * (p["G23"] * g23) ** 2
    return idx


def tsai_wu_field(mid_strain, curvature, shear_strain, layup, faces=(-0.5, 0.5)):
    """(nel,3),(nel,3),(nel,2) laminate local-frame strains + a ``Layup`` ->
    (nel, n_plies * len(faces)) Tsai-Wu failure indices. ``faces`` are through-ply
    positions as a fraction of the ply thickness about the ply mid-plane."""
    nply = layup.num_plies
    nel = mid_strain.shape[0]
    angles = csdl.reshape(layup.angles, (nply,))
    heights = csdl.reshape(layup.heights, (nply,))
    z_bot = -0.5 * csdl.sum(heights)

    cols = []
    for k in range(nply):
        h_k = heights[k]
        z_mid = z_bot + 0.5 * h_k
        props = ply_properties(layup.materials[k])
        for frac in faces:
            z = z_mid + frac * h_k
            # eps(z) = mid - z*kappa: this codebase's shell kinematics are
            # u(xi) = u_mid - xi*(E2 x theta) (hermit/fenics/stress.py), so the ply
            # strain runs *against* the curvature. Verified to 2e-16 against
            # ShellStressRM.eps(+h/2); a '+' here would put every ply at its mirrored
            # through-thickness position.
            eps_lam = mid_strain - csdl.expand(z.reshape((1,)), (nel, 3)) * curvature
            cols.append(_tsai_wu_index(eps_lam, shear_strain, angles[k], props).reshape((nel, 1)))
        z_bot = z_bot + h_k
    return csdl.concatenate(cols, axis=1)


def aggregate_failure(field, rho=100.0):
    """Kreisselmeier-Steinhauser smooth max of the whole failure field -> a single
    scalar. Conservative: ``>= true max``, by at most ``ln(n_entries) / rho``, and
    tight when the field concentrates (as it does approaching failure). ``< 1`` is safe."""
    return csdl.maximum(field, rho=rho)
