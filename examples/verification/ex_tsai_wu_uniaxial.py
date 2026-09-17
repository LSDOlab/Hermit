"""Tsai-Wu transverse-shear failure in a uniaxially loaded laminate

A short, thick one-ply cantilever receives a statically equivalent, uniformly
distributed tip force.  Away from the root this produces the nearly uniform ``xz``
shear state used here to exercise the interlaminar ``tau13`` contribution.  The
script independently recovers the ply stresses from the same DG0 strains and applies
Hermit's documented Tsai--Wu coefficients, then compares every failure-field value
and its KS aggregate.

``hm.stress_field`` intentionally rejects laminates (it is an isotropic von-Mises
output), so laminate ply stresses are recovered below from ``hm.strain_fields`` --
the exact strain data path used by ``hm.failure_field``.  Omitting ``tau13`` from the
hand formula changes this case materially, making it a regression guard for the
interlaminar-shear fix.
"""

import pathlib
import sys

import numpy as np
import csdl_alpha as csdl
from caddee_materials import TransverseMaterial

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import rect_plate  # noqa: E402
from _harness import Case, main  # noqa: E402

L, W, H, P = 0.30, 0.10, 0.040, 4.0e4
E1, E2, NU12, G12 = 45e9, 12e9, 0.28, 4.5e9
XT, XC, YT, YC, S12, S23 = 2e9, 2e9, 2e9, 2e9, 8e6, 2e9


def hand_tsai_wu(mid, curvature, shear):
    """Tsai--Wu at both faces from the documented one-ply, zero-angle stresses."""
    nu21 = NU12 * E2 / E1
    den = 1.0 - NU12 * nu21
    q11, q12, q22 = E1 / den, NU12 * E2 / den, E2 / den
    f1, f2 = 1 / XT - 1 / XC, 1 / YT - 1 / YC
    f11, f22, f66 = 1 / (XT * XC), 1 / (YT * YC), 1 / S12**2
    f12 = -0.5 * np.sqrt(f11 * f22)
    cols, tau13_terms = [], []
    for z in (-H / 2, H / 2):
        eps = mid - z * curvature  # Hermit's convention is eps(z)=mid-z*kappa.
        s1, s2 = q11 * eps[:, 0] + q12 * eps[:, 1], q12 * eps[:, 0] + q22 * eps[:, 1]
        t12, t13, t23 = G12 * eps[:, 2], G12 * shear[:, 0], E2 / (2 * (1 + 0.4)) * shear[:, 1]
        tau13 = (t13 / S12)**2  # TransverseMaterial stores S13 == S12.
        index = f1*s1 + f2*s2 + f11*s1**2 + f22*s2**2 + 2*f12*s1*s2 + f66*t12**2 + tau13 + (t23 / S23)**2
        cols.append(index); tau13_terms.append(tau13)
    return np.column_stack(cols), np.column_stack(tau13_terms)


def solve_at(n):
    """Maximum field/KS discrepancy relative to the hand Tsai--Wu calculation."""
    rec = csdl.Recorder(inline=True); rec.start()
    domain = hm.ShellDomain(rect_plate(L, W, nx=n, ny=max(2, n // 3)), element="CG2CG1")
    ply = TransverseMaterial(name="ud", EA=E1, ET=E2, vA=NU12, vT=0.4, GA=G12, density=1600.)
    ply.set_strength(F1t=XT, F1c=XC, F2t=YT, F2c=YC, F12=S12, F23=S23)
    layup = hm.Layup(ply, np.array([0.0]), np.array([H]), num_plies=1)
    material = hm.laminate(domain, layup=layup, density=1600.)
    bcs = hm.clamp(domain, where=lambda x: np.isclose(x[0], 0.))
    # Equal nodal shares are statically equivalent to a uniform tip traction; the
    # central cells then have the intended almost uniform transverse shear.
    edge = np.flatnonzero(np.isclose(domain.node_coords[:, 0], L))
    loads = None
    for k in edge:
        load = hm.point_load(domain, at=domain.node_coords[k], force=[0., 0., P / len(edge)])
        loads = load if loads is None else loads + load
    state = hm.solve(domain, material, loads, bcs)
    mid, curvature, shear = hm.strain_fields(state, space=("DG", 0), method="average")
    hand, tau13 = hand_tsai_wu(mid.values, curvature.values, shear.values)
    field = hm.failure_field(state).values
    fi = float(hm.failure_index(state, rho=100).value[0])
    peak = hand.max()
    hand_ks = peak + np.log(np.exp(100 * (hand - peak)).sum()) / 100
    tau_ratio = tau13.max() / max(np.abs(hand).max(), 1e-30)
    print(f"    max hand field={peak:.6e}, tau13 share={tau_ratio:.3f}, KS={hand_ks:.6e}")
    if tau_ratio < 0.05:
        raise RuntimeError("tau13 is too small to guard the interlaminar-shear term")
    err_field = np.max(np.abs(field - hand)) / max(1.0, np.abs(hand).max())
    err_ks = abs(fi - hand_ks) / max(1.0, abs(hand_ks))
    rec.stop()
    return max(err_field, err_ks)


CASE = Case(
    name="Tsai-Wu uniaxial transverse-shear guard",
    quantity="maximum hand-recovered Tsai-Wu field or KS discrepancy",
    reference=0.0, tolerance=1e-10,
    citation="Tsai--Wu coefficients and KS reduction independently evaluated from hermit/failure.py",
    levels=(6, 10), quick_level=6, solve=solve_at, monotone=False,
    notes="the hand criterion includes the TransverseMaterial S13=S12 tau13 term",
)

if __name__ == "__main__":
    main(CASE)
