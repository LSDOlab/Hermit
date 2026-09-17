"""Composite (LamAD CLT) material path: ``hm.laminate``."""

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit.bcs as hbc
import hermit.loads as hld
import hermit.material as hmat
import hermit.outputs as out
from hermit.domain import ShellDomain
from hermit._solve import solve
from conftest import clamped_at_x0


def _domain(plate_mesh):
    return ShellDomain(plate_mesh, element="CG2CG1")


def _state(dom, ref, material):
    return solve(dom, material, hld.pressure(dom, float(ref["pressure_z"])),
                 hbc.clamp(dom, where=clamped_at_x0))


def _iso_layup(angles_deg, total_h, E, nu):
    from hermit._laminate import Layup
    from caddee_materials import IsotropicMaterial

    mat = IsotropicMaterial(name="iso", E=E, nu=nu, G=E / (2 * (1 + nu)), density=1.0)
    angles = csdl.Variable(value=np.radians(np.asarray(angles_deg, dtype=float)), name="ply_angles")
    n = len(angles_deg)
    return Layup(mat, angles, np.full(n, total_h / n), num_plies=n), angles


def _ud_layup(angles_deg, total_h):
    """Unidirectional carbon/epoxy -- strongly orthotropic, so ply angle matters."""
    from hermit._laminate import Layup
    from caddee_materials import TransverseMaterial

    mat = TransverseMaterial(name="ud", EA=138e9, ET=10e9, vA=0.34, GA=7e9, vT=0.67, density=1.6e3)
    angles = csdl.Variable(value=np.radians(np.asarray(angles_deg, dtype=float)), name="ply_angles")
    n = len(angles_deg)
    return Layup(mat, angles, np.full(n, total_h / n), num_plies=n), angles


def test_single_iso_ply_reproduces_isotropic(plate_mesh, cantilever_ref, recorder):
    ref = cantilever_ref
    nn = int(ref["n_nodes"])
    E, nu, rd, h = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val", "h_val"))
    # nu=0 in the reference case -> keep it, LamAD handles it

    dom_iso = _domain(plate_mesh)
    iso = hmat.isotropic(dom_iso, E=E*np.ones(nn), nu=nu*np.ones(nn),
                         thickness=h*np.ones(nn), density=rd*np.ones(nn),
                         constitutive_space=("Lagrange", 1))
    s_iso = _state(dom_iso, ref, iso)

    layup, _ = _iso_layup([0.0], h, E, nu)
    dom_lam = _domain(plate_mesh)
    lam = hmat.laminate(dom_lam, layup=layup, density=rd*np.ones(nn),
                        constitutive_space=("Lagrange", 1))
    s_lam = _state(dom_lam, ref, lam)

    ci = float(np.ravel(out.compliance(s_iso).value)[0])
    cl = float(np.ravel(out.compliance(s_lam).value)[0])
    print(f"\ncompliance  iso={ci:.10e}  laminate[0]={cl:.10e}  rel={abs(ci-cl)/ci:.2e}")
    assert cl == pytest.approx(ci, rel=1e-9)
    assert float(np.ravel(out.mass(s_lam).value)[0]) == pytest.approx(
        float(np.ravel(out.mass(s_iso).value)[0]), rel=1e-12)


def test_fiber_aligned_is_stiffer_than_transverse(plate_mesh, cantilever_ref, recorder):
    """UD plies aligned with the beam axis (x) resist tip bending far better than 90deg."""
    ref = cantilever_ref
    nn = int(ref["n_nodes"])
    h = float(ref["h_val"])

    def compliance(angles):
        layup, _ = _ud_layup(angles, h)
        dom = _domain(plate_mesh)
        m = hmat.laminate(dom, layup=layup, density=1.6e3 * np.ones(nn),
                          constitutive_space=("Lagrange", 1))
        return float(np.ravel(out.compliance(_state(dom, ref, m)).value)[0])

    c_0 = compliance([0.0, 0.0, 0.0])
    c_90 = compliance([90.0, 90.0, 90.0])
    print(f"\n[0/0/0] compliance={c_0:.4e}   [90/90/90]={c_90:.4e}   ratio={c_90/c_0:.2f}")
    assert c_0 < c_90
    assert c_90 / c_0 > 3.0  # EA/ET ~ 14, so bending compliance ratio is large


def test_compliance_derivative_wrt_ply_angles(plate_mesh, cantilever_ref):
    ref = cantilever_ref
    nn = int(ref["n_nodes"])
    h = float(ref["h_val"])
    ang0 = [10.0, 80.0, 35.0]

    def run(delta, want_grad):
        rec = csdl.Recorder(inline=True); rec.start()
        layup, angles = _ud_layup(ang0, h)
        if delta is not None:
            angles.value[:] = angles.value + delta
        dom = _domain(plate_mesh)
        m = hmat.laminate(dom, layup=layup, density=1.6e3 * np.ones(nn),
                          constitutive_space=("Lagrange", 1))
        o = out.compliance(_state(dom, ref, m))
        val = float(np.ravel(o.value)[0])
        g = None
        if want_grad:
            g = np.asarray(csdl.experimental.PySimulator(rec).compute_totals([o], [angles])[o, angles]).ravel()
        rec.stop()
        return val, g

    _, g = run(None, True)
    assert np.all(np.isfinite(g)) and np.linalg.norm(g) > 0
    # directional central difference
    V = np.array([1.0, -0.5, 0.7]); V /= np.linalg.norm(V)
    step = 1e-4
    vp, _ = run(step * V, False)
    vm, _ = run(-step * V, False)
    dd_fd = (vp - vm) / (2 * step)
    dd_an = float(g @ V)
    rel = abs(dd_an - dd_fd) / abs(dd_fd)
    print(f"\nd(compliance)/d(angles).V : analytic={dd_an:.6e}  fd={dd_fd:.6e}  rel={rel:.2e}")
    assert rel < 1e-4


def test_laminate_transverse_shear_matches_the_element_frame_ordering(plate_mesh):
    """For a 0-deg UD ply the fibre runs along the element e0, so the shear stiffness in
    the e0 (1-3) plane is G13 = GA and the one in the e1 (2-3) plane is G23 = ET/(2(1+vT)).
    ``As`` is contracted with ``gamma = [gamma_e0z, gamma_e1z]``, so slot 0 must be G13."""
    rec = csdl.Recorder(inline=True); rec.start()
    h, kappa = 0.02, 0.833
    layup, _ = _ud_layup([0.0], h)
    dom = _domain(plate_mesh)
    mat = hmat.laminate(dom, layup=layup, density=1.6e3, constitutive_space=("DG", 0))
    As = np.asarray(mat.As.coeffs.value).reshape(-1, 2, 2)[0]
    rec.stop()

    EA, ET, vA, GA, vT = 138e9, 10e9, 0.34, 7e9, 0.67
    G13, G23 = GA, ET / (2.0 * (1.0 + vT))
    print(f"\nAs = {As.tolist()}   expected diag({kappa*G13*h:.4e}, {kappa*G23*h:.4e})")
    assert As[0, 0] == pytest.approx(kappa * G13 * h, rel=1e-9)   # e0 (1-3) shear
    assert As[1, 1] == pytest.approx(kappa * G23 * h, rel=1e-9)   # e1 (2-3) shear
