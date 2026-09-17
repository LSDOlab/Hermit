"""Navier bending solution for a symmetric cross-ply laminate

A simply supported, symmetric ``[0/90]s`` plate is loaded by one sinusoidal pressure
term.  For this specially orthotropic stack ``B = 0`` and ``D16 = D26 = 0``; the
reference deflection is the one-term Navier solution computed below directly from
the laminate's independently assembled bending matrix.  The check also prints and
asserts the negligible coupling terms, so the analytic assumption is explicit.
"""

import pathlib
import sys

import numpy as np
import csdl_alpha as csdl
from caddee_materials import TransverseMaterial

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import rect_plate  # noqa: E402
from _harness import Case, main, node_nearest  # noqa: E402

A, B, H, Q0 = 1.0, 0.8, 0.02, 1.0e4
E1, E2, NU12, G12 = 138e9, 10e9, 0.31, 7e9
ANGLES = np.radians([0.0, 90.0, 90.0, 0.0])


def qbar(theta):
    nu21 = NU12 * E2 / E1; den = 1.0 - NU12 * nu21
    q = np.array([[E1 / den, NU12 * E2 / den, 0.0], [NU12 * E2 / den, E2 / den, 0.0], [0.0, 0.0, G12]])
    c, s = np.cos(theta), np.sin(theta)
    te = np.array([[c*c, s*s, s*c], [s*s, c*c, -s*c], [-2*s*c, 2*s*c, c*c-s*s]])
    ts = np.array([[c*c, s*s, -2*s*c], [s*s, c*c, 2*s*c], [s*c, -s*c, c*c-s*s]])
    return ts @ q @ te


def reference_d():
    z = np.linspace(-H / 2, H / 2, len(ANGLES) + 1)
    return sum(qbar(t) * (z1**3 - z0**3) / 3 for t, z0, z1 in zip(ANGLES, z[:-1], z[1:]))


DREF = reference_d()
REFERENCE = Q0 / (np.pi**4 * (DREF[0, 0] / A**4 + 2 * (DREF[0, 1] + 2 * DREF[2, 2]) / (A**2 * B**2) + DREF[1, 1] / B**4))


def solve_at(n):
    """Midpoint transverse displacement on an ``n`` by ``n`` simply-supported mesh."""
    rec = csdl.Recorder(inline=True); rec.start()
    domain = hm.ShellDomain(rect_plate(A, B, nx=n, ny=n), element="CG2CG1")
    ply = TransverseMaterial(name="ud", EA=E1, ET=E2, vA=NU12, vT=0.4, GA=G12, density=1600.)
    layup = hm.Layup(ply, ANGLES, np.full(4, H / 4), num_plies=4)
    mat = hm.laminate(domain, layup=layup, density=1600.)
    # w=0 on all four edges is the essential simply-supported condition; a minimal
    # in-plane pin removes the otherwise free rigid translation without clamping rotation.
    edge_w = hm.pin(domain, where=lambda x: np.isclose(x[0], 0.) | np.isclose(x[0], A) |
                                        np.isclose(x[1], 0.) | np.isclose(x[1], B), dofs=("uz",))
    anchor = hm.pin(domain, where=lambda x: np.isclose(x[0], 0.) & np.isclose(x[1], 0.), dofs=("ux", "uy"))
    sinusoid = hm.from_function(
        domain, ("Lagrange", 1),
        lambda x: Q0 * np.sin(np.pi * x[:, 0] / A) * np.sin(np.pi * x[:, 1] / B),
    )
    state = hm.solve(domain, mat, hm.pressure(domain, sinusoid), edge_w + anchor)
    k, _ = node_nearest(domain, [A / 2, B / 2, 0.])
    w = abs(float(hm.nodal_displacement(state).value.reshape(-1, 3)[k, 2]))
    rec.stop()
    return w


assert abs(DREF[0, 2]) <= 1e-12 * abs(DREF[0, 0]) and abs(DREF[1, 2]) <= 1e-12 * abs(DREF[1, 1])
CASE = Case(
    name="Symmetric cross-ply sinusoidal bending", quantity="midpoint transverse deflection",
    reference=REFERENCE, tolerance=0.06,
    citation="one-term Navier solution evaluated from the independently assembled D matrix",
    levels=(8, 12, 16), quick_level=8, solve=solve_at,
    notes=f"D16/D11={DREF[0,2]/DREF[0,0]:.2e}, D26/D22={DREF[1,2]/DREF[1,1]:.2e}",
)

if __name__ == "__main__":
    main(CASE)
