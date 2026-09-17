"""Simply supported square plate under uniform pressure.

The square plate has ``h = a / 100``, making it thin enough that the Kirchhoff--Love
Navier solution is the appropriate reference.  Its centre deflection is computed
below from the odd-``m``, odd-``n`` double series, rather than importing the rounded
``0.00406 q a^4 / D`` textbook coefficient.  The two retained truncations are also
checked and reported, demonstrating convergence of the computed reference.

This uses the soft simple-support shell idealisation: all edge ``uz`` dofs are held,
rotations remain free for the natural zero-moment condition, and only three point
in-plane gauges remove rigid-body motion.  Pinning in-plane edge motion would make a
hard support and introduce a different membrane constraint.
"""

import pathlib
import sys

import csdl_alpha as csdl
import numpy as np

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import rect_plate                    # noqa: E402
from _harness import Case, main, node_nearest        # noqa: E402

A = B = 1.0
E, NU, H, Q0 = 1.0e7, 0.3, 0.01, 1.0e3
D = E * H**3 / (12.0 * (1.0 - NU**2))


def _navier_reference(n_odd):
    """Kirchhoff centre deflection using ``n_odd`` odd indices in each direction.

    The mode shapes must be **evaluated at the centre**, and that factor
    ``sin(m pi/2) sin(n pi/2)`` alternates in sign for odd ``m``, ``n`` -- it is not
    identically one. Summing the magnitudes instead inflates the series by 5.687 %
    (0.0042934 vs the correct 0.0040624 q a^4 / D, against Timoshenko's tabulated
    0.00406), which is enough to make a perfectly good solution look like a 5 %
    convergence failure that never closes under refinement.
    """
    m = np.arange(1, 2 * n_odd, 2, dtype=float)
    mm, nn = np.meshgrid(m, m, indexing="ij")
    centre = np.sin(mm * np.pi / 2.0) * np.sin(nn * np.pi / 2.0)
    series = np.sum(centre / (mm * nn * (mm**2 / A**2 + nn**2 / B**2)**2))
    return 16.0 * Q0 * series / (np.pi**6 * D)


REFERENCE = _navier_reference(801)
_REFERENCE_401 = _navier_reference(401)


def _soft_simple_support(domain):
    edge = lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], A)
                      | np.isclose(x[1], 0.0) | np.isclose(x[1], B))
    return (hm.pin(domain, where=edge, dofs=("uz",), method="strong")
            + hm.gauge(domain, at=[0.0, 0.0, 0.0], dofs=("ux", "uy"))
            + hm.gauge(domain, at=[A, 0.0, 0.0], dofs=("uy",)))


def solve_at(n):
    """Centre deflection on an ``n x n`` quad mesh."""
    mesh = rect_plate(A, B, nx=n, ny=n, cell="quad")
    rec = csdl.Recorder(inline=True)
    rec.start()
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=H, density=1.0)
    state = hm.solve(domain, material, hm.traction(domain, [0.0, 0.0, Q0]),
                     _soft_simple_support(domain))
    u = hm.nodal_displacement(state).value.reshape(-1, 3)
    rec.stop()
    k, dist = node_nearest(domain, [A / 2, B / 2, 0.0])
    if dist > 1e-9:
        print(f"    (note: centre sample node is {dist:.2e} away at n={n})")
    return abs(u[k, 2])


CASE = Case(
    name="Simply supported uniformly loaded square plate",
    quantity="centre transverse deflection",
    reference=REFERENCE,
    tolerance=0.02,
    citation="Computed Kirchhoff--Love odd Navier double series (801 terms/direction)",
    levels=(8, 12, 16, 24, 48),
    # No swept level meets this fixed 2% gate: keep the most diagnostic, affordable
    # level in CI rather than disguising the convergence miss by changing the target.
    quick_level=48,
    solve=solve_at,
    notes="a/h=100; soft simple support; 401-to-801 series change is "
          f"{abs(REFERENCE - _REFERENCE_401) / REFERENCE:.2e}; current FE sweep "
          "remains about 5% low",
)

if __name__ == "__main__":
    print("  Navier-series convergence: "
          f"401 odd terms { _REFERENCE_401:.10e}, 801 odd terms {REFERENCE:.10e}, "
          f"relative change {abs(REFERENCE - _REFERENCE_401) / REFERENCE:.2e}")
    main(CASE)
