"""Clamped circular plate under uniform pressure.

An isotropic disk of radius ``a`` and thin thickness ``h = a / 100`` is clamped on
its rim and loaded normally by uniform pressure.  It is thin enough that Kirchhoff--
Love theory is the correct reference, whose exact centre deflection is
``w_max = q a^4 / (64 D)``; the expression is evaluated below.  The mesh comes from
the ``disk()`` polar builder.  Its outer boundary is an inscribed polygon and hence
slightly under-represents the true circle, so the circumferential resolution is held
at eight times the radial level: the chordal geometric error is then much smaller
than the finite-element discretisation error.
"""

import pathlib
import sys

import csdl_alpha as csdl
import numpy as np

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import disk                          # noqa: E402
from _harness import Case, main, node_nearest        # noqa: E402

A = 1.0
E, NU, H, Q0 = 1.0e7, 0.3, 0.01, 1.0e3
D = E * H**3 / (12.0 * (1.0 - NU**2))
REFERENCE = Q0 * A**4 / (64.0 * D)


def solve_at(n):
    """Centre deflection on a polar mesh with ``n`` radial and ``8n`` angular sectors."""
    mesh = disk(A, nr=n, nt=8 * n)
    rec = csdl.Recorder(inline=True)
    rec.start()
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=H, density=1.0)
    rim = lambda x: np.isclose(x[0]**2 + x[1]**2, A**2, rtol=0.0, atol=1e-10)
    state = hm.solve(domain, material, hm.traction(domain, [0.0, 0.0, Q0]),
                     hm.clamp(domain, where=rim))
    u = hm.nodal_displacement(state).value.reshape(-1, 3)
    rec.stop()
    k, dist = node_nearest(domain, [0.0, 0.0, 0.0])
    if dist > 1e-12:
        print(f"    (note: centre sample node is {dist:.2e} away at n={n})")
    return abs(u[k, 2])


CASE = Case(
    name="Clamped circular plate under uniform pressure",
    quantity="centre transverse deflection",
    reference=REFERENCE,
    tolerance=0.02,
    citation="Computed Kirchhoff--Love circular-plate solution, w=q a^4/(64 D)",
    levels=(6, 10, 16, 24),
    quick_level=24,
    solve=solve_at,
    notes="a/h=100; disk rim has 8n inscribed-polygon sectors",
)

if __name__ == "__main__":
    main(CASE)
