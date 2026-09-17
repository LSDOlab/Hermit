"""Simply supported square plate under a sinusoidal pressure.

A square, isotropic plate of side ``a`` and thickness ``h = a / 100`` carries
``q(x, y) = q0 sin(pi x/a) sin(pi y/a)``.  It is deliberately thin, so
Kirchhoff--Love plate theory, rather than transverse-shear deformation, is the
reference.  Its Navier solution is one term exactly:
``w_max = q0 / (D pi^4 (1/a^2 + 1/b^2)^2)``, where
``D = E h^3 / (12 (1-nu^2))``.  This example evaluates that expression below;
it is the sharpest plate gate in this suite because no truncated series is involved.

The support is the *soft* simply supported shell idealisation: ``uz = 0`` on all
four edges and every rotation is free, so the natural bending-moment condition is
retained.  In-plane displacement is also free except for three point gauges that
remove only the planar rigid-body modes (``ux, uy`` at one corner and ``uy`` at a
second).  A hard support that pins in-plane edge motion is a different membrane
problem and can change the answer.
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
REFERENCE = Q0 / (D * np.pi**4 * (1.0 / A**2 + 1.0 / B**2)**2)


def _soft_simple_support(domain):
    """Soft support plus the minimum in-plane rigid-body gauges."""
    edge = lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], A)
                      | np.isclose(x[1], 0.0) | np.isclose(x[1], B))
    return (hm.pin(domain, where=edge, dofs=("uz",))
            + hm.gauge(domain, at=[0.0, 0.0, 0.0], dofs=("ux", "uy"))
            + hm.gauge(domain, at=[A, 0.0, 0.0], dofs=("uy",)))


def solve_at(n):
    """Centre deflection on an ``n x n`` quad mesh."""
    mesh = rect_plate(A, B, nx=n, ny=n, cell="quad")
    rec = csdl.Recorder(inline=True)
    rec.start()
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=H, density=1.0)
    traction = hm.from_function(
        domain, ("Lagrange", 2, (3,)),
        lambda x: np.column_stack((np.zeros(len(x)), np.zeros(len(x)),
                                   Q0 * np.sin(np.pi * x[:, 0] / A)
                                   * np.sin(np.pi * x[:, 1] / B))),
    )
    state = hm.solve(domain, material, hm.traction(domain, traction),
                     _soft_simple_support(domain))
    u = hm.nodal_displacement(state).value.reshape(-1, 3)
    rec.stop()
    k, dist = node_nearest(domain, [A / 2, B / 2, 0.0])
    if dist > 1e-9:
        print(f"    (note: centre sample node is {dist:.2e} away at n={n})")
    return abs(u[k, 2])


CASE = Case(
    name="Simply supported sinusoidally loaded square plate",
    quantity="centre transverse deflection",
    reference=REFERENCE,
    tolerance=0.02,
    citation="Computed Kirchhoff--Love single Navier mode",
    levels=(8, 12, 16, 24),
    quick_level=24,
    solve=solve_at,
    notes="a/h=100; soft simple support (uz only) with in-plane rigid-body gauges",
)

if __name__ == "__main__":
    main(CASE)
