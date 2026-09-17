"""Clamped square plate under uniform pressure.

The plate is thin (``h = a / 100``), so Kirchhoff--Love plate theory is the relevant
comparison.  Unlike a simply supported rectangle, the fully clamped plate has no
short closed form.  The reference here is therefore deliberately a *quoted* value,
``w_max = 0.00126 q a^4 / D``, not a computed result: Timoshenko and
Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed. (1959), Table 35 (p. 202),
the clamped rectangular-plate table.  The tolerance reflects its three-significant-
figure coefficient.
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
REFERENCE = 0.00126 * Q0 * A**4 / D


def solve_at(n):
    """Centre deflection on an ``n x n`` quad mesh."""
    mesh = rect_plate(A, B, nx=n, ny=n, cell="quad")
    rec = csdl.Recorder(inline=True)
    rec.start()
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=H, density=1.0)
    edge = lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], A)
                      | np.isclose(x[1], 0.0) | np.isclose(x[1], B))
    state = hm.solve(domain, material, hm.traction(domain, [0.0, 0.0, Q0]),
                     hm.clamp(domain, where=edge))
    u = hm.nodal_displacement(state).value.reshape(-1, 3)
    rec.stop()
    k, dist = node_nearest(domain, [A / 2, B / 2, 0.0])
    if dist > 1e-9:
        print(f"    (note: centre sample node is {dist:.2e} away at n={n})")
    return abs(u[k, 2])


CASE = Case(
    name="Clamped uniformly loaded square plate",
    quantity="centre transverse deflection",
    reference=REFERENCE,
    tolerance=0.01,
    citation="Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells, 2nd ed. "
             "(1959), Table 35, p. 202: quoted 0.00126",
    levels=(8, 12, 16, 24),
    quick_level=24,
    solve=solve_at,
    notes="a/h=100; reference coefficient is quoted to three significant figures",
)

if __name__ == "__main__":
    main(CASE)
