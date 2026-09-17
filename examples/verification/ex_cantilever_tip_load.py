"""Cantilever plate under a tip force

A flat, 10 by 2 by 0.2 cantilever plate is clamped at ``x = 0`` and carries a
total unit force in global ``-z`` at ``x = 10``.  The force is split equally
between the tip-edge nodes, which is statically equivalent to the requested
resultant (but is not the consistent quadratic edge-load interpolation).

The Euler--Bernoulli value is ``5.78703704e-4``.  The reference is the
Timoshenko value ``5.78842593e-4``, computed below from
``P L**3 / (3 E I) + P L / (k G A)``, where ``I = W h**3 / 12``,
``A = W h``, ``k = 5/6``, and ``G = E / (2 (1 + nu))``.  The corresponding
Euler--Bernoulli expression omits the second, shear-deflection term.  This
Reissner--Mindlin shell has transverse shear, so Timoshenko is the reference;
the Euler--Bernoulli value is reported to show that the distinction is small
for this slender plate.
"""

import pathlib
import sys

import csdl_alpha as csdl
import numpy as np

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import rect_plate  # noqa: E402
from _harness import Case, main, node_nearest  # noqa: E402

L, W, H, E, NU, P = 10.0, 2.0, 0.2, 4.32e8, 0.0, 1.0
I = W * H**3 / 12.0
A = W * H
K = 5.0 / 6.0
G = E / (2.0 * (1.0 + NU))
EULER_BERNOULLI = P * L**3 / (3.0 * E * I)
REFERENCE = EULER_BERNOULLI + P * L / (K * G * A)


def solve_at(n):
    """Tip displacement on an ``n x n/2`` quadrilateral mesh."""
    mesh = rect_plate(L, W, nx=n, ny=n // 2, cell="quad")
    rec = csdl.Recorder(inline=True)
    rec.start()
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=H, density=1.0)
    xyz = np.asarray(domain.node_coords)
    edge = np.flatnonzero(np.isclose(xyz[:, 0], L))
    load = None
    for k in edge:
        term = hm.point_load(domain, at=xyz[k], force=[0.0, 0.0, -P / len(edge)])
        load = term if load is None else load + term
    state = hm.solve(domain, material, load,
                     hm.clamp(domain, where=lambda x: np.isclose(x[0], 0.0)))
    u = hm.nodal_displacement(state).value.reshape(-1, 3)
    rec.stop()
    tip, distance = node_nearest(domain, [L, W / 2.0, 0.0])
    assert distance < 1e-12
    return abs(float(u[tip, 2]))


CASE = Case(
    name="Cantilever plate: total tip force",
    quantity="vertical displacement at the tip-edge midpoint",
    reference=REFERENCE,
    tolerance=0.02,
    citation="Computed Timoshenko beam formula in this file",
    levels=(8, 12, 16),
    quick_level=8,
    solve=solve_at,
    notes=f"Euler-Bernoulli={EULER_BERNOULLI:.8e}; Timoshenko={REFERENCE:.8e}",
)

if __name__ == "__main__":
    main(CASE)
