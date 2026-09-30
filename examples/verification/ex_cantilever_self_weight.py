"""Cantilever plate under its own weight

The 10 by 2 by 0.2 plate is clamped at ``x = 0`` and loaded only by
``hm.inertial_load`` at ``g = 9.81`` downward, with density ``rho = 1``.  For the
beam strip the line load is ``q = rho g H W``.  The computed reference is the
Euler--Bernoulli tip value ``q L**4 / (8 E I)`` plus the Timoshenko shear term
``q L**2 / (2 k G A)`` -- the same closed form as ``ex_cantilever_uniform_pressure``,
reached through the structure's mass instead of an applied pressure.
"""

import pathlib
import sys

import csdl_alpha as csdl
import numpy as np

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import rect_plate  # noqa: E402
from _harness import Case, main, node_nearest  # noqa: E402

L, W, H, E, NU, RHO, GRAVITY = 10.0, 2.0, 0.2, 4.32e8, 0.0, 1.0, 9.81
I = W * H**3 / 12.0
A = W * H
K = 5.0 / 6.0
G = E / (2.0 * (1.0 + NU))
Q = RHO * GRAVITY * H * W
EULER_BERNOULLI = Q * L**4 / (8.0 * E * I)
REFERENCE = EULER_BERNOULLI + Q * L**2 / (2.0 * K * G * A)


def solve_at(n):
    """Tip displacement on an ``n x n/2`` quadrilateral mesh."""
    mesh = rect_plate(L, W, nx=n, ny=n // 2, cell="quad")
    rec = csdl.Recorder(inline=True)
    rec.start()
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=H, density=RHO)
    weight = hm.inertial_load(domain, material, acceleration=[0.0, 0.0, -GRAVITY])
    state = hm.solve(domain, material, weight,
                     hm.clamp(domain, where=lambda x: np.isclose(x[0], 0.0)))
    u = hm.nodal_displacement(state).value.reshape(-1, 3)
    rec.stop()
    tip, distance = node_nearest(domain, [L, W / 2.0, 0.0])
    assert distance < 1e-12
    return abs(float(u[tip, 2]))


CASE = Case(
    name="Cantilever plate: self weight",
    quantity="vertical displacement at the tip-edge midpoint",
    reference=REFERENCE,
    tolerance=0.02,
    citation="Computed Timoshenko beam formula in this file",
    levels=(8, 12, 16),
    quick_level=12,
    solve=solve_at,
    notes=f"Euler-Bernoulli={EULER_BERNOULLI:.8e}; Timoshenko={REFERENCE:.8e}",
)

if __name__ == "__main__":
    main(CASE)
