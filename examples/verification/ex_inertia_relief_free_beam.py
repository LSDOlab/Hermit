"""Free-free beam balanced by inertia relief

The 10 by 2 by 0.2 plate has no supports. Each end carries a downward line load
totalling ``P / 2``; ``hm.inertia_relief`` balances them with the uniform upward
inertial load ``q = P / L``, and ``hm.gauge`` on all six dofs at the centre removes
the rigid-body modes without reacting anything. By symmetry each half is a
cantilever of length ``a = L / 2`` from the centre, loaded by ``-P / 2`` at its tip
and ``q`` along it, so the tip deflects relative to the centre by
``-5 P a**3 / (48 E I)`` (Euler--Bernoulli) plus ``-P a / (4 k G A)`` (Timoshenko
shear). The quantity is gauge-independent: it is a difference of two displacements.
"""

import pathlib
import sys

import csdl_alpha as csdl
import numpy as np

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import rect_plate  # noqa: E402
from _harness import Case, main, node_nearest  # noqa: E402

L, W, H, E, NU, RHO, P = 10.0, 2.0, 0.2, 4.32e8, 0.0, 1.0, 4.0
I = W * H**3 / 12.0
A = W * H
K = 5.0 / 6.0
G = E / (2.0 * (1.0 + NU))
HALF = L / 2.0
EULER_BERNOULLI = 5.0 * P * HALF**3 / (48.0 * E * I)
REFERENCE = EULER_BERNOULLI + P * HALF / (4.0 * K * G * A)


def solve_at(n):
    """Tip-minus-centre deflection on an ``n x n/2`` quadrilateral mesh."""
    mesh = rect_plate(L, W, nx=n, ny=n // 2, cell="quad")
    rec = csdl.Recorder(inline=True)
    rec.start()
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=H, density=RHO)
    end_load = [0.0, 0.0, -P / (2.0 * W)]
    loads = (hm.edge_traction(domain, end_load, where=hm.near("x", 0.0))
             + hm.edge_traction(domain, end_load, where=hm.near("x", L)))
    relief, _ = hm.inertia_relief(domain, material, loads)
    centre = [L / 2.0, W / 2.0, 0.0]
    state = hm.solve(domain, material, loads + relief,
                     hm.gauge(domain, at=centre, dofs=("ux", "uy", "uz", "rx", "ry", "rz")))
    u = hm.nodal_displacement(state).value.reshape(-1, 3)
    rec.stop()
    tip, d_tip = node_nearest(domain, [L, W / 2.0, 0.0])
    mid, d_mid = node_nearest(domain, centre)
    assert d_tip < 1e-12 and d_mid < 1e-12
    return abs(float(u[tip, 2] - u[mid, 2]))


CASE = Case(
    name="Free-free beam: inertia relief",
    quantity="tip deflection relative to the centre",
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
