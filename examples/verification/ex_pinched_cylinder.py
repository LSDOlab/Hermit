"""Pinched cylinder with end diaphragms

The hardest of the standard linear shell benchmarks. A cylinder of radius 300, length
600 and thickness 3 sits on rigid end diaphragms and is pinched by two opposed unit
point loads at midspan. The reference radial deflection under a load is
**1.8248e-5** (the value is usually quoted as ``1.8248e-5`` for ``P = 1``, ``E = 3e6``,
``nu = 0.3``).

The response is almost inextensional bending with a very localised load region, so
convergence is slow and this case is the standard discriminator for shear and
membrane locking. Only one octant is modelled, using three symmetry planes; the
applied load is therefore ``P/4``.

    conda activate hermit
    python examples/verification/ex_pinched_cylinder.py
    python examples/verification/ex_pinched_cylinder.py --quick
"""

import pathlib
import sys

import numpy as np
import csdl_alpha as csdl

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import cylinder_sector                  # noqa: E402
from _harness import Case, main, node_nearest          # noqa: E402

RADIUS, LENGTH, THICKNESS = 300.0, 600.0, 3.0
E, NU, LOAD = 3.0e6, 0.3, 1.0
REFERENCE = 1.8248e-5


def solve_at(n):
    """Radial deflection under the load on an ``n x n`` octant mesh."""
    # Octant: x in [0, L/2] (symmetry at midspan), theta in [0, 90] degrees.
    mesh = cylinder_sector(RADIUS, LENGTH / 2, nx=n, nt=n,
                           theta0=0.0, theta1=90.0, cell="triangle")

    rec = csdl.Recorder(inline=True)
    rec.start()

    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=THICKNESS, density=1.0)

    bcs = (
        # midspan symmetry (the cut normal to the axis)
        hm.symmetry(domain, where=lambda x: np.isclose(x[0], LENGTH / 2), normal=[1, 0, 0])
        # the two circumferential symmetry cuts, at theta = 0 and theta = 90 degrees
        + hm.symmetry(domain, where=lambda x: np.isclose(x[1], 0.0), normal=[0, 1, 0])
        + hm.symmetry(domain, where=lambda x: np.isclose(x[2], 0.0), normal=[0, 0, 1])
        # rigid diaphragm at the far end
        + hm.pin(domain, where=lambda x: np.isclose(x[0], 0.0), dofs=("uy", "uz"))
    )
    load = hm.point_load(domain, at=[LENGTH / 2, 0.0, RADIUS],
                         force=[0.0, 0.0, -LOAD / 4])
    state = hm.solve(domain, material, load, bcs)
    u = hm.nodal_displacement(state).value.reshape(-1, 3)

    rec.stop()

    k, dist = node_nearest(domain, [LENGTH / 2, 0.0, RADIUS])
    if dist > 1e-9:
        print(f"    (note: sample node is {dist:.2e} from the nominal point at n={n})")
    return abs(u[k, 2])


CASE = Case(
    name="Pinched cylinder with end diaphragms",
    quantity="radial deflection under the point load",
    reference=REFERENCE,
    tolerance=0.02,
    citation="MacNeal & Harder, 'A proposed standard set of problems to test finite "
             "element accuracy', Finite Elements in Analysis and Design 1(1), 1985",
    levels=(8, 16, 24, 32, 48),
    quick_level=48,
    solve=solve_at,
    notes="inextensional bending; converges from below and slowly -- a coarse mesh "
          "reading well under the reference is expected, not a defect",
)

if __name__ == "__main__":
    main(CASE)
