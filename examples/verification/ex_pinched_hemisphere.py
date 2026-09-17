"""Pinched hemisphere with an 18-degree hole

Pinched hemispherical shell with an 18 degree polar hole. R=10, thickness=0.04,
E=6.825e7, nu=0.3. Loaded by alternating inward/outward radial point forces P=2.0 at
90 degree intervals around the free equator. The equator and the hole edge are both FREE.
The reference radial displacement under the load is **0.0924**.

This case is nearly inextensional and converges slowly.  The model is the octant
between two symmetry planes.  Each of its two equatorial load points lies on a
symmetry plane, so it owns one half of each physical P=2 load: the forces applied to
the octant are therefore P/2=1.  A refinement through n=96 plateaus near 0.0935;
using P/2=0.5 instead converges near half the published value.

    conda activate hermit
    python examples/verification/ex_pinched_hemisphere.py
    python examples/verification/ex_pinched_hemisphere.py --quick
"""

import pathlib
import sys

import numpy as np
import csdl_alpha as csdl

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import sphere_octant                      # noqa: E402
from _harness import Case, main, node_nearest            # noqa: E402

RADIUS, THICKNESS, HOLE_ANGLE = 10.0, 0.04, 18.0
E, NU, LOAD = 6.825e7, 0.3, 2.0
REFERENCE = 0.0924


def solve_at(n):
    """Radial deflection under the load on an ``n x n`` octant mesh."""
    mesh = sphere_octant(RADIUS, HOLE_ANGLE, n, cell="triangle")

    rec = csdl.Recorder(inline=True)
    rec.start()

    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=THICKNESS, density=1.0)

    # Symmetry planes at theta=0 (y=0) and theta=90 (x=0)
    bcs = (
        hm.symmetry(domain, where=lambda x: np.isclose(x[1], 0.0), normal=[0, 1, 0])
        + hm.symmetry(domain, where=lambda x: np.isclose(x[0], 0.0), normal=[1, 0, 0])
        # Need to prevent rigid body translation in z. Fix uz at one node, e.g.,
        # the intersection of theta=0 and the hole.
        # Wait, let's fix uz at the hole edge on theta=0.
        # phi = 18 degrees, theta = 0. x = R*sin(18), y = 0, z = R*cos(18)
    )
    
    # Let's try fixing uz at the top of the hole to prevent rigid body z-translation:
    top_z = RADIUS * np.cos(np.radians(HOLE_ANGLE))
    bcs = bcs + hm.pin(domain, where=lambda x: np.isclose(x[2], top_z) & np.isclose(x[1], 0.0), dofs=("uz",))

    # At theta=0 (y=0): inward radial force. Direction is -x.
    # At theta=90 (x=0): outward radial force. Direction is +y.  Both points are
    # shared by two adjacent octants, hence this octant receives P/2 at each.
    load_in = hm.point_load(domain, at=[RADIUS, 0.0, 0.0], force=[-LOAD / 2, 0.0, 0.0])
    load_out = hm.point_load(domain, at=[0.0, RADIUS, 0.0], force=[0.0, LOAD / 2, 0.0])
    
    state = hm.solve(domain, material, load_in + load_out, bcs)
    u = hm.nodal_displacement(state).value.reshape(-1, 3)

    rec.stop()

    k, dist = node_nearest(domain, [RADIUS, 0.0, 0.0])
    if dist > 1e-9:
        print(f"    (note: sample node is {dist:.2e} from the nominal point at n={n})")
    
    # Return inward radial deflection (which is -ux)
    return -u[k, 0]


CASE = Case(
    name="Pinched hemisphere with an 18-degree hole",
    quantity="radial deflection under the load",
    reference=REFERENCE,
    tolerance=0.02,
    citation="MacNeal & Harder, 'A proposed standard set of problems to test finite "
             "element accuracy', Finite Elements in Analysis and Design 1(1), 1985",
    levels=(4, 8, 16, 24, 32, 48, 64, 96),
    quick_level=32,
    solve=solve_at,
    monotone=False,
    notes="inextensional bending; converges slowly. Each load point on a "
          "symmetry plane contributes half of its physical P=2 force to the octant.",
)

if __name__ == "__main__":
    main(CASE)
