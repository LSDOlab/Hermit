"""Pinched hemisphere with an 18-degree hole (quadrilaterals)

The MacNeal-Harder pinched hemisphere benchmark, with the same geometry, material,
loading, published radial-deflection reference (**0.0924**), and symmetry reduction
as ``ex_pinched_hemisphere.py``. Unlike a cylinder, a sphere is doubly curved: its
structured quadrilaterals are warped. It measures the global response alongside the
triangle case after the constant-normal curvature fix for LSDOlab/Hermit#7.
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
    """Radial deflection under the load on an ``n x n`` warped-quad octant mesh."""
    mesh = sphere_octant(RADIUS, HOLE_ANGLE, n, cell="quad")

    rec = csdl.Recorder(inline=True)
    rec.start()

    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=THICKNESS, density=1.0)
    bcs = (
        hm.symmetry(domain, where=lambda x: np.isclose(x[1], 0.0), normal=[0, 1, 0])
        + hm.symmetry(domain, where=lambda x: np.isclose(x[0], 0.0), normal=[1, 0, 0])
    )
    top_z = RADIUS * np.cos(np.radians(HOLE_ANGLE))
    bcs = bcs + hm.pin(
        domain,
        where=lambda x: np.isclose(x[2], top_z) & np.isclose(x[1], 0.0),
        dofs=("uz",),
    )
    load_in = hm.point_load(domain, at=[RADIUS, 0.0, 0.0], force=[-LOAD / 2, 0.0, 0.0])
    load_out = hm.point_load(domain, at=[0.0, RADIUS, 0.0], force=[0.0, LOAD / 2, 0.0])
    state = hm.solve(domain, material, load_in + load_out, bcs)
    u = hm.nodal_displacement(state).value.reshape(-1, 3)

    rec.stop()

    k, dist = node_nearest(domain, [RADIUS, 0.0, 0.0])
    if dist > 1e-9:
        print(f"    (note: sample node is {dist:.2e} from the nominal point at n={n})")
    return -u[k, 0]


CASE = Case(
    name="Pinched hemisphere with an 18-degree hole (warped quadrilaterals)",
    quantity="radial deflection under the load",
    reference=REFERENCE,
    tolerance=0.02,
    citation="MacNeal & Harder, 'A proposed standard set of problems to test finite "
             "element accuracy', Finite Elements in Analysis and Design 1(1), 1985",
    levels=(4, 8, 16, 24, 32, 48, 64, 96),
    quick_level=32,
    solve=solve_at,
    monotone=False,
    notes="same physical benchmark as the triangular case; spherical quad cells are warped. "
          "The constant-normal formulation remains within the published tolerance; "
          "direct energy and recovered-stress objectivity gates are in "
          "tests/test_prescribed_bcs.py.",
)


if __name__ == "__main__":
    main(CASE)
