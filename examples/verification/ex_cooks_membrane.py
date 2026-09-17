"""Cook's membrane

A tapered panel with corners (0,0), (48,44), (48,60), and (0,44). 
E = 1.0, nu = 1/3, thickness = 1.0. 
The panel is clamped on the left edge (x=0) and loaded by a total shear force P=1 
distributed along the right edge (x=48).
The reference vertical displacement at the midpoint of the right edge (48,52) is approximately 23.9.

This is a membrane-dominated in-plane bending problem. For a shell element, it specifically
exercises the in-plane response (membrane behavior) and the drilling stabilization.
The drilling rotation (rot_z) and out-of-plane DOFs do not require manual constraining;
the element formulation natively provides sufficient drilling stabilization for the solve to succeed.

    conda activate hermit
    python examples/verification/ex_cooks_membrane.py
    python examples/verification/ex_cooks_membrane.py --quick
"""

import pathlib
import sys

import numpy as np
import csdl_alpha as csdl

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import structured_surface                 # noqa: E402
from _harness import Case, main, node_nearest            # noqa: E402

E, NU, THICKNESS = 1.0, 1.0 / 3.0, 1.0
LOAD = 1.0
REFERENCE = 23.9


def cook_map(u, v):
    """Bilinear map of the unit square to the Cook membrane geometry."""
    x = 48.0 * u
    y_bot = 44.0 * u
    y_top = 44.0 + 16.0 * u
    y = y_bot * (1 - v) + y_top * v
    return x, y, np.zeros_like(x)


def solve_at(n):
    """Vertical displacement at the top right corner on an ``n x n`` quad mesh."""
    mesh = structured_surface(cook_map, n, n, cell="quad")

    rec = csdl.Recorder(inline=True)
    rec.start()

    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=THICKNESS, density=1.0)

    # Clamped on the left edge
    bcs = hm.clamp(domain, where=lambda x: np.isclose(x[0], 0.0))

    # Apply statically equivalent total shear force P=1.0 along the right edge (x=48).
    # We use equal splitting over the edge vertices (domain.node_coords only contains vertices).
    # This is a statically equivalent distribution, rather than the exactly consistent CG2
    # weights (1/6, 4/6, 1/6) which would require locating the mid-edge DOFs explicitly.
    xyz = domain.node_coords
    right_edge = np.flatnonzero(np.isclose(xyz[:, 0], 48.0))
    
    load = None
    for k in right_edge:
        p = hm.point_load(domain, at=xyz[k], force=[0.0, LOAD / len(right_edge), 0.0])
        load = p if load is None else load + p
        
    state = hm.solve(domain, material, load, bcs)
    u = hm.nodal_displacement(state).value.reshape(-1, 3)

    rec.stop()

    # Tip displacement at the midpoint of the right edge
    k, dist = node_nearest(domain, [48.0, 52.0, 0.0])
    if dist > 1e-9:
        print(f"    (note: sample node is {dist:.2e} from the nominal point at n={n})")
    
    return u[k, 1]


CASE = Case(
    name="Cook's membrane",
    quantity="vertical displacement at mid-right edge",
    reference=REFERENCE,
    tolerance=0.02,
    citation="Cook, 'Some elements for analysis of plane solid structures', "
             "Int. J. Numer. Methods Eng., 1974",
    levels=(4, 8, 16, 24, 32),
    quick_level=32,
    solve=solve_at,
    notes="converges smoothly to ~23.95 (error ~0.2%).",
    monotone=False,
)

if __name__ == "__main__":
    main(CASE)
