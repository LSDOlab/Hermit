"""MacNeal-Harder twisted beam (out-of-plane)

A beam of length 12, width 1.1, and thickness 0.32, twisted by 90 degrees from root to tip.
E = 29e6, nu = 0.22. The beam is clamped at the root and loaded by a unit tip load.
This test evaluates the element's ability to handle a continuous transition between
membrane and bending behaviors due to the twist.

There are two load cases: in-plane (y-direction) and out-of-plane (z-direction) with 
reference tip deflections of 1.754e-3 and 5.424e-3 respectively (MacNeal & Harder 1985).

Reasoning for load directions:
At the root (x=0), the beam lies in the xy plane (width in y, thickness in z). 
"In-plane" refers to loading in this xy plane (y-direction), while "out-of-plane"
refers to loading perpendicular to it (z-direction). 
A tip load in z (out-of-plane) primarily bends the root about its weak axis (y-axis),
resulting in the larger deflection (5.424e-3). 
A tip load in y (in-plane) primarily bends the root about its strong axis (z-axis),
resulting in the smaller deflection (1.754e-3).
This file uses the out-of-plane (load in z) case as the primary benchmark.

    conda activate hermit
    python examples/verification/ex_twisted_beam.py
"""

import pathlib
import sys

import numpy as np
import csdl_alpha as csdl

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import twisted_strip                      # noqa: E402
from _harness import Case, main, node_nearest            # noqa: E402

LENGTH, WIDTH, THICKNESS = 12.0, 1.1, 0.32
TWIST = 90.0
E, NU, LOAD = 29e6, 0.22, 1.0
REFERENCE = 5.424e-3


def solve_at(n):
    """Out-of-plane (z-direction) tip deflection of an ``n x 2`` quad mesh."""
    # The benchmark uses a mesh of N elements along the length and 2 across the width
    mesh = twisted_strip(LENGTH, WIDTH, TWIST, nx=n, ny=n // 6 + 2, cell="triangle")

    rec = csdl.Recorder(inline=True)
    rec.start()

    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=THICKNESS, density=1.0)

    # Clamped at the root (x=0)
    bcs = hm.clamp(domain, where=lambda x: np.isclose(x[0], 0.0))

    # Apply total unit load at the tip in the z direction (out-of-plane).
    xyz = domain.node_coords
    tip_nodes = np.flatnonzero(np.isclose(xyz[:, 0], LENGTH))
    
    # We apply the load statically equivalent by splitting it equally.
    load = None
    for k in tip_nodes:
        p = hm.point_load(domain, at=xyz[k], force=[0.0, 0.0, LOAD / len(tip_nodes)])
        load = p if load is None else load + p
        
    state = hm.solve(domain, material, load, bcs)
    u = hm.nodal_displacement(state).value.reshape(-1, 3)

    rec.stop()

    # The tip deflection is usually measured at the center of the tip
    k, dist = node_nearest(domain, [LENGTH, 0.0, 0.0])
    if dist > 1e-9:
        print(f"    (note: sample node is {dist:.2e} from the nominal point at n={n})")
    
    return abs(u[k, 2])


CASE = Case(
    name="MacNeal-Harder twisted beam (out-of-plane tip load)",
    quantity="tip deflection in z",
    reference=REFERENCE,
    tolerance=0.02,
    citation="MacNeal & Harder, 'A proposed standard set of problems to test finite "
             "element accuracy', Finite Elements in Analysis and Design 1(1), 1985",
    levels=(12, 24, 48, 96),
    quick_level=24,
    solve=solve_at,
    notes="triangles, deliberately: on WARPED quadrilateral cells this case "
          "converges to 7.62e-3, a 40% error that refinement does not fix -- "
          "see ex_warped_quad_consistency",
    monotone=False,
)

if __name__ == "__main__":
    main(CASE)
