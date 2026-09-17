"""MacNeal-Harder curved beam (out-of-plane load)

A 90-degree curved beam with inner radius 4.12, outer radius 4.32, and thickness 0.1,
lying in the xy plane. E = 1e7, nu = 0.25. The beam is clamped at one end (theta=0)
and loaded by a unit tip load at the free end (theta=90).
This is deliberately a very high aspect ratio, single-element-wide strip, providing 
a severe test for shear and membrane locking.

There are two load cases: in-plane and out-of-plane, with references 0.08734 and 
0.5022 respectively. 
Because the beam lies in the xy-plane (z=0), an out-of-plane load unequivocally acts 
in the global z-direction. The in-plane load could be applied in various directions 
(e.g., radial or tangential), making its definition slightly more ambiguous. 
Therefore, this file uses the out-of-plane (z-direction) load case as it is structurally 
unambiguous. A load applied uniformly across the free tip's nodes is used.

    conda activate hermit
    python examples/verification/ex_curved_beam.py
    python examples/verification/ex_curved_beam.py --quick
"""

import pathlib
import sys

import numpy as np
import csdl_alpha as csdl

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import curved_beam                        # noqa: E402
from _harness import Case, main, node_nearest            # noqa: E402

R_INNER, R_OUTER = 4.12, 4.32
THICKNESS = 0.1
E, NU, LOAD = 1e7, 0.25, 1.0
REFERENCE = 0.5022


def solve_at(n):
    """Out-of-plane (z-direction) tip deflection on a 1 x ``n`` quad mesh."""
    # curved_beam uses nr for radial elements, ns for tangential (sweep) elements
    mesh = curved_beam(R_INNER, R_OUTER, sweep=90.0, nr=1, ns=n, cell="quad")

    rec = csdl.Recorder(inline=True)
    rec.start()

    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=THICKNESS, density=1.0)

    # Clamped at theta=0 (which corresponds to y=0 and x>0)
    bcs = hm.clamp(domain, where=lambda x: np.isclose(x[1], 0.0) & (x[0] > 0))

    # Apply total unit load at the free end (theta=90, corresponds to x=0 and y>0).
    # We spread the load statically equivalent across the nodes at the free tip.
    xyz = domain.node_coords
    tip_nodes = np.flatnonzero(np.isclose(xyz[:, 0], 0.0) & (xyz[:, 1] > 0))
    
    load = None
    for k in tip_nodes:
        p = hm.point_load(domain, at=xyz[k], force=[0.0, 0.0, LOAD / len(tip_nodes)])
        load = p if load is None else load + p
        
    state = hm.solve(domain, material, load, bcs)
    u = hm.nodal_displacement(state).value.reshape(-1, 3)

    rec.stop()

    # The tip deflection is usually measured at the centroid of the free end
    k, dist = node_nearest(domain, [0.0, (R_INNER + R_OUTER) / 2.0, 0.0])
    if dist > 1e-9:
        print(f"    (note: sample node is {dist:.2e} from the nominal point at n={n})")
    
    return abs(u[k, 2])


CASE = Case(
    name="MacNeal-Harder curved beam (out-of-plane tip load)",
    quantity="tip deflection in z",
    reference=REFERENCE,
    tolerance=0.02,
    citation="MacNeal & Harder, 'A proposed standard set of problems to test finite "
             "element accuracy', Finite Elements in Analysis and Design 1(1), 1985",
    levels=(6, 12, 24, 48),
    quick_level=24,
    solve=solve_at,
    notes="single-element-wide strip is a severe test for locking; converges from below to ~0.482 (error ~3.9% at centroid)",
    monotone=True,
    open_finding=(
        "Ruled out a shared cause with Cook's membrane (which was purely a measurement point error). "
        "For this curved beam, measuring at the inner radius instead of the centroid exaggerated the error "
        "(0.476 vs 0.482). The remaining deficit at the centroid (plateaus at ~0.482, 3.9% low) is a genuine "
        "element limitation on this deliberately extreme 1-element-wide strip. Refining the width (nr=4) "
        "closes the gap to 0.492 (1.9% error), and the in-plane load case converges to ~0.0885 (1.3% error). "
        "So this is a specific limitation for out-of-plane twisting on extreme aspect ratios, not a general "
        "membrane offset."
    ),
)

if __name__ == "__main__":
    main(CASE)
