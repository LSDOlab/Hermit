"""Scordelis-Lo roof

The first of the two "shell obstacle course" barrel-vault benchmarks. A cylindrical
roof of radius 25, length 50 and thickness 0.25 spans two rigid end diaphragms; the
straight edges are free. It carries its own weight, 90 per unit area, acting in
global ``-z``. The quantity everyone quotes is the vertical deflection at the midspan
of a free edge: **0.3024**.

The case is membrane-dominated with a significant bending boundary layer near the
free edges, so it punishes an element that cannot represent in-plane stretching and
bending together. It converges from below.

    conda activate hermit
    python examples/verification/ex_scordelis_lo.py            # convergence sweep
    python examples/verification/ex_scordelis_lo.py --quick    # single cheap level
"""

import pathlib
import sys

import numpy as np
import csdl_alpha as csdl

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import cylinder_sector                  # noqa: E402
from _harness import Case, main, node_nearest          # noqa: E402

RADIUS, LENGTH, HALF_ANGLE = 25.0, 50.0, 40.0
E, NU, THICKNESS = 4.32e8, 0.0, 0.25
SELF_WEIGHT = 90.0                                     # force per unit area, -z
REFERENCE = 0.3024


def solve_at(n, *, cell="triangle"):
    """Vertical deflection at the free-edge midspan on an ``n x n`` mesh."""
    mesh = cylinder_sector(RADIUS, LENGTH, HALF_ANGLE, nx=n, nt=n, cell=cell)

    rec = csdl.Recorder(inline=True)
    rec.start()

    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=THICKNESS, density=1.0)
    # Rigid diaphragms: the two curved ends are held in the plane of the diaphragm
    # (v = w = 0) and left free along the axis.
    diaphragms = hm.pin(
        domain,
        where=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], LENGTH),
        dofs=("uy", "uz"),
    )
    # Self weight is a global -z body force per unit area, not a pressure along the
    # shell normal -- on a curved roof those are very different loads.
    state = hm.solve(domain, material,
                     hm.traction(domain, [0.0, 0.0, -SELF_WEIGHT]), diaphragms)
    u = hm.nodal_displacement(state).value.reshape(-1, 3)

    rec.stop()

    y_edge = RADIUS * np.sin(np.radians(HALF_ANGLE))
    k, dist = node_nearest(domain, [LENGTH / 2, y_edge, RADIUS * np.cos(np.radians(HALF_ANGLE))])
    if dist > 1e-9:
        print(f"    (note: sample node is {dist:.2e} from the nominal point at n={n})")
    return abs(u[k, 2])


def make_case(cell="triangle"):
    """Return the Scordelis-Lo case for either supported surface cell type."""
    if cell not in {"triangle", "quad"}:
        raise ValueError(f"unsupported cell type {cell!r}")
    label = ("Scordelis-Lo roof" if cell == "triangle"
             else "Scordelis-Lo roof (quadrilaterals)")
    return Case(
        name=label,
        quantity="vertical deflection at free-edge midspan",
        reference=REFERENCE,
        tolerance=0.02,
        citation="MacNeal & Harder, 'A proposed standard set of problems to test finite "
                 "element accuracy', Finite Elements in Analysis and Design 1(1), 1985",
        levels=(8, 16, 24, 32),
        quick_level=24,
        solve=lambda n: solve_at(n, cell=cell),
        notes=("developable cylindrical surface: its structured quadrilaterals are "
               "planar" if cell == "quad" else
               "converges from below; 0.3086 is the alternative deep-shell-theory value"),
    )


CASE = make_case()

if __name__ == "__main__":
    main(CASE)
