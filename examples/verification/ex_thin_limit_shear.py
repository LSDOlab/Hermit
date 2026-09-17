"""Thin-limit transverse-shear study for a simply supported plate.

This fixes a reasonably fine square mesh and reuses it while the Case ``level``
denotes slenderness ``a/h`` rather than mesh size.  Each plate is compared with the
analytic Reissner--Mindlin one-mode answer (Kirchhoff bending plus transverse shear)
under sinusoidal pressure.  At the thin end, ``a/h = 500``, the Kirchhoff component
dominates; that is precisely where a locking element would spuriously stiffen.  The
``a/h = 5`` and 10 endpoints intentionally retain appreciable shear deformation;
the analytic shear term makes them useful controls for the thin-plate cases.

As in the simply-supported plate examples, this is the soft support: edge ``uz`` is
held, rotations are free, and three in-plane point gauges remove only rigid motion.
``monotone=False`` is intentional because these levels are material slendernesses,
not a mesh-refinement sequence.
"""

import pathlib
import sys

import csdl_alpha as csdl
import numpy as np

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import rect_plate                    # noqa: E402
from _harness import Case, main, node_nearest        # noqa: E402

A = B = 1.0
E, NU, Q0 = 1.0e7, 0.3, 1.0e3
MESH_LEVEL = 24


def _reference(slenderness):
    """Exact one-mode Reissner--Mindlin deflection: bending plus shear."""
    h = A / float(slenderness)
    d = E * h**3 / (12.0 * (1.0 - NU**2))
    g = E / (2.0 * (1.0 + NU))
    kappa = 5.0 / 6.0
    k2 = np.pi**2 * (1.0 / A**2 + 1.0 / B**2)
    return Q0 / (d * k2**2) + Q0 / (kappa * g * h * k2)


def _soft_simple_support(domain):
    edge = lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[0], A)
                      | np.isclose(x[1], 0.0) | np.isclose(x[1], B))
    return (hm.pin(domain, where=edge, dofs=("uz",))
            + hm.gauge(domain, at=[0.0, 0.0, 0.0], dofs=("ux", "uy"))
            + hm.gauge(domain, at=[A, 0.0, 0.0], dofs=("uy",)))


def solve_at(slenderness):
    """Return FE/reference ratio for one fixed mesh at ``a/h = slenderness``."""
    h = A / float(slenderness)
    mesh = rect_plate(A, B, nx=MESH_LEVEL, ny=MESH_LEVEL, cell="quad")
    rec = csdl.Recorder(inline=True)
    rec.start()
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=h, density=1.0)
    traction = hm.from_function(
        domain, ("Lagrange", 2, (3,)),
        lambda x: np.column_stack((np.zeros(len(x)), np.zeros(len(x)),
                                   Q0 * np.sin(np.pi * x[:, 0] / A)
                                   * np.sin(np.pi * x[:, 1] / B))),
    )
    state = hm.solve(domain, material, hm.traction(domain, traction),
                     _soft_simple_support(domain))
    u = hm.nodal_displacement(state).value.reshape(-1, 3)
    rec.stop()
    k, dist = node_nearest(domain, [A / 2, B / 2, 0.0])
    if dist > 1e-9:
        print(f"    (note: centre sample node is {dist:.2e} away at a/h={slenderness})")
    value = abs(u[k, 2])
    ref = _reference(slenderness)
    print(f"    a/h={slenderness:>3}: FE={value:.6e}, RM reference={ref:.6e}, "
          f"FE/reference={value / ref:.6f}")
    return value / ref


CASE = Case(
    name="Simply supported plate thin-limit shear study",
    quantity="FE / analytic Reissner--Mindlin centre-deflection ratio",
    reference=1.0,
    tolerance=0.02,
    citation="Computed Reissner--Mindlin single mode: Kirchhoff bending + kappa G h shear",
    levels=(5, 10, 50, 100, 500),
    quick_level=50,
    solve=solve_at,
    monotone=False,
    notes="level means a/h, not mesh size; fixed 24x24 mesh; a falling thin-end ratio is locking",
)

if __name__ == "__main__":
    main(CASE)
