"""Mass and centre of gravity of a plate with uniform and graded thickness

The rectangular plate has ``L = 10``, ``W = 2``, density ``rho = 3``, and base
thickness ``t0 = 0.2``.  It checks a uniform thickness and the field
``t(x) = t0 (1 + x/L)``, constructed with :func:`hm.from_function`.

The closed forms are ``m_uniform = rho W t0 L`` and ``cg_uniform = (L/2, W/2, 0)``;
for the graded field they are ``m = 3 rho W t0 L / 2`` and
``cg = (5 L / 9, W/2, 0)``.  The residual gate is zero-reference and prints mass
and every centre-of-gravity component for both cases.  Its tight tolerance also
guards the user-node to FE-degree-of-freedom ordering path used by the field.
"""

import pathlib
import sys

import csdl_alpha as csdl
import numpy as np

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import rect_plate  # noqa: E402
from _harness import Case, main  # noqa: E402

L, W, T0, RHO = 10.0, 2.0, 0.2, 3.0
UNIFORM_MASS = RHO * W * T0 * L
GRADED_MASS = 1.5 * UNIFORM_MASS
UNIFORM_CG = np.array([L / 2.0, W / 2.0, 0.0])
GRADED_CG = np.array([5.0 * L / 9.0, W / 2.0, 0.0])


def _outputs(domain, thickness):
    material = hm.isotropic(domain, E=1.0, nu=0.0, thickness=thickness, density=RHO)
    state = hm.solve(domain, material, hm.traction(domain, [0.0, 0.0, 0.0]),
                     hm.clamp(domain, where=lambda x: np.isclose(x[0], 0.0)))
    return float(hm.mass(state).value[0]), np.asarray(hm.center_of_gravity(state).value)


def solve_at(n):
    """Maximum absolute residual for both exact mass/CG cases on an ``n x n/2`` mesh."""
    mesh = rect_plate(L, W, nx=n, ny=n // 2, cell="quad")
    rec = csdl.Recorder(inline=True)
    rec.start()
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    uniform = hm.constant(domain, ("Lagrange", 1), T0)
    graded = hm.from_function(domain, ("Lagrange", 1),
                              lambda xyz: T0 * (1.0 + xyz[:, 0] / L))
    mass_u, cg_u = _outputs(domain, uniform)
    mass_g, cg_g = _outputs(domain, graded)
    rec.stop()
    print(f"    uniform: mass={mass_u:.12e}, cg={cg_u}")
    print(f"    graded : mass={mass_g:.12e}, cg={cg_g}")
    return max(abs(mass_u - UNIFORM_MASS), abs(mass_g - GRADED_MASS),
               float(np.max(abs(cg_u - UNIFORM_CG))), float(np.max(abs(cg_g - GRADED_CG))))


CASE = Case(
    name="Plate mass and centre of gravity",
    quantity="maximum absolute residual across uniform and graded exact mass/CG",
    reference=0.0,
    tolerance=1e-10,
    citation="Closed-form area integrals computed in this file",
    levels=(2, 4, 8),
    quick_level=2,
    solve=solve_at,
    monotone=False,
    notes="zero reference means every reported mass and CG component matches its integral",
)

if __name__ == "__main__":
    main(CASE)
