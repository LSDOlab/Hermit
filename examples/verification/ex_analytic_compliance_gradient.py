"""Analytic compliance gradient of a thin cantilever plate.

+A clamped, isotropic rectangular plate under uniform normal pressure has one uniform
thickness design variable ``t``.  In pure bending its stiffness is proportional to
``t**3``, hence ``C(t) = k/t**3`` and ``dC/dt = -3*C/t`` exactly.  This example
compares that closed-form derivative with the CSDL total derivative, providing an
adjoint check which is independent of finite differences.

Reissner--Mindlin transverse shear adds a term proportional to ``1/t`` to the
compliance, so the pure-bending identity is not mathematically exact for this shell.
The deliberately thin ``t = 0.01`` plate makes the residual small; the script reports
the measured derivative ratio and infers the corresponding shear fraction
``(3/2) * (1 - ratio)``.  The 1% tolerance is fixed to admit that physical, not
numerical, contamination.
"""

import pathlib
import sys

import csdl_alpha as csdl
import numpy as np

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import rect_plate  # noqa: E402
from _harness import Case, main  # noqa: E402

LENGTH, WIDTH = 4.0, 1.0
E, NU, THICKNESS, PRESSURE = 4.32e8, 0.0, 0.01, 1.0


def solve_at(n):
    """Return ``dC/dt`` divided by the thin-plate pure-bending identity."""
    mesh = rect_plate(LENGTH, WIDTH, nx=4 * n, ny=n, cell="quad")
    rec = csdl.Recorder(inline=True)
    rec.start()
    t = csdl.Variable(value=THICKNESS, name="uniform_thickness")
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=t, density=1.0)
    state = hm.solve(domain, material, hm.pressure(domain, PRESSURE),
                     hm.clamp(domain, where=lambda x: np.isclose(x[0], 0.0)))
    compliance = hm.compliance(state)
    sim = csdl.experimental.PySimulator(rec)
    derivative = float(np.asarray(sim.compute_totals([compliance], [t])[compliance, t]).ravel()[0])
    value = float(np.asarray(compliance.value).ravel()[0])
    rec.stop()

    bending = -3.0 * value / THICKNESS
    ratio = derivative / bending
    shear_fraction = 1.5 * (1.0 - ratio)
    print(f"    C={value:.6e}  dC/dt(CSDL)={derivative:.6e}  "
          f"-3C/t={bending:.6e}  ratio={ratio:.8f}  "
          f"inferred shear fraction={shear_fraction:.3e}")
    return ratio


CASE = Case(
    name="Analytic compliance gradient: uniform thin cantilever",
    quantity="dC/dt (CSDL) / (-3 C/t)",
    reference=1.0,
    tolerance=0.01,
    citation="Euler--Bernoulli pure-bending scaling: D = E t^3 / [12(1-nu^2)]",
    levels=(2, 4, 6),
    quick_level=2,
    solve=solve_at,
    monotone=False,
    notes="remaining discrepancy is quantified Reissner--Mindlin shear, not FD error",
)


if __name__ == "__main__":
    main(CASE)
