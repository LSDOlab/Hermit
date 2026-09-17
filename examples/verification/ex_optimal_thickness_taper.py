"""Compliance-optimal cantilever taper: recovering the analytic exponent

Minimise the compliance of a uniformly loaded cantilever plate at fixed mass, with the
thickness restricted to the one-parameter family ``t(x) = A ((L - x) / L + eps) ** p``,
and check that the optimiser recovers the analytic exponent **p = 1**.

In the beam limit the bending stiffness goes as ``t**3`` and the compliance is
``C = int M(x)**2 / t(x)**3 dx``. Minimising that at fixed ``int t dx`` gives
``-3 M**2 / t**4 + lam = 0``, so ``t**4`` is proportional to ``M**2`` and ``t`` to
``sqrt(M)``. This is also what the fully stressed condition ``sigma = 6M/(w t**2)``
= const gives. Under uniform load ``M(x) = q (L - x)**2 / 2``, so the optimum is the
**linear wedge** ``t proportional to (L - x)``, i.e. ``p = 1``. (Verified independently
against a direct 1-D SLSQP solve of the same functional: relative L2 error 2.4e-6.)

Why the design space is parameterised rather than free
------------------------------------------------------

An earlier version of this example gave every CG1 node its own thickness and compared
the result against the wedge. That comparison is not meaningful, and the reason is
worth stating because it is easy to mistake for an optimiser bug.

Bending stiffness goes as ``int t**3`` while mass goes as ``int t``. At fixed mass,
making the thickness *oscillate* therefore raises the stiffness -- so a free nodal
design space has oscillatory, mesh-dependent optima that genuinely beat any smooth
profile in the discrete objective. Measured on this fixture at equal mass: uniform
``C = 0.056408``, the analytic wedge ``C = 0.017066``, and the free-nodal optimum
``C = 0.003685`` -- the checkerboard is five times better than the wedge, and it is
*right* to be. The optimiser was correct; the target was not.

This is the classical ill-posedness of unregularised thickness design, and the usual
remedies are a filter, a perimeter penalty, or a restricted design space. The last is
used here because it also makes the example sharper: instead of comparing profiles, the
optimiser is asked to *find* the exponent, and the analytic result says what it must
find. It also exercises exactly what Hermit exists for -- CSDL total derivatives
through the shell solve driving a real optimiser.

Diagnostics behind that conclusion, for the record: the mass gradient agrees with
central finite differences to 1.6e-8, every entry is strictly positive and the entries
sum to ``rho * area``; no file/FE ordering permutation explains the oscillation; and
starting the free-nodal solve *at* the wedge moves away from it to a lower-compliance
oscillatory design rather than staying put.

    conda activate hermit
    python examples/verification/ex_optimal_thickness_taper.py
"""

import pathlib
import sys

import numpy as np
import csdl_alpha as csdl

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import rect_plate                        # noqa: E402
from _harness import Case, main                         # noqa: E402

LENGTH, WIDTH = 4.0, 1.0
E, NU, H0, RHO, PRESSURE = 4.32e8, 0.0, 0.05, 1.0, 1.0
EPS = 1.0e-3          # keeps the tip thickness positive so log() is defined
ANALYTIC_P = 1.0


def solve_at(n):
    """Optimise ``(A, p)`` on a ``4n x n`` plate; return the recovered exponent ``p``."""
    mesh = rect_plate(LENGTH, WIDTH, nx=4 * n, ny=n, cell="quad")

    rec = csdl.Recorder(inline=True)
    rec.start()

    domain = hm.ShellDomain(mesh, element="CG2CG1")
    xi = (LENGTH - np.asarray(domain.node_coords)[:, 0]) / LENGTH + EPS

    # t(x) = A * xi**p, with p free. A variable exponent needs exp(p log xi); xi is a
    # constant array, so this stays a clean CSDL expression in the two design variables.
    scale = csdl.Variable(value=np.array([H0]), name="scale")
    exponent = csdl.Variable(value=np.array([0.3]), name="exponent")   # deliberately
    # started away from the answer, so recovering p = 1 is a real result and not the
    # initial guess surviving.
    log_xi = csdl.Variable(value=np.log(xi))
    thickness = csdl.expand(scale, xi.shape) * csdl.exp(csdl.expand(exponent, xi.shape) * log_xi)

    material = hm.isotropic(domain, E=E, nu=NU,
                            thickness=hm.from_nodal(domain, thickness), density=RHO)
    state = hm.solve(domain, material, hm.pressure(domain, PRESSURE),
                     hm.clamp(domain, where=lambda x: np.isclose(x[0], 0.0)))
    compliance, mass = hm.compliance(state), hm.mass(state)

    mass_target = RHO * H0 * LENGTH * WIDTH
    scale.set_as_design_variable(lower=1e-3, upper=1.0)
    exponent.set_as_design_variable(lower=0.0, upper=3.0)
    mass.set_as_constraint(lower=mass_target, upper=mass_target)
    compliance.set_as_objective()

    from modopt import CSDLAlphaProblem, PySLSQP

    simulator = csdl.experimental.PySimulator(rec)
    problem = CSDLAlphaProblem(problem_name="hermit_taper_exponent", simulator=simulator)
    PySLSQP(problem, solver_options={"maxiter": 200, "acc": 1e-12}).solve()

    p = float(np.ravel(exponent.value)[0])
    c = float(np.ravel(compliance.value)[0])
    m = float(np.ravel(mass.value)[0])
    rec.stop()

    print(f"    recovered p={p:.6f} (analytic {ANALYTIC_P})  "
          f"compliance={c:.6e}  mass={m:.6f} (target {mass_target:.6f})")
    return p


CASE = Case(
    name="Compliance-optimal taper: recovered exponent",
    quantity="optimiser-recovered exponent p in t(x) = A ((L-x)/L)**p",
    reference=ANALYTIC_P,
    # The optimiser has to find p through the shell solve's adjoint, and the plate is
    # not exactly the beam the analytic result assumes (finite width, transverse
    # shear, a non-zero tip thickness from EPS), so a few percent is the honest gate.
    tolerance=0.05,
    citation="Computed in this file: minimising int M^2/t^3 at fixed int t gives "
             "t proportional to sqrt(M), hence (L-x) under uniform load; cross-checked "
             "against a direct 1-D SLSQP solve to 2.4e-6",
    levels=(2, 3, 4),
    quick_level=3,
    solve=solve_at,
    monotone=False,
    # The optimiser stack is an optional Hermit extra, so a plain install -- and CI's
    # environment -- legitimately lacks it. Every other case here needs only the
    # solver itself; this is the one that drives it from an optimiser.
    requires=("modopt", "pyslsqp"),
    notes="p starts at 0.3, deliberately away from the answer. A free per-node "
          "thickness design space is ill-posed here and does NOT converge to this "
          "wedge -- see the module docstring; that is a property of the objective, "
          "not an optimiser fault.",
)

if __name__ == "__main__":
    main(CASE)
