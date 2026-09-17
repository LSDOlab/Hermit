"""Thickness optimization

Minimise the compliance of a cantilever plate at fixed mass. Mirrors the optimization
block of femo_alpha ``ex_simple_shell.py``: nodal (CG1) thickness design variables, an
equality mass constraint at the baseline mass, PySLSQP.

What this example demonstrates is the *plumbing* -- CSDL total derivatives through the
shell solve driving a real optimiser. It is deliberately **not** a demonstration of a
well-posed design problem, and the result is worth understanding before you copy the
setup.

A free per-node thickness field is the classical ill-posed thickness design problem.
Bending stiffness goes as ``int t**3`` while mass goes as ``int t``, so at fixed mass
an *oscillating* thickness genuinely raises the stiffness: the optimum is a
mesh-dependent checkerboard that really does beat any smooth profile in the discrete
objective. The optimiser is right; the design space is the problem. Do not expect the
root-thick / tip-thin wedge that beam theory gives -- see
``examples/verification/ex_optimal_thickness_taper.py``, which measures the gap (at
equal mass on its fixture: uniform 0.056, analytic wedge 0.017, free-nodal 0.0037) and
recovers the analytic answer by restricting the design space instead.

The usual remedies are a filter, a perimeter penalty, or a restricted design space.

    conda activate hermit
    python examples/advanced_examples/ex_thickness_opt.py
"""

import pathlib

import numpy as np
import csdl_alpha as csdl

import hermit as hm

MESH = pathlib.Path(__file__).parents[2] / "tests" / "meshes" / "plate_2x10_quad_4x20.xdmf"

E_VAL, NU_VAL, H_VAL, RHO_VAL = 4.32e8, 0.0, 0.2, 1.0
PRESSURE_Z = 2.0
WIDTH, LENGTH = 2.0, 10.0


def clamped_at_x0(x):
    return np.less(x[0], 1e-12)


def main():
    mesh = hm.read_mesh(MESH)

    rec = csdl.Recorder(inline=True)
    rec.start()

    domain = hm.ShellDomain(mesh, element="CG2CG1")
    thickness = csdl.Variable(value=H_VAL * np.ones(domain.n_nodes), name="thickness")
    material = hm.isotropic(domain, E=E_VAL, nu=NU_VAL,
                            thickness=hm.from_nodal(domain, thickness), density=RHO_VAL)
    state = hm.solve(domain, material, hm.pressure(domain, PRESSURE_Z),
                     hm.clamp(domain, where=clamped_at_x0))
    compliance, mass = hm.compliance(state), hm.mass(state)

    print(f"baseline: compliance={float(np.ravel(compliance.value)[0]):.6e}  "
          f"mass={float(np.ravel(mass.value)[0]):.6e}")

    mass_0 = RHO_VAL * H_VAL * WIDTH * LENGTH
    thickness.set_as_design_variable(lower=1e-2, upper=10.0)
    mass.set_as_constraint(lower=mass_0, upper=mass_0)
    compliance.set_as_objective()

    from modopt import CSDLAlphaProblem, PySLSQP

    sim = csdl.experimental.PySimulator(rec)
    prob = CSDLAlphaProblem(problem_name="hermit_plate_thickness", simulator=sim)
    optimizer = PySLSQP(prob, solver_options={"maxiter": 200, "acc": 1e-9})
    optimizer.solve()
    optimizer.print_results()

    rec.stop()

    print("optimized:")
    print(f"  compliance : {float(np.ravel(compliance.value)[0]):.6e}")
    print(f"  mass       : {float(np.ravel(mass.value)[0]):.6e}  (target {mass_0})")
    print(f"  thickness  : min {thickness.value.min():.4f}  max {thickness.value.max():.4f}")


if __name__ == "__main__":
    main()
