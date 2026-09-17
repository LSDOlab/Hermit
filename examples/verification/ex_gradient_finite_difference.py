"""Nodal-thickness compliance gradient versus central finite differences.

The cantilever thickness is a CG1 nodal CSDL design variable, supplied through
``hm.from_nodal``.  The CSDL total derivative of compliance is checked at three
interior design dofs against central differences.  A fixed step study is printed:
large steps expose Taylor truncation, while the smallest steps eventually expose
subtractive cancellation.  The reported Case quantity is the best relative agreement
over that study; its tolerance is intentionally an FD tolerance, not a solver claim.
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
E, NU, THICKNESS, PRESSURE = 4.32e8, 0.0, 0.05, 1.0
RELATIVE_STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def _compliance(mesh, thickness, want_gradient=False):
    rec = csdl.Recorder(inline=True)
    rec.start()
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    t = csdl.Variable(value=thickness.copy(), name="nodal_thickness")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=hm.from_nodal(domain, t), density=1.0)
    state = hm.solve(domain, material, hm.pressure(domain, PRESSURE),
                     hm.clamp(domain, where=lambda x: np.isclose(x[0], 0.0)))
    output = hm.compliance(state)
    value = float(np.asarray(output.value).ravel()[0])
    grad = None
    if want_gradient:
        grad = np.asarray(csdl.experimental.PySimulator(rec).compute_totals([output], [t])[output, t]).ravel()
    rec.stop()
    return value, grad


def solve_at(n):
    """Return the best CSDL/central-difference agreement for three nodal dofs."""
    mesh = rect_plate(LENGTH, WIDTH, nx=4 * n, ny=n, cell="quad")
    nn = mesh.geometry.x.shape[0]
    t0 = THICKNESS * np.ones(nn)
    _, gradient = _compliance(mesh, t0, want_gradient=True)
    xyz = mesh.geometry.x
    # Interior points avoid both the clamped boundary and a free-edge corner.
    targets = ([LENGTH / 4, WIDTH / 2, 0.0], [LENGTH / 2, WIDTH / 2, 0.0],
               [3 * LENGTH / 4, WIDTH / 2, 0.0])
    dofs = [int(np.argmin(np.linalg.norm(xyz - p, axis=1))) for p in targets]
    best = np.inf
    print("    relative step       max relative error at dofs", dofs)
    for rel_step in RELATIVE_STEPS:
        step = rel_step * THICKNESS
        errors = []
        for k in dofs:
            tp, tm = t0.copy(), t0.copy()
            tp[k] += step
            tm[k] -= step
            cp, _ = _compliance(mesh, tp)
            cm, _ = _compliance(mesh, tm)
            fd = (cp - cm) / (2.0 * step)
            errors.append(abs(gradient[k] - fd) / max(abs(fd), 1e-14))
        worst = max(errors)
        best = min(best, worst)
        print(f"    {rel_step:13.1e}       {worst:.3e}")
    print(f"    best relative agreement = {best:.3e}")
    return 1.0 - best


CASE = Case(
    name="Nodal thickness gradient: central finite differences",
    quantity="1 - best relative CSDL/FD gradient error",
    reference=1.0,
    tolerance=2e-3,
    citation="Second-order central-difference formula, evaluated by this script",
    levels=(2, 3),
    quick_level=2,
    solve=solve_at,
    monotone=False,
    notes="step study separates FD truncation and cancellation; tolerance is FD-limited",
)


if __name__ == "__main__":
    main(CASE)
