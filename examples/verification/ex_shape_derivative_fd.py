"""Mesh-coordinate compliance derivative versus central finite differences.

The same clamped plate is first given a gentle smooth bend, then differentiated with
respect to the ``z`` coordinate at three interior mesh nodes.  Geometry is made live with
``hm.geometry(domain, node_disp=shape)``; the CSDL total therefore includes both the
shape derivative of the shell solve and that of compliance.  Central finite
differences use a fixed step study, reporting the best agreement rather than hiding
the truncation/cancellation trade-off.

The node perturbations are deliberately small.  Reissner--Mindlin penalty systems
become ill-conditioned for large local warps, and UFL omits the non-smooth derivative
of the stabilization cell-diameter scale; neither effect is an adjoint accuracy test.
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
STEPS = (1e-2, 1e-3, 1e-4, 1e-5)


def _compliance(mesh, shape, want_gradient=False):
    rec = csdl.Recorder(inline=True)
    rec.start()
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    nd = csdl.Variable(value=shape.copy(), name="node_disp")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=THICKNESS, density=1.0)
    state = hm.solve(domain, material, hm.pressure(domain, PRESSURE),
                     hm.clamp(domain, where=lambda x: np.isclose(x[0], 0.0)),
                     geometry=hm.geometry(domain, node_disp=nd))
    output = hm.compliance(state)
    value = float(np.asarray(output.value).ravel()[0])
    grad = None
    if want_gradient:
        grad = np.asarray(csdl.experimental.PySimulator(rec).compute_totals([output], [nd])[output, nd])
    rec.stop()
    return value, grad


def solve_at(n):
    """Return best agreement for individual interior ``z`` coordinate derivatives."""
    mesh = rect_plate(LENGTH, WIDTH, nx=4 * n, ny=n, cell="quad")
    nn = mesh.geometry.x.shape[0]
    # At a perfectly flat configuration these individual out-of-plane derivatives
    # vanish by reflection symmetry.  A gentle, smooth baseline bend makes the local
    # coordinate sensitivities nonzero without entering the penalty-conditioning
    # regime warned about in the shape-derivative documentation.
    shape0 = np.zeros((nn, 3))
    shape0[:, 2] = 0.10 * (mesh.geometry.x[:, 0] / LENGTH) ** 2
    shape0[:, 0] = 0.005 * mesh.geometry.x[:, 0] / LENGTH
    _, gradient = _compliance(mesh, shape0, want_gradient=True)
    xyz = mesh.geometry.x
    targets = ([LENGTH / 4, WIDTH / 2, 0.0], [LENGTH / 2, WIDTH / 2, 0.0],
               [3 * LENGTH / 4, WIDTH / 2, 0.0])
    nodes = [int(np.argmin(np.linalg.norm(xyz - p, axis=1))) for p in targets]
    best = np.inf
    print("    absolute step       max relative error at nodes", nodes, "(z coordinate)")
    for step in STEPS:
        errors = []
        for k in nodes:
            plus, minus = shape0.copy(), shape0.copy()
            plus[k, 2] += step
            minus[k, 2] -= step
            cp, _ = _compliance(mesh, plus)
            cm, _ = _compliance(mesh, minus)
            fd = (cp - cm) / (2.0 * step)
            errors.append(abs(gradient.reshape(nn, 3)[k, 2] - fd) / max(abs(fd), 1e-14))
        worst = max(errors)
        best = min(best, worst)
        print(f"    {step:13.1e}       {worst:.3e}")
    print(f"    best relative agreement = {best:.3e}")
    return 1.0 - best


CASE = Case(
    name="Shape derivative: nodal coordinates versus central finite differences",
    quantity="1 - best relative CSDL/FD shape-gradient error",
    reference=1.0,
    tolerance=5e-3,
    citation="UFL CoordinateDerivative; central differences evaluated by this script",
    levels=(2, 3),
    quick_level=2,
    solve=solve_at,
    monotone=False,
    notes="small local warps avoid penalty-conditioning artifacts; tolerance is FD-limited",
)


if __name__ == "__main__":
    main(CASE)
