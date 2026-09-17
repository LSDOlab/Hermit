"""Cantilever plate under a tip moment

The 10 by 2 by 0.2 cantilever plate is clamped at ``x = 0`` and carries a
total unit moment about global ``+y`` at its tip.  The resultant is spread over the
tip-edge nodes with *consistent* (trapezoidal) weights rather than equally --
equal splitting is statically equivalent but over-loads the two corner nodes,
which leaves the tip rotation 0.3 % off. See ``_consistent_edge_weights``.
The computed references are ``w_tip = 8.68055556e-5`` from
``M L**2 / (2 E I)`` and ``theta_tip = 1.73611111e-5`` from ``M L / (E I)``,
with ``I = W h**3 / 12``.

This is exact, not merely convergent: constant curvature makes the deflection
quadratic (and CG2 contains it) and the rotation linear (and CG1 contains it),
so the finite element solution is the analytical one up to round-off. Both
quantities land within ~1e-9 at every refinement, and the gate is tight enough
to notice if that ever stops being true.
"""

import pathlib
import sys

import csdl_alpha as csdl
import numpy as np

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import rect_plate  # noqa: E402
from _harness import Case, main, node_nearest  # noqa: E402

L, W, H, E, NU, M = 10.0, 2.0, 0.2, 4.32e8, 0.0, 1.0
I = W * H**3 / 12.0
W_REFERENCE = M * L**2 / (2.0 * E * I)
THETA_REFERENCE = M * L / (E * I)


def _consistent_edge_weights(ys):
    """Normalised trapezoidal weights for a uniform line load on CG1 nodes.

    Splitting the resultant *equally* over the edge nodes is statically equivalent
    but **not consistent**: for a uniform distributed moment on a CG1 field the
    consistent nodal values are trapezoidal (interior nodes get a full element
    length, the two end nodes get half), so equal splitting over-loads the corners
    by a factor of two. That is a self-equilibrated perturbation, so the deflection
    barely notices it -- but the tip *rotation* is measured right where the
    perturbation lives, and it lands 0.3 % off.

    With these weights the rotation is exact to ~2e-10 instead. Measured on this
    fixture: equal splitting gives theta errors of 3.1e-3 / 3.4e-3 / 2.6e-3 at
    n = 8 / 16 / 24; trapezoidal gives 2.7e-10 / 2.0e-9 / 1.6e-9.
    """
    ys = np.asarray(ys, dtype=float)
    h = np.diff(ys)
    w = np.zeros(ys.size)
    w[:-1] += h / 2.0
    w[1:] += h / 2.0
    return w / w.sum()


def solve_at(n):
    """Maximum relative error of tip deflection and rotation on an ``n x n/2`` mesh."""
    mesh = rect_plate(L, W, nx=n, ny=n // 2, cell="quad")
    rec = csdl.Recorder(inline=True)
    rec.start()
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=H, density=1.0)
    xyz = np.asarray(domain.node_coords)
    edge = np.flatnonzero(np.isclose(xyz[:, 0], L))
    edge = edge[np.argsort(xyz[edge, 1])]
    load = None
    for k, weight in zip(edge, _consistent_edge_weights(xyz[edge, 1])):
        term = hm.point_load(domain, at=xyz[k], moment=[0.0, M * weight, 0.0])
        load = term if load is None else load + term
    state = hm.solve(domain, material, load,
                     hm.clamp(domain, where=lambda x: np.isclose(x[0], 0.0)))
    u = hm.nodal_displacement(state).value.reshape(-1, 3)
    theta = hm.nodal_rotation(state).value.reshape(-1, 3)
    rec.stop()
    tip, distance = node_nearest(domain, [L, W / 2.0, 0.0])
    assert distance < 1e-12
    w = abs(float(u[tip, 2]))
    rotation = abs(float(theta[tip, 1]))
    w_error = abs(w / W_REFERENCE - 1.0)
    theta_error = abs(rotation / THETA_REFERENCE - 1.0)
    print(f"    w_tip={w:.8e} (ref {W_REFERENCE:.8e}), "
          f"theta_y={rotation:.8e} (ref {THETA_REFERENCE:.8e})")
    return max(w_error, theta_error)


CASE = Case(
    name="Cantilever plate: total tip moment",
    quantity="maximum relative error of tip deflection and y rotation",
    reference=0.0,
    # The exactness claim is about the discretisation, so the only floor left is
    # floating point in the direct solve; measured residuals are ~1e-9.
    tolerance=1e-7,
    citation="Computed Euler-Bernoulli constant-curvature formulas in this file",
    levels=(8, 12, 16),
    quick_level=16,
    solve=solve_at,
    monotone=False,
    notes="reference zero denotes exact reproduction of both computed quantities",
)

if __name__ == "__main__":
    main(CASE)
