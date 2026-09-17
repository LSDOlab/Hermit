"""Elastic-energy and compliance identity

For Hermit's linear static solve, :func:`hm.compliance` is the applied-load work:
``outputs.py`` includes distributed terms in its compliance form and adds the
direct point-load vector product separately.  Stored elastic energy must therefore
equal one half of that work.  This example checks the identity independently for
a pressure load and for total tip point loads split equally over the tip edge.

There is no external reference value: the zero-reference scalar is the largest
relative identity residual, while the printed values make the convention visible.
"""

import pathlib
import sys

import csdl_alpha as csdl
import numpy as np

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import rect_plate  # noqa: E402
from _harness import Case, main  # noqa: E402

L, W, H, E, NU = 10.0, 2.0, 0.2, 4.32e8, 0.0


def _identity(state):
    compliance = float(hm.compliance(state).value[0])
    energy = float(hm.elastic_energy(state).value[0])
    return compliance, energy, abs(energy / (0.5 * compliance) - 1.0)


def solve_at(n):
    """Largest pressure/point-load energy-identity residual on an ``n x n/2`` mesh."""
    mesh = rect_plate(L, W, nx=n, ny=n // 2, cell="quad")
    rec = csdl.Recorder(inline=True)
    rec.start()
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=H, density=1.0)
    bcs = hm.clamp(domain, where=lambda x: np.isclose(x[0], 0.0))
    pressure_state = hm.solve(domain, material, hm.pressure(domain, 2.0), bcs)
    xyz = np.asarray(domain.node_coords)
    edge = np.flatnonzero(np.isclose(xyz[:, 0], L))
    point_loads = None
    for k in edge:
        term = hm.point_load(domain, at=xyz[k], force=[0.0, 0.0, -1.0 / len(edge)])
        point_loads = term if point_loads is None else point_loads + term
    point_state = hm.solve(domain, material, point_loads, bcs)
    cp, up, ep = _identity(pressure_state)
    cf, uf, ef = _identity(point_state)
    rec.stop()
    print(f"    pressure: compliance={cp:.12e}, energy={up:.12e}, residual={ep:.3e}")
    print(f"    points  : compliance={cf:.12e}, energy={uf:.12e}, residual={ef:.3e}")
    return max(ep, ef)


CASE = Case(
    name="Elastic energy equals half compliance",
    quantity="largest relative energy-identity residual for pressure and point loads",
    reference=0.0,
    # Not a fitted tolerance: the identity is exact in exact arithmetic, so the
    # only floor is round-off in the direct solve of a thin-shell operator.
    # Measured at ~3e-9, and -- the point -- *identical* for a penalty clamp and a
    # strong clamp (2.0e-10 / 3.0e-9 / 2.9e-9 vs 4.0e-10 / 3.1e-9 / 3.0e-9 at
    # n = 4 / 8 / 12), which rules out the penalty parameter as the cause.
    tolerance=1e-7,
    citation="Hermit outputs.py: elastic_energy and compliance conventions",
    levels=(4, 8, 12),
    quick_level=4,
    solve=solve_at,
    monotone=False,
    notes="residual is round-off in the direct solve, not a discretisation error: "
          "it does not shrink under refinement and is the same for penalty and "
          "strong clamps",
)

if __name__ == "__main__":
    main(CASE)
