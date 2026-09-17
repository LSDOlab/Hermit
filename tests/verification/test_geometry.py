"""Sanity gates on the analytic mesh builders every benchmark depends on.

The surface area check is not busywork. ``create_mesh`` takes cell vertices in
**basix** order, and for a quadrilateral that is the tensor product
``(0,0), (1,0), (0,1), (1,1)`` -- not the cyclic order meshio/XDMF uses. Handing it
cyclic quads builds tangled bowtie cells: the mesh constructs, ``ShellDomain``
accepts it, the solve converges, and displacements come out roughly seven orders of
magnitude wrong. Nothing downstream notices. Integrating ``1 * dx`` does: a bowtie's
two lobes cancel, so the total area collapses.
"""

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "examples" / "verification"))

from _geometry import (  # noqa: E402
    annulus,
    curved_beam,
    cylinder_sector,
    disk,
    rect_plate,
    sphere_octant,
    twisted_strip,
)


def area(mesh):
    """Surface area, ``sum(1 * dx)``."""
    import ufl
    from dolfinx.fem import assemble_scalar, form

    return assemble_scalar(form(1.0 * ufl.dx(mesh)))


def min_cell_area(mesh):
    """Smallest cell area -- a tangled cell shows up here as ~0 even when the total
    happens to look plausible."""
    import ufl
    from dolfinx.fem import Function, functionspace, assemble_scalar, form
    from dolfinx.fem.petsc import assemble_vector

    V = functionspace(mesh, ("DG", 0))
    v = ufl.TestFunction(V)
    areas = assemble_vector(form(v * ufl.dx))
    areas.assemble()
    return float(np.min(areas.array))


CASES = [
    # (builder, analytic area, relative tolerance)
    (lambda: rect_plate(10.0, 2.0, nx=20, ny=4, cell="quad"), 20.0, 1e-12),
    (lambda: rect_plate(10.0, 2.0, nx=20, ny=4, cell="triangle"), 20.0, 1e-12),
    (lambda: cylinder_sector(25.0, 50.0, 40.0, nx=24, nt=24),
     25.0 * math.radians(80.0) * 50.0, 2e-3),
    (lambda: cylinder_sector(300.0, 300.0, nx=24, nt=24, theta0=0.0, theta1=90.0),
     300.0 * (math.pi / 2) * 300.0, 2e-3),
    (lambda: sphere_octant(10.0, 18.0, n=24),
     10.0**2 * (math.pi / 2) * math.cos(math.radians(18.0)), 5e-3),
    (lambda: disk(1.0, nr=8, nt=48), math.pi, 5e-3),
    (lambda: annulus(1.0, 2.0, nr=8, nt=48, cell="quad"), math.pi * 3.0, 5e-3),
    (lambda: curved_beam(4.12, 4.32, 90.0, nr=2, ns=24),
     math.radians(90.0) * (4.32**2 - 4.12**2) / 2, 5e-3),
]


@pytest.mark.parametrize("build,expected,rtol", CASES,
                         ids=[f"case{i}" for i in range(len(CASES))])
def test_surface_area_matches_analytic(build, expected, rtol):
    mesh = build()
    got = area(mesh)
    assert got == pytest.approx(expected, rel=rtol), (
        f"area {got:.6g} != analytic {expected:.6g} -- tangled cells or a bad "
        f"parametrisation (see this module's docstring)"
    )


@pytest.mark.parametrize("build,expected,rtol", CASES,
                         ids=[f"case{i}" for i in range(len(CASES))])
def test_no_degenerate_cells(build, expected, rtol):
    mesh = build()
    assert min_cell_area(mesh) > 0.0, "a cell has non-positive area (tangled winding)"


def test_twisted_strip_area_slightly_exceeds_flat():
    """A twisted strip is developable-ish but not flat: its area is close to, and not
    below, the untwisted L*W."""
    flat = 12.0 * 1.1
    got = area(twisted_strip(12.0, 1.1, 90.0, nx=48, ny=4))
    assert flat <= got < flat * 1.05


def test_wrap_v_does_not_duplicate_the_seam():
    """The closed annulus must weld v=1 onto v=0 rather than leaving a coincident
    duplicate row of nodes (which would split the shell along the seam)."""
    m = annulus(1.0, 2.0, nr=4, nt=16, cell="quad")
    n_nodes = m.topology.index_map(0).size_local
    assert n_nodes == 5 * 16, f"expected (nr+1)*nt = 80 welded nodes, got {n_nodes}"
