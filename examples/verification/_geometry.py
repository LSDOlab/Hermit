"""Analytic mesh builders for the verification suite -- built in memory, no files.

Every benchmark geometry here is a mapped structured grid, so one helper
(:func:`structured_surface`) covers almost all of them: give it a parametrisation
``fn(u, v) -> (x, y, z)`` over the unit square and a grid size, and it welds the
triangles or quads and hands back a DOLFINx mesh.

Why in memory rather than ``.xdmf`` on disk:

* the ``hermit`` conda env has **no meshio** (``tests/meshes/generate.py`` imports
  it and therefore cannot be run here -- its plate meshes are committed outputs),
* a benchmark's whole point is a *convergence sweep*, so the mesh is a function of
  the refinement level rather than a fixed artefact, and
* it keeps binary files out of the repo.

Node order is the order points are created ("file" order), which is what
``hm.from_nodal`` / ``hm.from_cells`` and ``domain.node_coords`` expect.
"""

from __future__ import annotations

import basix.ufl
import numpy as np
import ufl
from dolfinx.mesh import create_mesh
from mpi4py import MPI

__all__ = [
    "structured_surface",
    "rect_plate",
    "cylinder_sector",
    "sphere_octant",
    "disk",
    "annulus",
    "twisted_strip",
    "hyperbolic_paraboloid",
    "curved_beam",
]

_CELL_NODES = {"triangle": 3, "quad": 4}
# basix spells it "quadrilateral"; "quad" is the shorter name used throughout this
# module (and by meshio), so translate on the way into the element.
_BASIX_CELL = {"triangle": "triangle", "quad": "quadrilateral"}


def _build(points, cells, cell_type):
    """``dolfinx.mesh.Mesh`` from raw points/cells, across DOLFINx 0.9 and 0.11.

    The two versions **swap the last two positional arguments** of ``create_mesh``:
    0.11 is ``(comm, cells, e, x)``, 0.9 is ``(comm, cells, x, domain)``. Getting it
    wrong does not raise anything legible -- 0.11 fails deep inside NumPy with
    "the requested array would exceed the maximum number of dimension of 64" -- so
    dispatch on the signature rather than on a version string.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    cells = np.ascontiguousarray(cells, dtype=np.int64)
    el = basix.ufl.element("Lagrange", _BASIX_CELL[cell_type], 1, shape=(3,))
    domain = ufl.Mesh(el)

    import inspect

    params = list(inspect.signature(create_mesh).parameters)
    if params[2] == "e":                       # DOLFINx 0.11
        return create_mesh(MPI.COMM_WORLD, cells, domain, points)
    return create_mesh(MPI.COMM_WORLD, cells, points, domain)   # DOLFINx 0.9


def structured_surface(fn, nu, nv, *, cell="triangle", wrap_v=False, diag="right"):
    """Mesh the surface ``fn(u, v) -> (x, y, z)`` over ``u, v in [0, 1]^2``.

    ``fn`` is evaluated on the ``(nu + 1) x (nv + 1)`` grid and must be vectorised
    over arrays. ``wrap_v`` welds ``v = 1`` back onto ``v = 0`` for a closed surface
    of revolution (the seam row is not duplicated). ``diag`` picks the triangle
    split direction; ``"alternating"`` flips it cell by cell, which removes the
    directional bias a uniform split imposes on a coarse mesh.
    """
    if cell not in _CELL_NODES:
        raise ValueError(f"cell must be one of {sorted(_CELL_NODES)}, got {cell!r}")

    nv_pts = nv if wrap_v else nv + 1
    u = np.linspace(0.0, 1.0, nu + 1)
    v = np.linspace(0.0, 1.0, nv + 1)[:nv_pts]
    uu, vv = np.meshgrid(u, v, indexing="xy")
    x, y, z = fn(uu.ravel(), vv.ravel())
    points = np.column_stack([x, y, z])

    def node(i, j):
        return (j % nv_pts) * (nu + 1) + i

    cells = []
    for j in range(nv):
        for i in range(nu):
            a, b = node(i, j), node(i + 1, j)
            d, c = node(i, j + 1), node(i + 1, j + 1)
            if cell == "quad":
                # basix orders quadrilateral vertices by tensor product --
                # (0,0), (1,0), (0,1), (1,1) -- NOT cyclic around the cell. The
                # cyclic order that meshio/XDMF uses (and that
                # tests/meshes/generate.py writes) is permuted by the XDMF reader,
                # but create_mesh takes the basix order as given: hand it a cyclic
                # quad and you get silently tangled bowtie cells, a solve that runs
                # happily, and answers off by seven orders of magnitude.
                cells.append([a, b, d, c])
            elif diag == "alternating" and (i + j) % 2:
                cells += [[a, b, d], [b, c, d]]
            else:
                cells += [[a, b, c], [a, c, d]]
    return _build(points, cells, cell)


def rect_plate(length=10.0, width=2.0, nx=20, ny=4, *, cell="quad", z=0.0):
    """Flat rectangular plate in the ``z = const`` plane, ``x in [0, L]``, ``y in [0, W]``."""
    return structured_surface(
        lambda u, v: (length * u, width * v, np.full_like(u, z)),
        nx, ny, cell=cell)


def cylinder_sector(radius=25.0, length=50.0, half_angle=40.0, nx=16, nt=16,
                    *, cell="triangle", theta0=None, theta1=None):
    """Cylindrical shell panel, axis along ``x``.

    The section runs over ``theta`` measured from the ``+z`` axis toward ``+y``:
    ``y = R sin(theta)``, ``z = R cos(theta)``. By default it is symmetric,
    ``theta in [-half_angle, +half_angle]`` degrees (the Scordelis-Lo roof); pass
    ``theta0`` / ``theta1`` in degrees for an explicit range (the pinched-cylinder
    octant uses ``0..90``).
    """
    t0 = np.radians(-half_angle if theta0 is None else theta0)
    t1 = np.radians(half_angle if theta1 is None else theta1)

    def fn(u, v):
        t = t0 + (t1 - t0) * v
        return length * u, radius * np.sin(t), radius * np.cos(t)

    return structured_surface(fn, nx, nt, cell=cell)


def sphere_octant(radius=10.0, hole_angle=18.0, n=16, *, cell="triangle"):
    """Octant of a spherical shell with a polar hole -- the pinched-hemisphere benchmark.

    ``phi`` (polar angle from ``+z``) runs from ``hole_angle`` degrees to 90 degrees
    (the equator); ``theta`` (azimuth) runs 0 to 90 degrees. The free edge at the hole
    and the free equator are both traction free in that benchmark.
    """
    p0, p1 = np.radians(hole_angle), np.pi / 2

    def fn(u, v):
        phi = p0 + (p1 - p0) * u
        th = (np.pi / 2) * v
        return (radius * np.sin(phi) * np.cos(th),
                radius * np.sin(phi) * np.sin(th),
                radius * np.cos(phi))

    return structured_surface(fn, n, n, cell=cell)


def disk(radius=1.0, nr=8, nt=32, *, z=0.0):
    """Flat circular plate, polar triangle mesh with a single centre node.

    A mapped structured grid degenerates at ``r = 0`` (a whole row of coincident
    points), so the centre is emitted once and the innermost ring is a triangle fan
    around it. Always triangles.
    """
    pts = [[0.0, 0.0, z]]
    for i in range(1, nr + 1):
        r = radius * i / nr
        th = np.linspace(0.0, 2 * np.pi, nt, endpoint=False)
        pts += [[r * np.cos(t), r * np.sin(t), z] for t in th]
    cells = [[0, 1 + j, 1 + (j + 1) % nt] for j in range(nt)]      # fan at the centre
    for i in range(nr - 1):
        inner, outer = 1 + i * nt, 1 + (i + 1) * nt
        for j in range(nt):
            jn = (j + 1) % nt
            cells += [[inner + j, outer + j, outer + jn],
                      [inner + j, outer + jn, inner + jn]]
    return _build(np.array(pts), cells, "triangle")


def annulus(r_inner=1.0, r_outer=2.0, nr=8, nt=32, *, cell="quad", z=0.0):
    """Flat annular plate, closed in the circumferential direction."""
    def fn(u, v):
        r = r_inner + (r_outer - r_inner) * u
        th = 2 * np.pi * v
        return r * np.cos(th), r * np.sin(th), np.full_like(u, z)

    return structured_surface(fn, nr, nt, cell=cell, wrap_v=True)


def twisted_strip(length=12.0, width=1.1, twist=90.0, nx=24, ny=2, *, cell="quad"):
    """MacNeal-Harder twisted beam: a strip twisted ``twist`` degrees root to tip.

    The beam axis is ``x``; the cross-section rotates linearly about it, so the
    section at ``x`` is spanned by ``(0, cos a, sin a)`` with ``a = twist * x / L``.
    """
    def fn(u, v):
        a = np.radians(twist) * u
        s = width * (v - 0.5)
        return length * u, s * np.cos(a), s * np.sin(a)

    return structured_surface(fn, nx, ny, cell=cell)


def hyperbolic_paraboloid(length=4.0, width=2.0, warp=0.25, nx=16, ny=8,
                          *, cell="quad"):
    """Rectangular hypar ``z = warp * x * (y - width / 2)``.

    ``warp`` is the constant mixed-curvature rate (and has inverse-length units).
    The bilinear ``x*y`` term leaves every structured quadrilateral non-planar;
    although the corner-to-plane distance scales with cell area, its fixed mixed
    warp rate is retained by every refinement level.
    """
    def fn(u, v):
        x, y = length * u, width * v
        return x, y, warp * x * (y - width / 2)

    return structured_surface(fn, nx, ny, cell=cell)


def curved_beam(r_inner=4.12, r_outer=4.32, sweep=90.0, nr=1, ns=6, *, cell="quad"):
    """MacNeal-Harder curved beam: a 90-degree ring segment lying in the ``z = 0`` plane."""
    def fn(u, v):
        r = r_inner + (r_outer - r_inner) * v
        th = np.radians(sweep) * u
        return r * np.cos(th), r * np.sin(th), np.zeros_like(u)

    return structured_surface(fn, ns, nr, cell=cell)
