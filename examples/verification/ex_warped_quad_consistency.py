"""Warped quadrilateral cells preserve rigid-body consistency

**This case records a resolved defect and guards the formulation choice.** Hermit's
constant-normal bending curvature now gives warped quadrilaterals the same effective
flat-facet theory as triangles. Before that change, a doubly-curved surface whose
quadrilateral cells are *warped* -- four corners that do not lie in a plane --
converged cleanly and monotonically to a wrong answer.

The fixture is the MacNeal-Harder twisted beam, whose reference is known
(``5.424e-3``, see ``ex_twisted_beam.py``). A twisted strip is a ruled but
*non-developable* surface, so a structured quad mesh on it is necessarily warped,
while a triangle mesh on the same nodes is not -- every triangle is planar by
construction. Meshing the same geometry both ways isolates the cell shape as the only
difference:

===========  ==================  ==================
refinement   triangles           warped quads
===========  ==================  ==================
12 x 4       5.3879e-3 (0.993)   7.6097e-3 (1.403)
24 x 6       5.4081e-3 (0.997)   7.6150e-3 (1.404)
48 x 8       5.4133e-3 (0.998)   7.6168e-3 (1.404)
===========  ==================  ==================

Triangles converged to the reference. Quads converged to a value **40 % too large**,
and refinement moved it *away* from the reference rather than toward it -- the
signature of a consistency error, not a discretisation error. With the
constant-normal measure, the same sweep is:

===========  ==================  ==================  ==============
refinement   triangles           warped quads        quad / triangle
===========  ==================  ==================  ==============
12 x 4       5.3879e-3 (0.993)   5.4024e-3 (0.996)   1.00270
24 x 6       5.4081e-3 (0.997)   5.4122e-3 (0.998)   1.00075
48 x 8       5.4133e-3 (0.998)   5.4148e-3 (0.998)   1.00026
===========  ==================  ==================  ==============

The quad and triangle sequences now converge together and both approach the
MacNeal-Harder reference.

The quad answer is not merely inaccurate, it is unphysical: untwisted beam theory
gives ``P L^3 / (3 E I)`` with ``I = w t^3 / 12``, or ``6.6125e-3``, for a tip load
perpendicular to the wide face. A 90-degree twist rotates the stiff axis into the load
path and can only *stiffen* that response, so any correct answer must be below
``6.6125e-3``. The published ``5.424e-3`` is; the quad result, ``7.617e-3``, is 15 %
above a bound it cannot legitimately exceed.

Root cause of the former result -- established, not conjectured. The old curvature
measure (``ElasticModel._bending_curvature``) was::

    kappa = sym(gradv_local(grad(cross(E2, theta)), E01))

Under a rigid-body motion the rotation field is a constant vector ``theta = omega``, so
by the product rule ``grad(cross(E2, omega)) = cross(grad(E2), omega)``. On a planar
cell ``E2`` is constant, ``grad(E2) = 0``, and ``kappa`` vanishes identically -- as it
must. On a warped cell ``E2`` varies *within* the cell, so ``kappa`` is **non-zero for
a constant rotation field**: the measure differentiates the shell normal along with the
rotation. The element therefore stores energy under rigid-body motion, which is a
consistency violation, not an accuracy limitation.

Measured directly, as spurious elastic energy under an exact rigid motion
``u = c + omega x x``, ``theta = omega`` (normalised by ``E t^3``):

==========  ===========  ===========
warp        quad         triangle
==========  ===========  ===========
0           8.5e-30      2.2e-29
1e-4        2.7e-9       2.3e-29
1e-3        2.7e-7       2.3e-29
1e-2        2.7e-5       2.4e-29
==========  ===========  ===========

It scales as ``warp**2``, exactly as ``kappa`` proportional to ``grad(E2)`` predicts,
and splitting the energy by term puts **all** of it in the bending term -- membrane,
shear and drilling stay at ~1e-30 for both cell types at every warp. Independently
ruled out: quadrature (forcing degrees 2 to 20 changes nothing) and the drilling
stabilisation (scaling it over four orders of magnitude moves the result 0.2 %).

Hermit now holds ``E2`` constant in this measure and constructs curvature from
``grad(theta)`` alone. A constant rigid rotation therefore gives zero curvature
structurally. Triangles are no longer required merely because geometry is doubly
curved; quads and triangles now implement the same flat-facet theory. The remaining
limitation is physical rather than cell-specific: moderately thick, strongly curved,
membrane-loaded shells may need a full covariant formulation (see the user guide).
The gate below asserts that the warped-quad and triangle answers agree; the old 1.407
ratio fails it by a wide margin.

    conda activate hermit
    python examples/verification/ex_warped_quad_consistency.py
"""

import pathlib
import sys

import numpy as np
import csdl_alpha as csdl

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import twisted_strip                      # noqa: E402
from _harness import Case, main                          # noqa: E402

LENGTH, WIDTH, THICKNESS, TWIST = 12.0, 1.1, 0.32, 90.0
E, NU, LOAD = 29.0e6, 0.22, 1.0
PUBLISHED = 5.424e-3
# Tip load perpendicular to the wide face on the UNTWISTED beam. A 90-degree twist
# can only stiffen this, so any correct twisted answer lies below it.
UNTWISTED_BOUND = LOAD * LENGTH**3 / (3.0 * E * WIDTH * THICKNESS**3 / 12.0)


def _tip_deflection(nx, ny, cell):
    rec = csdl.Recorder(inline=True)
    rec.start()
    domain = hm.ShellDomain(twisted_strip(LENGTH, WIDTH, TWIST, nx=nx, ny=ny, cell=cell),
                            element="CG2CG1")
    material = hm.isotropic(domain, E=E, nu=NU, thickness=THICKNESS, density=1.0)
    bcs = hm.clamp(domain, where=lambda x: np.isclose(x[0], 0.0), method="strong")
    xyz = np.asarray(domain.node_coords)
    tip = np.flatnonzero(np.isclose(xyz[:, 0], LENGTH))
    load = None
    for k in tip:
        term = hm.point_load(domain, at=xyz[k], force=[0.0, 0.0, LOAD / len(tip)])
        load = term if load is None else load + term
    state = hm.solve(domain, material, load, bcs)
    u = hm.nodal_displacement(state).value.reshape(-1, 3)
    rec.stop()
    k = int(np.argmin(np.abs(xyz[:, 0] - LENGTH) + np.abs(xyz[:, 1]) + np.abs(xyz[:, 2])))
    return float(u[k, 2])


def solve_at(n):
    """Ratio of the warped-quad answer to the triangle answer on the same geometry."""
    ny = n // 6 + 2
    tri = _tip_deflection(n, ny, "triangle")
    quad = _tip_deflection(n, ny, "quad")
    print(f"    triangles={tri:.6e} ({tri / PUBLISHED:.4f} of ref)   "
          f"warped quads={quad:.6e} ({quad / PUBLISHED:.4f} of ref)")
    assert quad <= UNTWISTED_BOUND, (
        f"warped-quad answer {quad:.6e} exceeds untwisted bound {UNTWISTED_BOUND:.6e}")
    return quad / tri


CASE = Case(
    name="Warped quadrilateral rigid-body consistency",
    quantity="ratio of the warped-quad tip deflection to the triangle tip deflection",
    reference=1.0,
    tolerance=0.02,
    citation="Measured on this fixture; the underlying twisted-beam reference is "
             "MacNeal & Harder, Finite Elements in Analysis and Design 1(1), 1985",
    levels=(12, 24, 48),
    quick_level=24,
    solve=solve_at,
    monotone=False,
    notes="the pre-fix ratio converged to 1.407; the constant-normal curvature makes "
          "rigid-body objectivity structural and the ratio now converges to 1.",
)

if __name__ == "__main__":
    main(CASE)
