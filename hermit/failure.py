"""``failure_index`` / ``failure_field`` -- Tsai-Wu failure, wired to ``Orientation``.

``hermit._laminate.failure.tsai_wu_field`` documents its input as **laminate**-frame
strains, but the DG0 strain-measure output is in the **element** in-plane frame. So
the *same* ``Tε(θ)`` (engineering-Voigt strain rotation, element -> laminate axes)
the constitutive path already applies (``hermit.fenics.kinematics.strain_rotation`` /
``hermit.csdl_helpers.rotate_abd``) is applied to the strains first, before calling
``tsai_wu_field`` (unmodified -- it already uses the correct ``(xz, yz)``
transverse-shear ordering, see that module's docstring). This is covered by
``tests/test_material_api.py::test_failure_index_respects_orientation`` ("two
spellings of one physical laminate must agree").

This module's entry points take mid-surface strain / curvature / shear directly,
in the **element frame, FE cell order** -- the raw DG0 strain-measure convention
``hermit.fenics.ops.strain_fields`` produces. ``hm.failure_index(state, ...)`` /
``hm.failure_field(state, ...)`` (``hermit.outputs``) are the public entry points;
they assemble those measures off a solved ``ShellState`` and call in here.
"""

from __future__ import annotations

import numpy as np
import csdl_alpha as csdl

from ._laminate import aggregate_failure, tsai_wu_field


def _apply_teps(c, s, v0, v1, v2):
    """``Tε(θ)`` applied component-wise to an engineering-Voigt vector ``[v0, v1,
    v2] = [xx, yy, 2xy]`` -- the same rotation as
    ``hermit.fenics.kinematics.strain_rotation``'s matrix, written out termwise (vectorised over ``(n,)``
    CSDL arrays) rather than assembled as an ``(n, 3, 3)`` matrix."""
    c2, s2, sc = c * c, s * s, s * c
    out0 = c2 * v0 + s2 * v1 + sc * v2
    out1 = s2 * v0 + c2 * v1 - sc * v2
    out2 = -2.0 * sc * v0 + 2.0 * sc * v1 + (c2 - s2) * v2
    return out0, out1, out2


def _apply_shear_rotation(c, s, g0, g1):
    """``R(θ)`` applied to the transverse-shear pair ``[g0, g1] = [xz, yz]`` --
    ``hermit.fenics.kinematics.shear_rotation``'s matrix, written out termwise."""
    return c * g0 + s * g1, -s * g0 + c * g1


def _theta_fe_order(orientation, domain, n_cells):
    """``(n_cells,)`` theta, **FE cell order** -- matching ``mid_strain`` /
    ``curvature`` / ``shear_strain``'s native ordering (``strain_measure_op``'s raw
    output, before any file-order reindexing)."""
    if orientation is None:
        return csdl.Variable(value=np.zeros(n_cells))
    if orientation.kind == "angle":
        theta_file = orientation.value.cell_values()               # csdl, file order
        theta = theta_file[list(domain.cell_input_idx)]             # -> FE order
        if not isinstance(theta, csdl.Variable):
            theta = csdl.Variable(value=np.asarray(theta))
        # cell_values() is (n_cells, bs), and a scalar orientation has bs == 1, so this
        # arrives rank-2 while every strain component downstream is (n_cells,). Without
        # the flatten the first Teps multiply raises "Shapes (n, 1) and (n,) not
        # compatible".
        return csdl.reshape(theta, (n_cells,))
    raise ValueError(
        "a fiber_direction orientation needs its cos/sin passed in as "
        "orientation_cs -- the angle between a global direction and the element "
        "frame depends on the mesh, so evaluating it here in numpy would freeze it "
        "at the reference geometry and drop a term from the shape gradient. "
        "hm.failure_index / hm.failure_field supply it; see "
        "hermit.fenics.ops.orientation_cos_sin_spec.")


def failure_field(mid_strain, curvature, shear_strain, material, orientation_cs=None, *, faces=(-0.5, 0.5)):
    """``(n_cells, 3)`` / ``(n_cells, 3)`` / ``(n_cells, 2)`` **element**-frame
    engineering-Voigt strain / curvature / shear (FE cell order -- e.g.
    ``model.strain_measure_op()``'s raw output) + a composite ``Material`` ->
    ``(n_cells, n_plies * len(faces))`` Tsai-Wu failure indices, rotated into the
    laminate frame by ``material.orientation`` first (``None`` => no rotation)."""
    if material.layup is None:
        raise ValueError("failure_field needs a composite material (material.layup)")
    domain = material.domain
    n = mid_strain.shape[0]

    if orientation_cs is not None:
        c, s = orientation_cs[:, 0], orientation_cs[:, 1]
    else:
        theta = _theta_fe_order(material.orientation, domain, n)
        c, s = csdl.cos(theta), csdl.sin(theta)

    e0, e1, e2 = mid_strain[:, 0], mid_strain[:, 1], mid_strain[:, 2]
    k0, k1, k2 = curvature[:, 0], curvature[:, 1], curvature[:, 2]
    g0, g1 = shear_strain[:, 0], shear_strain[:, 1]
    eL0, eL1, eL2 = _apply_teps(c, s, e0, e1, e2)
    kL0, kL1, kL2 = _apply_teps(c, s, k0, k1, k2)
    gL0, gL1 = _apply_shear_rotation(c, s, g0, g1)

    r = lambda v: csdl.reshape(v, (n, 1))
    eps_lam = csdl.concatenate((r(eL0), r(eL1), r(eL2)), axis=1)
    kappa_lam = csdl.concatenate((r(kL0), r(kL1), r(kL2)), axis=1)
    gamma_lam = csdl.concatenate((r(gL0), r(gL1)), axis=1)
    return tsai_wu_field(eps_lam, kappa_lam, gamma_lam, material.layup, faces=faces)


def failure_index(mid_strain, curvature, shear_strain, material, orientation_cs=None, *, rho=100.0, faces=(-0.5, 0.5)):
    """KS-aggregated scalar (``< 1`` safe) -- see ``failure_field``."""
    field = failure_field(mid_strain, curvature, shear_strain, material,
                          orientation_cs=orientation_cs, faces=faces)
    return aggregate_failure(field, rho=rho)
