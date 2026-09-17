"""``hermit.bcs`` (``clamp``/``pin``/``symmetry``/``gauge`` + ``__add__``) against a
stored reference.

Every BC kind is driven through the real (unmodified) ``ShellSolveOp`` and compared to
a reference ``disp_solid`` bit-for-bit (or to machine precision). The reference is a
*stored* result (``legacy_ref``, written at commit ``81884ce`` by the since-deleted
``tests/capture_legacy_reference.py``) rather than a live call, so it needs no second
in-process path.

``_run`` drives ``hm.solve`` end to end: it builds a plain ``ShellDomain`` and every
``BoundaryConditions`` is located against that domain's own ``domain.W``, which
``_pde_for`` reuses. That is what makes the two **strong**-BC gates
(``test_strong_clamp_matches_legacy``, ``test_gauge_matches_legacy``) safe: dolfinx
silently drops a ``DirichletBC`` located against a different -- even structurally
identical -- ``FunctionSpace`` instance, and ``hm.solve`` asserts ``bcs.domain is
domain`` so that cannot happen here. (Penalty BCs are UFL measures and were always
immune.)
"""

import numpy as np
import pytest

import hermit
import hermit.loads as hld
import hermit.outputs as hout
from hermit._solve import solve
from hermit.domain import ShellDomain
from hermit.bcs import (
    BoundaryConditions,
    _DOF_NAMES,
    clamp,
    gauge,
    near,
    on_plane,
    pin,
    symmetry,
)
from conftest import assert_matches_legacy, clamped_at_x0


def _run(mesh, nn, E, nu, rho, h, pz, make_bcs):
    """The captured reference case: CG2CG1, nodal isotropic material on the default
    ``("Lagrange", 1)`` constitutive space, and a nodal ``(nn, 3)`` pressure spelled
    as a CG1 ``hm.traction``."""
    dom = ShellDomain(mesh, element="CG2CG1")
    material = hermit.isotropic(
        dom, E=E * np.ones(nn), nu=nu * np.ones(nn), thickness=h * np.ones(nn),
        density=rho * np.ones(nn), constitutive_space=("Lagrange", 1))
    p = np.zeros((nn, 3)); p[:, 2] = pz
    loads = hld.traction(dom, hermit.from_nodal(dom, p), space=("Lagrange", 1))
    return solve(dom, material, loads, make_bcs(dom))


def _mask_key(mask):
    """``legacy_api_reference.npz`` key for a legacy ``bc_dof_mask`` penalty solve."""
    return "bcs_dof_mask_" + "".join(str(b) for b in mask) + "__disp_solid"


@pytest.fixture
def ref_args(cantilever_ref):
    r = cantilever_ref
    return dict(nn=int(r["n_nodes"]), E=float(r["E_val"]), nu=float(r["nu_val"]),
               rho=float(r["rho_val"]), h=float(r["h_val"]), pz=float(r["pressure_z"]))


# -- the gate that matters: penalty clamp -------------------------------------

def test_penalty_clamp_matches_legacy(plate_mesh, ref_args, legacy_ref, recorder):
    legacy_disp = legacy_ref["bcs_penalty_clamp__disp_solid"]
    legacy_c = float(legacy_ref["bcs_penalty_clamp__compliance"])

    def new_bc(dom):
        return clamp(dom, where=clamped_at_x0)

    new = _run(plate_mesh, **ref_args, make_bcs=new_bc)
    assert_matches_legacy(new.disp_solid.value, legacy_disp)
    c = hout.compliance(new)
  # 1e-7, not 1e-9: the constant-normal curvature (#7) is analytically identical
    # on this flat fixture but evaluates a different expression tree, which the
    # beta=1e15 penalty system amplifies to ~7e-9. See tests/conftest.py,
    # assert_matches_legacy, for the full argument.
    assert legacy_c == pytest.approx(float(np.ravel(c.value)[0]), rel=1e-7)


def test_strong_clamp_matches_legacy(plate_mesh, ref_args, legacy_ref, recorder):
    legacy_disp = legacy_ref["bcs_strong_clamp__disp_solid"]

    def new_bc(dom):
        # Strong DirichletBCs must be located against the very FunctionSpace the solve
        # assembles against, or dolfinx drops them silently (no error, just a wrong
        # answer). `dom` is the domain _run solves on, so this locates against the
        # `dom.W` that hm.solve's ShellPDE uses.
        return clamp(dom, where=clamped_at_x0, method="strong")

    new = _run(plate_mesh, **ref_args, make_bcs=new_bc)
    assert_matches_legacy(new.disp_solid.value, legacy_disp)


def test_gauge_matches_legacy(plate_mesh, ref_args, legacy_ref, recorder):
    vtx = plate_mesh.geometry.x[0]
    # legacy: _gauge_bcs(W, isclose(x, vtx, atol=1e-9), dofs 0..5) as the whole BCData
    legacy_disp = legacy_ref["bcs_gauge__disp_solid"]
    assert np.array_equal(vtx, legacy_ref["bcs_gauge__gauge_at"])

    def new_bc(dom):
        # gauge points are strong DirichletBCs -- see test_strong_clamp_matches_legacy
        # on why they must be located against the solve's own `dom.W`.
        return gauge(dom, at=vtx, dofs=_DOF_NAMES)

    new = _run(plate_mesh, **ref_args, make_bcs=new_bc)
    assert_matches_legacy(new.disp_solid.value, legacy_disp)
    # a single gauge point does not remove every rigid-body mode -- both sides should
    # at least agree that the solve was under-constrained the same way (not a hang).
    assert np.isfinite(new.disp_solid.value).all()


# -- pin: named dof -> mask slot, and pin(all six) == clamp -------------------

def test_pin_all_dofs_equals_clamp(plate_mesh, ref_args, recorder):
    def clamp_bc(dom):
        return clamp(dom, where=clamped_at_x0)

    def pin_all_bc(dom):
        return pin(dom, where=clamped_at_x0, dofs=_DOF_NAMES)

    a = _run(plate_mesh, **ref_args, make_bcs=clamp_bc)
    b = _run(plate_mesh, **ref_args, make_bcs=pin_all_bc)
    assert np.array_equal(a.disp_solid.value, b.disp_solid.value)


@pytest.mark.parametrize("dofs,expected_mask", [
    (("uz", "ry"), (0, 0, 1, 0, 1, 0)),
    (("ux",), (1, 0, 0, 0, 0, 0)),
    (("rx", "rz"), (0, 0, 0, 1, 0, 1)),
])
def test_pin_named_dof_matches_legacy_bc_dof_mask(plate_mesh, ref_args, legacy_ref, recorder,
                                                  dofs, expected_mask):
    """Each named dof maps to the expected ``bc_dof_mask`` slot (a masked penalty BC).

    The mask itself is the whole content of this gate, and it is asserted exactly.
    The *solve* is only compared for masks that leave a well-posed problem: pinning a
    single translation (``ux``) or two rotations (``rx``/``rz``) does not remove the
    rigid-body modes, so the residual is near-singular and the solution is dominated
    by an arbitrary null-space component -- max|disp| 1.7e8 and 2.3e7 against 8.7e-3
    for a real clamp. Those are not reproducible in any useful sense: on one machine,
    merely changing the BLAS thread count moves them by 15% to 150%. They only ever
    agreed with the stored reference because it was once recomputed in the same
    process, sharing every rounding decision.
    """
    bcs = pin(ShellDomain(plate_mesh, element="CG2CG1"), where=clamped_at_x0, dofs=dofs)
    assert bcs.penalty_terms[0].mask == expected_mask

    if float(np.abs(legacy_ref[_mask_key(expected_mask)]).max()) > 1.0:
        pytest.skip("legacy reference for this mask is an under-constrained solve "
                    "(rigid-body modes free) -- the mask assertion above is the gate")
    new = _run(plate_mesh, **ref_args, make_bcs=lambda dom: pin(dom, where=clamped_at_x0, dofs=dofs))
    assert_matches_legacy(new.disp_solid.value, legacy_ref[_mask_key(expected_mask)])


# -- symmetry -----------------------------------------------------------------

@pytest.mark.parametrize("normal,expected_mask", [
    ([1, 0, 0], (1, 0, 0, 0, 1, 1)),
    ([0, 0, 1], (0, 0, 1, 1, 1, 0)),
    ([0, 0, -1], (0, 0, 1, 1, 1, 0)),   # sign of the normal doesn't change the mask
])
def test_symmetry_mask_convention(plate_mesh, normal, expected_mask):
    dom = hermit.ShellDomain(plate_mesh)
    bcs = symmetry(dom, where=clamped_at_x0, normal=normal)
    assert bcs.penalty_terms[0].mask == expected_mask


def test_symmetry_matches_equivalent_pin(plate_mesh, recorder):
    """An x-normal symmetry plane is exactly ``pin`` of the three dofs it fixes, and
    both must land on the ``bc_dof_mask`` slots (1, 0, 0, 0, 1, 1).

    Was a solve compared against the stored ``disp_solid`` for that mask. That
    reference is an under-constrained solve (max|disp| 3.4e7 vs 8.7e-3 for a clamp --
    a symmetry plane alone leaves rigid-body modes free), so it is numerically
    arbitrary and moves by tens of percent with the BLAS thread count. The mask
    equality below is the real content and is exact.
    """
    dom = ShellDomain(plate_mesh, element="CG2CG1")
    sym = symmetry(dom, where=clamped_at_x0, normal=[1, 0, 0])
    equivalent = pin(dom, where=clamped_at_x0, dofs=("ux", "ry", "rz"))
    assert sym.penalty_terms[0].mask == (1, 0, 0, 0, 1, 1)
    assert sym.penalty_terms[0].mask == equivalent.penalty_terms[0].mask


def test_symmetry_non_axis_aligned_normal_raises(plate_mesh):
    dom = hermit.ShellDomain(plate_mesh)
    with pytest.raises(ValueError, match="axis-aligned"):
        symmetry(dom, where=clamped_at_x0, normal=[1, 1, 0])


# -- near / on_plane helpers ----------------------------------------------

def test_near_and_on_plane_select_expected_vertices(plate_mesh):
    x = plate_mesh.geometry.x.T
    near_mask = near("x", 0.0)(x)
    plane_mask = on_plane([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])(x)
    legacy_mask = clamped_at_x0(x)
    assert near_mask.sum() > 0
    assert np.array_equal(near_mask, legacy_mask)
    assert np.array_equal(near_mask, plane_mask)


def test_near_accepts_axis_name_or_index(plate_mesh):
    x = plate_mesh.geometry.x.T
    assert np.array_equal(near(0, 0.0)(x), near("x", 0.0)(x))
    assert np.array_equal(near(2, 0.0)(x), near("z", 0.0)(x))


# -- __add__ merge semantics ------------------------------------------------

def test_add_disjoint_regions_keeps_both_terms(plate_mesh):
    dom = hermit.ShellDomain(plate_mesh)
    a = clamp(dom, where=lambda x: np.less(x[0], 1e-12))
    b = pin(dom, where=lambda x: np.greater(x[0], 9.999), dofs=("uz",))
    merged = a + b
    assert len(merged.penalty_terms) == 2
    masks = {t.mask for t in merged.penalty_terms}
    assert masks == {tuple(1 for _ in range(6)), (0, 0, 1, 0, 0, 0)}
    # several differently-masked regions -> BCData.penalty_terms (see
    # test_orientation_form.py for the ElasticModel._penalty_residual summing gate
    # this feeds).
    bcd = merged.to_bc_data()
    assert bcd.dss is None and bcd.dSS is None and bcd.bc_dof_mask is None
    assert len(bcd.penalty_terms) == 2
    assert {m for _, _, m in bcd.penalty_terms} == masks


def test_add_overlapping_regions_later_wins(plate_mesh):
    dom = hermit.ShellDomain(plate_mesh)
    where_a = lambda x: np.less(x[0], 1e-12)
    first = clamp(dom, where=where_a)                    # all 6, region A
    second = pin(dom, where=where_a, dofs=("uz",))        # same region, later -> should win
    merged = first + second
    assert len(merged.penalty_terms) == 1
    assert merged.penalty_terms[0].mask == (0, 0, 1, 0, 0, 0)

    # reversed order: clamp added last should reclaim the region
    merged_rev = second + first
    assert len(merged_rev.penalty_terms) == 1
    assert merged_rev.penalty_terms[0].mask == (1, 1, 1, 1, 1, 1)


def test_add_beta_conflict_raises(plate_mesh):
    dom = hermit.ShellDomain(plate_mesh)
    a = clamp(dom, where=lambda x: np.less(x[0], 1e-12), penalty_beta=1e14)
    b = pin(dom, where=lambda x: np.greater(x[0], 9.999), dofs=("uz",), penalty_beta=1e16)
    with pytest.raises(ValueError, match="penalty_beta mismatch"):
        a + b
    # a strong-only (no penalty term) side never conflicts
    c = gauge(dom, at=plate_mesh.geometry.x[0], dofs=("uz",))
    merged = a + c
    assert merged.penalty_beta == 1e14


def test_add_wrong_domain_raises(plate_mesh, tri_mesh):
    dom1 = hermit.ShellDomain(plate_mesh)
    dom2 = hermit.ShellDomain(tri_mesh)
    a = clamp(dom1, where=lambda x: np.less(x[0], 1e-12))
    b = clamp(dom2, where=lambda x: np.less(x[0], 1e-12))
    with pytest.raises(ValueError, match="same ShellDomain"):
        a + b
