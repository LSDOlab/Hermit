"""State-based postprocess outputs."""

import warnings

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit.bcs as hbc
import hermit.loads as hld
import hermit.material as hmat
import hermit.outputs as out
from hermit.domain import ShellDomain
from hermit._solve import solve
from conftest import clamped_at_x0


def _state(mesh, ref, *, pressure=None):
    E, nu, rho, h, pz = (float(ref[k]) for k in
                         ("E_val", "nu_val", "rho_val", "h_val", "pressure_z"))
    domain = ShellDomain(mesh, element="CG2CG1")
    material = hmat.isotropic(
        domain, E=E * np.ones(domain.n_nodes), nu=nu * np.ones(domain.n_nodes),
        thickness=h * np.ones(domain.n_nodes), density=rho * np.ones(domain.n_nodes),
        constitutive_space=("Lagrange", 1),
    )
    return solve(domain, material, hld.pressure(domain, pz if pressure is None else pressure),
                 hbc.clamp(domain, where=clamped_at_x0))


def _max(field):
    return float(np.abs(field.coeffs.value).max())


def _laminate_state(mesh, ref, *, pressure=2.0):
    from caddee_materials import TransverseMaterial
    from hermit._laminate import Layup

    ply = TransverseMaterial(name="ud", EA=138e9, ET=10e9, vA=0.34, vT=0.4,
                             GA=7e9, density=1.6e3)
    ply.set_strength(F1t=1500e6, F1c=1200e6, F2t=50e6, F2c=200e6,
                     F12=70e6, F23=40e6)
    angles = csdl.Variable(value=np.radians([0.0, 90.0, 0.0]))
    layup = Layup(ply, angles, np.full(3, 0.2 / 3), num_plies=3)
    domain = ShellDomain(mesh, element="CG2CG1")
    material = hmat.laminate(domain, layup=layup, density=1.6e3 * np.ones(domain.n_nodes))
    return solve(domain, material, hld.pressure(domain, pressure),
                 hbc.clamp(domain, where=clamped_at_x0))


def test_scalar_outputs_match_reference(plate_mesh, cantilever_ref, recorder):
    state = _state(plate_mesh, cantilever_ref)
    values = {
        "compliance": out.compliance(state),
        "mass": out.mass(state),
        "elastic_energy": out.elastic_energy(state),
    }
    # ``aggregated_stress`` is deliberately NOT gated against the fixture any more.
    # ``rmshell_cantilever.npz``'s value (958117.0189439328) is femo's softabs floor
    # ln(2)/50 dressed up as a stress -- a constant that a matching assertion would
    # accept for *any* displacement field. Removing the softabs makes it unreproducible
    # as well as meaningless; the key stays in the npz as a record of what femo
    # computed. Real gates: the three ``test_aggregated_stress_*`` tests below.
    for name, value in values.items():
        assert float(value.value[0]) == pytest.approx(float(cantilever_ref[name]), rel={
            "compliance": 1e-8, "mass": 1e-10, "elastic_energy": 1e-7,
        }[name])
    assert np.allclose(out.center_of_gravity(state).value, cantilever_ref["cg"], rtol=1e-7)


def test_nodal_outputs_match_reference(plate_mesh, cantilever_ref, recorder):
    state = _state(plate_mesh, cantilever_ref)
    assert np.allclose(out.nodal_displacement(state).value, cantilever_ref["disp_extracted"], rtol=1e-6)
    assert np.allclose(out.nodal_rotation(state).value, cantilever_ref["rotations"], rtol=1e-6)


def test_stress_field_methods_scaling_and_bending_sanity(plate_mesh, cantilever_ref, recorder):
    state = _state(plate_mesh, cantilever_ref)
    projected = out.stress_field(state, method="project")
    interpolated = out.stress_field(state, method="interpolate")
    s_projected, s_interpolated = _max(projected), _max(interpolated)
    assert projected.space == ("DG", 2, ())
    assert projected.block_size == 1
    assert s_projected == pytest.approx(s_interpolated, rel=2e-2)

    p = float(cantilever_ref["pressure_z"])
    scaled = _max(out.stress_field(_state(plate_mesh, cantilever_ref, pressure=10 * p)))
    ratio = scaled / s_projected
    print(f"\nstress project={s_projected:.6e} interpolate={s_interpolated:.6e}  10p/p={ratio:.12g}")
    assert ratio == pytest.approx(10.0, rel=1e-9)

    h = float(cantilever_ref["h_val"])
    expected = 3.0 * p * 10.0 ** 2 / h ** 2
    hand_ratio = s_projected / expected
    print(f"stress hand={expected:.6e}  FE/hand={hand_ratio:.6g}")
    assert 0.5 < hand_ratio < 2.0


def test_aggregated_stress_brackets_field_max_when_scaled(plate_mesh, cantilever_ref, recorder):
    """With ``m ~ 1/max(vm)`` the aggregate tracks the stress field -- from *below*.

    ``pnorm_stress`` is the area-*averaged* integral p-norm ``(1/A) int (m vm)^rho dx``,
    so ``(1/m) pnorm^(1/rho) <= max(vm)`` exactly: the mean of a rho-th power cannot
    exceed the pointwise maximum. It approaches the max from below as rho grows -- the
    mirror image of the discrete log-sum-exp bound ``max <= KS <= max + ln(n)/rho`` that
    ``failure_index`` satisfies from above, so "within the KS bound" here means
    ``max*(fraction)^(1/rho) <= agg <= max``.

    The lower end is set by how much area sits near the peak: if vm held its maximum
    over just one 0.5x0.5 cell of the 2x10 plate, ``agg >= max*(0.25/20)^(1/100) =
    0.9571*max``. vm is not constant over that cell so this is an estimate, not a
    theorem -- but it is the right size. Measured agg/max: 0.86641245 (rho=20),
    0.94194261 (rho=50), 0.97048721 (rho=100), 0.98513296 (rho=200).
    """
    state = _state(plate_mesh, cantilever_ref)
    peak = _max(out.stress_field(state))
    m = out.stress_scaling(state)
    assert m == pytest.approx(1.0 / peak, rel=1e-12)

    ratios = {}
    for rho in (20, 50, 100, 200):
        agg = float(np.ravel(out.aggregated_stress(state, rho=rho, m=m).value)[0])
        ratios[rho] = agg / peak
    print(f"\naggregate/max(stress_field): " +
          "  ".join(f"rho={r}: {v:.8f}" for r, v in ratios.items()))
    assert all(v <= 1.0 + 1e-12 for v in ratios.values())     # exact upper bound
    assert ratios[100] > 0.9571                                # peak-cell estimate
    assert ratios[20] < ratios[50] < ratios[100] < ratios[200]  # tightens with rho


def test_aggregated_stress_scales_with_load(plate_mesh, cantilever_ref, recorder):
    """10x the pressure with ``m`` scaled by 1/10 -> exactly 10x the aggregate.

    This is the gate the retired fixture assertion could not be: the old softabs floor
    was bit-identical (958117.0189439327) across a 10x load range.
    """
    p = float(cantilever_ref["pressure_z"])
    base, loud = _state(plate_mesh, cantilever_ref), _state(plate_mesh, cantilever_ref, pressure=10 * p)
    m = out.stress_scaling(base)
    a1 = float(np.ravel(out.aggregated_stress(base, m=m).value)[0])
    a10 = float(np.ravel(out.aggregated_stress(loud, m=m / 10.0).value)[0])
    f1, f10 = _max(out.stress_field(base)), _max(out.stress_field(loud))
    print(f"\naggregate {a1:.10e} -> {a10:.10e}  ratio={a10 / a1:.14f}"
          f"   stress_field ratio={f10 / f1:.14f}")
    assert a10 / a1 == pytest.approx(10.0, rel=1e-9)
    assert a10 / a1 == pytest.approx(f10 / f1, rel=1e-9)


def test_degenerate_scaling_warns_and_is_no_longer_constant(plate_mesh, cantilever_ref, recorder):
    """The femo default ``m=1e-6`` is badly scaled for this fixture and must say so.

    ``(m*vm)**100`` is 1.3e-186 there -- past half of float64's exponent range, one
    modest load change from flushing to zero. With the softabs gone the *value* is now
    right anyway (``m`` cancels exactly), which is the sharpest evidence the constant
    is gone: the badly scaled call and the well scaled one agree to 1e-12, and neither
    is 958117.
    """
    state = _state(plate_mesh, cantilever_ref)
    m = out.stress_scaling(state)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        good = float(np.ravel(out.aggregated_stress(state, m=m).value)[0])
    assert [w for w in caught if "degenerate" in str(w.message)] == []

    with pytest.warns(UserWarning, match="degenerate stress-aggregate scaling"):
        bad = float(np.ravel(out.aggregated_stress(state).value)[0])   # femo default m=1e-6
    print(f"\nm={m:.6e}: {good:.10e}   m=1e-6: {bad:.10e}")
    assert bad == pytest.approx(good, rel=1e-12)          # m cancels: not a constant
    assert bad != pytest.approx(958117.0189439328, rel=1e-6)   # the retired softabs floor


def test_strain_fields_frames_and_pure_bending(plate_mesh, cantilever_ref, recorder):
    state = _state(plate_mesh, cantilever_ref)
    membrane, curvature, shear = out.strain_fields(state)
    for field, shape, kind in ((membrane, (3,), "strain2"),
                               (curvature, (3,), "strain2"),
                               (shear, (2,), "shear2")):
        assert field.space == ("DG", 2, shape)
        assert field.kind == kind
        assert field.n_dofs == field.n_scalar_dofs * int(np.prod(shape))
    assert _max(curvature) > 1e-10
    assert _max(membrane) < 1e-8 * _max(curvature)

    global_fields = out.strain_fields(state, space=("CG", 1), frame="global")
    assert [f.space for f in global_fields] == [("Lagrange", 1, (6,)),
                                                  ("Lagrange", 1, (6,)),
                                                  ("Lagrange", 1, (3,))]
    assert all(f.global_frame for f in global_fields)
    local_fields = out.strain_fields(state, space=("DG", 1), frame="local")
    assert [f.space for f in local_fields] == [("DG", 1, (3,)), ("DG", 1, (3,)),
                                                 ("DG", 1, (2,))]
    assert all(not f.global_frame for f in local_fields)
    with pytest.raises(ValueError, match=r"continuous \(CG\)"):
        out.strain_fields(state, space=("CG", 1), frame="local")


def test_displacement_and_rotation_fields_match_vertices_and_scale(plate_mesh, cantilever_ref, recorder):
    state = _state(plate_mesh, cantilever_ref)
    displacement, rotation = out.displacement_field(state), out.rotation_field(state)
    nodal_disp = np.asarray(out.nodal_displacement(state).value)
    nodal_rot = np.asarray(out.nodal_rotation(state).value)

    for field, nodal in ((displacement, nodal_disp), (rotation, nodal_rot)):
        coords = field.domain.dof_coords(field.space)
        distance = np.linalg.norm(coords[:, None, :] - plate_mesh.geometry.x[None, :, :], axis=2)
        vertex_dofs = np.flatnonzero(distance.min(axis=1) < 1e-12)
        local_vertices = distance[vertex_dofs].argmin(axis=1)
        got = np.asarray(field.coeffs.value).reshape(-1, 3)[vertex_dofs]
        want = nodal[field.domain.node_input_idx[local_vertices]]
        assert np.allclose(got, want, rtol=1e-12, atol=1e-12)

    scaled_state = _state(plate_mesh, cantilever_ref,
                          pressure=10 * float(cantilever_ref["pressure_z"]))
    for field, scaled in ((displacement, out.displacement_field(scaled_state)),
                          (rotation, out.rotation_field(scaled_state))):
        ratio = _max(scaled) / _max(field)
        print(f"\n{field.kind} 10p/p={ratio:.12g}")
        assert ratio == pytest.approx(10.0, rel=1e-9)


def test_failure_field_shape_aggregate_and_isotropic_guard(plate_mesh, cantilever_ref, recorder):
    state = _laminate_state(plate_mesh, cantilever_ref)
    field = out.failure_field(state)
    values = np.asarray(field.values)
    assert field.space == ("DG", 0, (6,))
    assert values.shape == (state.domain.n_cells, 6)
    failure = float(out.failure_index(state, rho=100).value[0])
    maximum = float(values.max())
    print(f"\nfailure KS={failure:.6e} max={maximum:.6e}")
    assert maximum <= failure <= maximum + np.log(values.size) / 100.0 + 1e-9

    isotropic = _state(plate_mesh, cantilever_ref)
    with pytest.raises(ValueError, match="composite"):
        out.failure_field(isotropic)


def test_stress_outputs_reject_laminate_without_isotropic_constants(plate_mesh, recorder):
    domain = ShellDomain(plate_mesh)
    material = hmat.thickness_only(domain, thickness=0.01, density=1600.0)
    # A surrogate state is sufficient: the material guard runs before any assembly.
    from hermit._solve import ShellState
    ndof = domain.W.dofmap.index_map.size_local * domain.W.dofmap.index_map_bs
    state = ShellState(domain, None, material, hld.pressure(domain, 1.0),
                       hbc.clamp(domain, where=clamped_at_x0),
                       disp_solid=np.zeros(ndof))
    with pytest.raises(ValueError, match="isotropic-only.*failure_index"):
        out.aggregated_stress(state)
    with pytest.raises(ValueError, match="isotropic-only.*failure_index"):
        out.pnorm_stress(state)
    with pytest.raises(ValueError, match="isotropic-only.*failure_index"):
        out.stress_field(state)
