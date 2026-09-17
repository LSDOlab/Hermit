"""``hermit.material`` (``Material``/``Orientation``/``isotropic``/``laminate``/
``composite``/``thickness_only``) against stored reference results (``legacy_ref``),
plus the ``failure_index`` orientation handling (``hermit.failure`` -- see that
module's docstring).

Drives ``ShellSolveOp`` / ``ShellScalarFormsOp`` directly (like ``test_pde_raw.py`` /
``test_orientation_form.py``) rather than through the public ``hm.solve``, which
``test_solve_api.py`` covers end to end. Every gate here picks a
``constitutive_space`` that matches a real ``ShellPDE``'s fixed ``VABD``/``VAs``/``VT``
space, so the new ``Material``'s ``Field``s can be fed straight into the existing,
already-adjoint-tested ops.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit
import hermit.bcs as hbc
import hermit.material as hm
from hermit.domain import ShellDomain
from hermit.fenics.ops import ShellFieldFormsOp, ShellScalarFormsOp, ShellSolveOp, strain_fields
from hermit.fenics.shell_pde import ShellPDE
from hermit._laminate import Layup
from conftest import clamped_at_x0


# -- shared helpers ------------------------------------------------------------

def _raw(dom, elementwise=True):
    """The raw FE operands these gates drive directly: a ``ShellPDE`` on ``dom.W``
    with the elementwise fixed material space these gates' ``("DG", 0)``
    ``constitutive_space`` matches, plus a penalty-clamp ``BCData`` at
    ``clamped_at_x0``."""
    pde = ShellPDE(dom.mesh, element=dom.element, W=dom.W, element_wise_material=elementwise)
    pde._solve_form_cache = {}
    return pde, hbc.clamp(dom, where=clamped_at_x0).to_bc_data()


def _pressure_vec(pde, pz):
    n = pde.VF.dofmap.index_map.size_local
    arr = np.zeros((n, 3)); arr[:, 2] = pz
    return csdl.Variable(value=arr.reshape(-1))


_ORIENT_ARG_NAME = {"angle": "fiber_angle", "direction": "fiber_direction"}


def _solve_new_material(pde, bc, dom, material, pz):
    """Drive ``ShellSolveOp`` / ``ShellScalarFormsOp`` straight off a new-surface
    ``Material``'s ``Field``s (FE dof order already) -- returns
    ``(compliance, mass, disp_solid)``."""
    f = _pressure_vec(pde, pz)
    m_ = csdl.Variable(value=np.zeros(f.shape[0]))
    mesh_nodes = csdl.Variable(value=dom.node_coords)

    arg_names = ["A", "B", "D", "As", "thickness", "f", "m"]
    fe_kwargs = dict(A=material.A.coeffs, B=material.B.coeffs, D=material.D.coeffs,
                     As=material.As.coeffs, thickness=material.thickness.coeffs,
                     f=f, m=m_, mesh_nodes=mesh_nodes)
    orient_arg = None
    if material.orientation is not None:
        oname = _ORIENT_ARG_NAME[material.orientation.kind]
        arg_names.append(oname)
        orient_arg = (oname, material.orientation.value.space)
        fe_kwargs[oname] = material.orientation.value.coeffs

    op = ShellSolveOp(pde, bc, tuple(arg_names), None, form_cache=pde._solve_form_cache,
                      orientation=orient_arg)
    disp = op.evaluate(SimpleNamespace(**fe_kwargs))

    sop = ShellScalarFormsOp(pde, {
        "compliance": (pde.compliance_form(), ("disp_solid", "f", "m")),
        "mass": (pde.mass_form(), ("thickness", "density")),
    })
    fe2 = dict(disp_solid=disp, f=f, m=m_, mesh_nodes=mesh_nodes,
              thickness=material.thickness.coeffs, density=material.density.coeffs)
    out = sop.evaluate(SimpleNamespace(**fe2))
    return out.compliance, out.mass, disp


def _ud():
    from caddee_materials import TransverseMaterial

    m = TransverseMaterial(name="ud", EA=138e9, ET=10e9, vA=0.34, vT=0.4, GA=7e9, density=1.6e3)
    m.set_strength(F1t=1500e6, F1c=1200e6, F2t=50e6, F2c=200e6, F12=70e6, F23=40e6)
    return m


def _layup(angles_deg, total_h=0.02):
    n = len(angles_deg)
    a = csdl.Variable(value=np.radians(np.asarray(angles_deg, dtype=float)), name="ply_angles")
    return Layup(_ud(), a, np.full(n, total_h / n), num_plies=n)


# -- gate: hm.isotropic vs legacy isotropic_material ----------------------------

def test_isotropic_matches_legacy_compliance_and_mass(plate_mesh, cantilever_ref, legacy_ref, recorder):
    ref = cantilever_ref
    E, nu, rho, h, pz = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val", "h_val", "pressure_z"))

    c_legacy = float(legacy_ref["material_isotropic_dg0__compliance"])
    m_legacy = float(legacy_ref["material_isotropic_dg0__mass"])

    nel = plate_mesh.topology.index_map(2).size_local
    dom = ShellDomain(plate_mesh, element="CG2CG1")
    pde, bc = _raw(dom)
    mat = hm.isotropic(dom, E=E * np.ones(nel), nu=nu * np.ones(nel),
                       thickness=h * np.ones(nel), density=rho * np.ones(nel),
                       constitutive_space=("DG", 0))
    c_new, m_new, _ = _solve_new_material(pde, bc, dom, mat, pz)
    c_new_v, m_new_v = float(np.ravel(c_new.value)[0]), float(np.ravel(m_new.value)[0])

    print(f"\ncompliance  legacy={c_legacy:.12e}  new={c_new_v:.12e}  "
         f"rel={abs(c_new_v - c_legacy) / c_legacy:.2e}")
    print(f"mass        legacy={m_legacy:.12e}  new={m_new_v:.12e}  "
         f"rel={abs(m_new_v - m_legacy) / m_legacy:.2e}")
    assert c_new_v == pytest.approx(c_legacy, rel=1e-8)
    assert m_new_v == pytest.approx(m_legacy, rel=1e-12)


def test_isotropic_derivative_wrt_thickness_matches_legacy(plate_mesh, cantilever_ref, legacy_ref):
    ref = cantilever_ref
    E, nu, rho, h, pz = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val", "h_val", "pressure_z"))
    nel = plate_mesh.topology.index_map(2).size_local

    def run_new(scale):
        rec = csdl.Recorder(inline=True); rec.start()
        dom = ShellDomain(plate_mesh, element="CG2CG1")
        pde, bc = _raw(dom)
        s = csdl.Variable(value=float(scale), name="t_scale")
        t = s * csdl.Variable(value=h * np.ones(nel))
        mat = hm.isotropic(dom, E=E * np.ones(nel), nu=nu * np.ones(nel), thickness=t,
                           density=rho * np.ones(nel), constitutive_space=("DG", 0))
        c, _, _ = _solve_new_material(pde, bc, dom, mat, pz)
        g = csdl.experimental.PySimulator(rec).compute_totals([c], [s])[c, s]
        val, grad = float(np.ravel(c.value)[0]), float(np.ravel(g)[0])
        rec.stop()
        return val, grad

    v_legacy = float(legacy_ref["material_isotropic_dg0__compliance"])
    g_legacy = float(legacy_ref["material_isotropic_dg0__dcompliance_dtscale"])
    v_new, g_new = run_new(1.0)
    print(f"\nd(compliance)/d(t_scale)  legacy={g_legacy:.10e}  new={g_new:.10e}  "
         f"rel={abs(g_new - g_legacy) / abs(g_legacy):.2e}")
    assert v_new == pytest.approx(v_legacy, rel=1e-8)
    assert g_new == pytest.approx(g_legacy, rel=1e-6)


# -- gate: hm.laminate vs legacy laminate_material -------------------------------
#
# These compare a *laminate* compliance against the stored 0.1 fixture at rel=1e-6,
# looser than the isotropic gates above (1e-8), and deliberately so. The layups here
# are unsymmetric ([10, -25, 55], [10, 80, 35]), so compute_clt's B matrix is a
# difference of ply contributions -- catastrophic cancellation, which amplifies
# machine-to-machine float divergence well beyond the isotropic path's. Measured on
# the same DOLFINx 0.11, different machine: rel = 5.25e-8 (1.3537251164742212 vs the
# stored 1.3537251875494756), where the isotropic case stays under 1e-8. 1e-6 keeps
# ~20x margin over that and still sits four orders under any real discrepancy -- a
# wrong ABD, space or ordering moves these at percent level. Mass stays at 1e-12: it
# is a quadrature of the material field, with no solve and no CLT in it.

def test_laminate_matches_legacy_compliance_and_mass(plate_mesh, cantilever_ref, legacy_ref, recorder):
    ref = cantilever_ref
    pz, rho = float(ref["pressure_z"]), 1.6e3

    c_legacy = float(legacy_ref["material_laminate_10_-25_55__compliance"])
    m_legacy = float(legacy_ref["material_laminate_10_-25_55__mass"])

    dom = ShellDomain(plate_mesh, element="CG2CG1")
    pde, bc = _raw(dom)
    nel = plate_mesh.topology.index_map(2).size_local
    mat = hm.laminate(dom, layup=_layup([10.0, -25.0, 55.0]), density=rho * np.ones(nel),
                      constitutive_space=("DG", 0))
    c_new, m_new, _ = _solve_new_material(pde, bc, dom, mat, pz)
    c_new_v, m_new_v = float(np.ravel(c_new.value)[0]), float(np.ravel(m_new.value)[0])

    print(f"\ncompliance  legacy={c_legacy:.12e}  new={c_new_v:.12e}  "
         f"rel={abs(c_new_v - c_legacy) / c_legacy:.2e}")
    print(f"mass        legacy={m_legacy:.12e}  new={m_new_v:.12e}  "
         f"rel={abs(m_new_v - m_legacy) / m_legacy:.2e}")
    assert c_new_v == pytest.approx(c_legacy, rel=1e-6)
    assert m_new_v == pytest.approx(m_legacy, rel=1e-12)


def test_laminate_derivative_wrt_ply_angles_matches_legacy(plate_mesh, legacy_ref):
    pz, rho, ang0 = 1.0e3, 1.6e3, [10.0, 80.0, 35.0]
    nel = plate_mesh.topology.index_map(2).size_local

    def run_new():
        rec = csdl.Recorder(inline=True); rec.start()
        dom = ShellDomain(plate_mesh, element="CG2CG1")
        pde, bc = _raw(dom)
        layup = _layup(ang0)
        mat = hm.laminate(dom, layup=layup, density=rho * np.ones(nel), constitutive_space=("DG", 0))
        c, _, _ = _solve_new_material(pde, bc, dom, mat, pz)
        g = csdl.experimental.PySimulator(rec).compute_totals([c], [layup.angles])[c, layup.angles]
        val, grad = float(np.ravel(c.value)[0]), np.asarray(g).ravel().copy()
        rec.stop()
        return val, grad

    v_legacy = float(legacy_ref["material_laminate_10_80_35__compliance"])
    g_legacy = legacy_ref["material_laminate_10_80_35__dcompliance_dangles"]
    v_new, g_new = run_new()
    rel_g = np.linalg.norm(g_new - g_legacy) / np.linalg.norm(g_legacy)
    print(f"\ncompliance  legacy={v_legacy:.10e}  new={v_new:.10e}")
    print(f"d(compliance)/d(angles)  legacy={g_legacy}  new={g_new}  rel(||.||)={rel_g:.2e}")
    assert v_new == pytest.approx(v_legacy, rel=1e-6)
    assert rel_g < 1e-6


# -- other Material constructors -------------------------------------------------

def test_isotropic_ignores_orientation(plate_mesh, recorder):
    dom = hermit.ShellDomain(plate_mesh)
    orient = hm.fiber_direction(dom, [0.0, 1.0, 0.0])
    mat = hm.isotropic(dom, E=70e9, nu=0.3, thickness=0.01, density=2700.0, orientation=orient)
    assert mat.orientation is None


def test_default_constitutive_space_is_dg_max_degree(plate_mesh, recorder):
    dom = hermit.ShellDomain(plate_mesh)
    nn = dom.n_nodes
    E_cg1 = hermit.from_nodal(dom, 70e9 * np.ones(nn))         # ("Lagrange", 1)
    nu_dg0 = hermit.from_cells(dom, 0.3 * np.ones(dom.n_cells))  # ("DG", 0)
    mat = hm.isotropic(dom, E=E_cg1, nu=nu_dg0, thickness=0.01, density=2700.0)
    assert mat.A.space[:2] == ("DG", 1)


def test_thickness_only_leaves_abd_none(plate_mesh, recorder):
    dom = hermit.ShellDomain(plate_mesh)
    mat = hm.thickness_only(dom, thickness=0.01, density=2700.0)
    assert mat.A is None and mat.B is None and mat.D is None and mat.As is None
    assert mat.thickness is not None and mat.density is not None


def test_composite_takes_fields_directly(plate_mesh, recorder):
    dom = hermit.ShellDomain(plate_mesh)
    n = dom.n_cells
    A = np.tile(np.eye(3), (n, 1, 1))
    B = np.zeros((n, 3, 3))
    D = np.tile(np.eye(3), (n, 1, 1))
    As = np.tile(np.eye(2), (n, 1, 1))
    mat = hm.composite(dom, A=A, B=B, D=D, As=As, thickness=0.01, density=2700.0)
    assert mat.A.space[:2] == ("DG", 0)
    assert np.allclose(mat.A.values[0].reshape(3, 3), np.eye(3))


# -- gate: failure_index respects orientation (the defect fix) ------------------

def test_failure_index_with_fiber_angle_matches_fiber_direction(plate_mesh):
    """A scalar ``fiber_angle`` must reach ``failure_index`` at all, and agree with the
    equivalent ``fiber_direction``.

    On this flat plate ``local_frames()`` is exactly the identity (e0=x, e1=y), so
    ``fiber_angle(pi/2)`` and ``fiber_direction([0, 1, 0])`` are the same laminate.

    Regression gate: ``_theta_fe_order``'s angle branch returned ``cell_values()``
    unflattened -- ``(n_cells, 1)`` against ``(n_cells,)`` strains -- so this raised
    "Shapes (n, 1) and (n,) not compatible" instead of computing anything. Nothing
    covered it: the orientation gate above uses ``fiber_direction`` (a different,
    numpy branch) and ``None``.
    """
    import hermit.failure as hf

    pz = 1.0e3

    def run(orientation_builder):
        rec = csdl.Recorder(inline=True); rec.start()
        dom = ShellDomain(plate_mesh, element="CG2CG1")
        pde, bc = _raw(dom)
        nel = plate_mesh.topology.index_map(2).size_local
        specs = strain_fields(pde, space=("DG", 0), method="average")
        orientation = orientation_builder(dom)
        mat = hm.laminate(dom, layup=_layup([0.0, 45.0, 0.0]), density=1.6e3 * np.ones(nel),
                          orientation=orientation, constitutive_space=("DG", 0))
        c, _, disp = _solve_new_material(pde, bc, dom, mat, pz)
        if orientation is not None and orientation.kind == "direction":
            from hermit.fenics.ops import orientation_cos_sin_spec
            specs["orientation_cs"] = orientation_cos_sin_spec(pde, "fiber_direction", coeff_space=orientation.value.space, target_space=("DG", 0), method="average")
        strain_op = ShellFieldFormsOp(pde, specs, coefficients={"fiber_direction": orientation.value.space} if orientation is not None and orientation.kind == "direction" else None)
        sv = strain_op.evaluate(SimpleNamespace(disp_solid=disp, mesh_nodes=csdl.Variable(value=dom.node_coords), fiber_direction=orientation.value.coeffs if orientation is not None and orientation.kind == "direction" else None))
        fi = hf.failure_index(sv.mid_strain, sv.curvature, sv.shear_strain, mat, orientation_cs=getattr(sv, "orientation_cs", None), rho=100.0)
        out = float(np.ravel(c.value)[0]), float(np.ravel(fi.value)[0])
        rec.stop()
        return out

    c_ang, fi_ang = run(lambda dom: hm.fiber_angle(dom, np.pi / 2))
    c_dir, fi_dir = run(lambda dom: hm.fiber_direction(dom, [0.0, 1.0, 0.0]))
    # rel 1e-6, not tighter: the two forms are not bit-identical even at the pinned
    # _ORIENTED_QUADRATURE_DEGREE, because fiber_direction's cos/sin come from a
    # tangent-plane projection (a sqrt and a division) while fiber_angle's are cos/sin
    # of a coefficient -- see the comment above _ORIENTED_QUADRATURE_DEGREE in
    # hermit/fenics/elastic_model.py. Measured here: 2.1e-8.
    assert c_ang == pytest.approx(c_dir, rel=1e-6)
    assert fi_ang == pytest.approx(fi_dir, rel=1e-6)


def test_failure_index_respects_orientation(plate_mesh):
    """[0/0/0] with fibres along y and [90/90/90] with fibres along e0 are the same
    physical laminate -- compliance agrees exactly, and with hermit.failure's Tε(θ)
    rotation failure_index must agree too (without it: 44.38 vs 60.88, 27% apart --
    see hermit/failure.py's module docstring)."""
    import hermit.failure as hf

    pz = 1.0e3

    def run(angles_deg, orientation_builder):
        rec = csdl.Recorder(inline=True); rec.start()
        dom = ShellDomain(plate_mesh, element="CG2CG1")
        pde, bc = _raw(dom)
        nel = plate_mesh.topology.index_map(2).size_local
        layup = _layup(angles_deg)
        orientation = orientation_builder(dom) if orientation_builder is not None else None
        mat = hm.laminate(dom, layup=layup, density=1.6e3 * np.ones(nel), orientation=orientation,
                          constitutive_space=("DG", 0))
        c, _, disp = _solve_new_material(pde, bc, dom, mat, pz)
        mesh_nodes = csdl.Variable(value=dom.node_coords)
        specs = strain_fields(pde, space=("DG", 0), method="average")
        if orientation is not None and orientation.kind == "direction":
            from hermit.fenics.ops import orientation_cos_sin_spec
            specs["orientation_cs"] = orientation_cos_sin_spec(pde, "fiber_direction", coeff_space=orientation.value.space, target_space=("DG", 0), method="average")
        strain_op = ShellFieldFormsOp(pde, specs, coefficients={"fiber_direction": orientation.value.space} if orientation is not None and orientation.kind == "direction" else None)
        sv = strain_op.evaluate(SimpleNamespace(disp_solid=disp, mesh_nodes=mesh_nodes, fiber_direction=orientation.value.coeffs if orientation is not None and orientation.kind == "direction" else None))
        fi = hf.failure_index(sv.mid_strain, sv.curvature, sv.shear_strain, mat, orientation_cs=getattr(sv, "orientation_cs", None), rho=100.0)
        c_v, fi_v = float(np.ravel(c.value)[0]), float(np.ravel(fi.value)[0])
        rec.stop()
        return c_v, fi_v

    c_y, fi_y = run([0.0, 0.0, 0.0], lambda dom: hm.fiber_direction(dom, [0.0, 1.0, 0.0]))
    c_90, fi_90 = run([90.0, 90.0, 90.0], None)

    print(f"\n[0/0/0] fibre=y   compliance={c_y:.6e}  failure_index={fi_y:.6e}")
    print(f"[90/90/90]        compliance={c_90:.6e}  failure_index={fi_90:.6e}")
    # 1e-5, not 1e-6: two spellings of the same orientation now differ by 1.07e-6 of
    # round-off under the constant-normal curvature (#7). The defect this gate exists
    # to catch -- failure_index ignoring orientation -- was a 27% error.
    assert c_y == pytest.approx(c_90, rel=1e-5)
    assert fi_y == pytest.approx(fi_90, rel=1e-5)
