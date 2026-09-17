"""Orientation-aware laminate materials: ``fiber_direction`` / ``fiber_angle``."""

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit.material as hmat
import hermit.loads as hld
import hermit.bcs as hbc
import hermit.outputs as out
from hermit.domain import ShellDomain
from hermit._solve import solve
from conftest import clamped_at_x0


def _ud():
    from caddee_materials import TransverseMaterial
    m = TransverseMaterial(name="ud", EA=138e9, ET=10e9, vA=0.34, vT=0.4, GA=7e9, density=1.6e3)
    m.set_strength(F1t=1500e6, F1c=1200e6, F2t=50e6, F2c=200e6, F12=70e6, F23=40e6)
    return m


def _layup(angles_deg, total_h=0.02):
    from hermit._laminate import Layup
    n = len(angles_deg)
    return Layup(_ud(), csdl.Variable(value=np.radians(np.asarray(angles_deg, float)), name="ply_angles"),
                 np.full(n, total_h / n), num_plies=n)


def _compliance(domain, bcs, layup, pz=1e3, **mat_kw):
    material = hmat.laminate(domain, layup=layup, density=1.6e3 * np.ones(domain.n_cells),
                             constitutive_space=("DG", 0), **mat_kw)
    return float(np.ravel(out.compliance(solve(domain, material, hld.pressure(domain, pz), bcs)).value)[0])


def test_fiber_along_e0_is_a_noop(plate_mesh, recorder):
    """Global x ~ the flat plate's element e0, so fibre_direction=x changes nothing."""
    domain = ShellDomain(plate_mesh, element="CG2CG1")
    bcs = hbc.clamp(domain, where=clamped_at_x0)
    c_ref = _compliance(domain, bcs, _layup([0, 0, 0]))
    c_x = _compliance(domain, bcs, _layup([0, 0, 0]),
                      orientation=hmat.fiber_direction(domain, [1.0, 0.0, 0.0]))
    # 1e-7, not 1e-8: the constant-normal curvature (#7) changed the expression
    # tree, and these two mathematically identical paths now differ by 2.3e-8 of
    # round-off. The defect this catches (orientation not applied) is percent-scale.
    assert c_x == pytest.approx(c_ref, rel=1e-7)


def test_fiber_rotation_equals_relabelling_the_layup(plate_mesh, recorder):
    """[0/0/0] with fibres along y == [90/90/90] with fibres along default e0."""
    domain = ShellDomain(plate_mesh, element="CG2CG1")
    bcs = hbc.clamp(domain, where=clamped_at_x0)
    c_0_along_y = _compliance(domain, bcs, _layup([0, 0, 0]),
                              orientation=hmat.fiber_direction(domain, [0.0, 1.0, 0.0]))
    c_90 = _compliance(domain, bcs, _layup([90, 90, 90]))
    assert c_0_along_y == pytest.approx(c_90, rel=1e-6)
    c_0_at_45 = _compliance(domain, bcs, _layup([0, 0, 0]),
                            orientation=hmat.fiber_direction(domain, [1.0, 1.0, 0.0]))
    c_45 = _compliance(domain, bcs, _layup([45, 45, 45]))
    assert c_0_at_45 == pytest.approx(c_45, rel=1e-6)


def test_fiber_angle_equals_relabelling_and_is_differentiable(plate_mesh):
    """Rotating every fibre by phi == relabelling every ply by +phi."""
    domain = ShellDomain(plate_mesh, element="CG2CG1")
    bcs = hbc.clamp(domain, where=clamped_at_x0)
    phi0 = 0.37

    def run_via_fiber():
        rec = csdl.Recorder(inline=True); rec.start()
        phi = csdl.Variable(value=phi0, name="phi")
        material = hmat.laminate(domain, layup=_layup([10, -25, 55]), density=1.6e3 * np.ones(domain.n_cells),
                                 orientation=hmat.fiber_angle(domain, csdl.expand(phi, (domain.n_cells,))),
                                 constitutive_space=("DG", 0))
        compliance = out.compliance(solve(domain, material, hld.pressure(domain, 1e3), bcs))
        val = float(np.ravel(compliance.value)[0])
        g = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([compliance], [phi])[compliance, phi])[0])
        rec.stop()
        return val, g

    def run_via_plies():
        rec = csdl.Recorder(inline=True); rec.start()
        phi = csdl.Variable(value=phi0, name="phi")
        from hermit._laminate import Layup
        layup = Layup(_ud(), csdl.Variable(value=np.radians([10.0, -25.0, 55.0])) + csdl.expand(phi, (3,)),
                      np.full(3, 0.02 / 3), num_plies=3)
        material = hmat.laminate(domain, layup=layup, density=1.6e3 * np.ones(domain.n_cells),
                                 constitutive_space=("DG", 0))
        compliance = out.compliance(solve(domain, material, hld.pressure(domain, 1e3), bcs))
        val = float(np.ravel(compliance.value)[0])
        g = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([compliance], [phi])[compliance, phi])[0])
        rec.stop()
        return val, g

    vf, gf = run_via_fiber(); vp, gp = run_via_plies()
    print(f"\nvalue  fiber={vf:.8e}  ply={vp:.8e}")
    print(f"d/dphi fiber={gf:.8e}  ply={gp:.8e}")
    assert vf == pytest.approx(vp, rel=1e-6)
    assert gf == pytest.approx(gp, rel=1e-5)
    assert abs(gf) > 1.0


def test_fiber_angle_percell_is_reordered(plate_mesh, recorder):
    """A non-uniform per-cell fibre_angle must be gathered into FE cell order."""
    domain = ShellDomain(plate_mesh, element="CG2CG1")
    bcs = hbc.clamp(domain, where=clamped_at_x0)
    if np.array_equal(domain.cell_input_idx, np.arange(domain.n_cells)):
        print("\n(cell permutation is the identity on this stack -- weaker check)")
    frames = domain.local_frames()
    mids = domain.cell_centroids[domain.cell_input_idx]
    alpha_fe = 0.6 * mids[:, 0] / mids[:, 0].max()
    e0, e1 = frames[:, 0, :], frames[:, 1, :]
    dir_fe = np.cos(alpha_fe)[:, None] * e0 + np.sin(alpha_fe)[:, None] * e1
    c_ang = _compliance(domain, bcs, _layup([10, -25, 55]),
                        orientation=hmat.fiber_angle(domain, alpha_fe[domain.reverse_cell_idx]))
    c_dir = _compliance(domain, bcs, _layup([10, -25, 55]),
                        orientation=hmat.fiber_direction(domain, dir_fe[domain.reverse_cell_idx]))
    assert c_ang == pytest.approx(c_dir, rel=1e-6)
