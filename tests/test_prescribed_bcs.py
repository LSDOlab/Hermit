"""Patch tests for non-zero prescribed shell boundary conditions.

The ordinary patch tests use triangles. The warped-quadrilateral gates separately
cover rigid-body objectivity in both the energy and recovered-stress paths.
"""

import pathlib
import sys

import numpy as np
import pytest
import hermit as hm

# Path insert, not `from examples.verification...`: `examples` is not a package and
# only resolves as one when pytest happens to run with the component repo as rootdir.
# The documented workspace command is `./scripts/run pytest repos/hermit/tests` from
# ~/work/hermit, where it does not. Same idiom as tests/test_edge_loads.py.
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples" / "verification"))
from _geometry import structured_surface  # noqa: E402


def _all_boundary(x):
    return (np.isclose(x[0], 0.0) | np.isclose(x[0], 10.0)
            | np.isclose(x[1], 0.0) | np.isclose(x[1], 2.0))


def _problem(tri_mesh):
    dom = hm.ShellDomain(tri_mesh, element="CG2CG1")
    mat = hm.isotropic(dom, E=100.0, nu=0.25, thickness=0.1, density=1.0)
    return dom, mat, hm.Loads(dom)


def _field_values(field):
    return np.asarray(field.coeffs.value).reshape(-1, field.block_size)


def test_prescribed_membrane_patch_strong_callable(tri_mesh, recorder):
    """An affine in-plane displacement gives constant membrane strain exactly."""
    dom, mat, loads = _problem(tri_mesh)
    a, b, c, d = 0.02, -0.03, 0.04, 0.01

    def displacement(x):  # hm.from_function convention: (N, 3) -> (N, 6)
        out = np.zeros((len(x), 6))
        out[:, 0] = a * x[:, 0] + b * x[:, 1]
        out[:, 1] = c * x[:, 0] + d * x[:, 1]
        # The shell's drilling strain is curl(u)/2 + theta_z.  This constant
        # rotation makes the affine membrane state an exact zero-drilling state.
        out[:, 5] = 0.5 * (c - b)
        return out

    state = hm.solve(dom, mat, loads,
                     hm.clamp(dom, where=_all_boundary, method="strong", value=displacement))
    strain, _, shear = hm.strain_fields(state, space=("DG", 0), method="interpolate", frame="global")
    # Hermit's displacement/curvature convention reports the negative of this
    # Cartesian engineering strain; the important patch-test gate is its exact
    # constancy in every interior cell.
    assert np.max(np.abs(_field_values(strain) + [a, d, 0.0, 0.0, 0.0, b + c])) < 2e-13
    assert np.max(np.abs(_field_values(shear))) < 2e-13


def test_prescribed_bending_patch_strong(tri_mesh, recorder):
    """Quadratic transverse displacement plus linear rotation has constant curvature."""
    dom, mat, loads = _problem(tri_mesh)
    kxx, kyy, kxy = 0.06, -0.04, 0.03

    def bending_state(x):
        X, Y = x[:, 0], x[:, 1]
        out = np.zeros((len(x), 6))
        out[:, 2] = 0.5 * kxx * X**2 + kxy * X * Y + 0.5 * kyy * Y**2
        out[:, 3] = kxy * X + kyy * Y
        out[:, 4] = -kxx * X - kxy * Y
        return out

    state = hm.solve(dom, mat, loads,
                     hm.clamp(dom, where=_all_boundary, method="strong", value=bending_state))
    _, curvature, shear = hm.strain_fields(state, space=("DG", 0), method="interpolate", frame="global")
    assert np.max(np.abs(_field_values(curvature) + [kxx, kyy, 0.0, 0.0, 0.0, 2 * kxy])) < 2e-12
    assert np.max(np.abs(_field_values(shear))) < 2e-12


@pytest.mark.parametrize("method", ["penalty", "strong"])
def test_prescribed_rigid_motion_has_zero_strain_energy(method, tri_mesh, recorder):
    """A solve under exact translation + infinitesimal rotation remains strain-free."""
    dom, mat, loads = _problem(tri_mesh)
    translation = np.array([0.3, -0.2, 0.1])
    omega = np.array([0.04, -0.05, 0.03])

    def rigid_motion(x):
        out = np.empty((len(x), 6))
        out[:, :3] = translation + np.cross(omega, x)
        out[:, 3:] = omega
        return out

    state = hm.solve(dom, mat, loads,
                     hm.clamp(dom, where=_all_boundary, method=method, value=rigid_motion))
    energy = float(np.asarray(hm.elastic_energy(state).value).reshape(-1)[0])
    # Penalty enforces an exact representable field through a finite (but huge)
    # coefficient, hence its conditioning-limited tolerance is intentionally looser.
    assert abs(energy) < (1e-18 if method == "strong" else 1e-12)


def _warped_rigid_motion_state():
    # One bilinear saddle cell is deliberately non-planar, and every one of its
    # dofs is prescribed. That prevents an unconstrained interior from relaxing
    # away any spurious strain that the exact boundary field must expose.
    mesh = structured_surface(lambda u, v: (u, v, 0.01 * u * v), 1, 1, cell="quad")
    dom = hm.ShellDomain(mesh, element="CG2CG1")
    mat = hm.isotropic(dom, E=100.0, nu=0.25, thickness=0.1, density=1.0)
    translation = np.array([0.3, -0.2, 0.1])
    omega = np.array([0.04, -0.05, 0.03])

    def rigid_motion(x):
        out = np.empty((len(x), 6))
        out[:, :3] = translation + np.cross(omega, x)
        out[:, 3:] = omega
        return out

    return hm.solve(
        dom,
        mat,
        hm.Loads(dom),
        hm.clamp(dom, where=lambda x: np.ones(x.shape[1], dtype=bool),
                 method="strong", value=rigid_motion),
    )


def test_prescribed_rigid_motion_on_warped_quads_has_zero_strain_energy(recorder):
    """A whole-boundary exact RBM solve stores no energy on a warped quad."""
    state = _warped_rigid_motion_state()
    energy = float(np.asarray(hm.elastic_energy(state).value).reshape(-1)[0])
    print(f"\nwarped-quad rigid-motion energy={energy:.16e}")
    assert abs(energy) < 1e-18, f"warped quad rigid motion stored energy {energy:.6e}"


def test_prescribed_rigid_motion_on_warped_quads_has_zero_recovered_stress(recorder):
    """Stress recovery uses the same objective through-thickness strain as energy.

    Both quantities are checked: von Mises exercises the public stress-output path,
    while ``eps_top = eps_mid - h*kappa/2`` pins the strain and sign beneath it.
    The prescribed displacement and rotation have nonzero magnitude, so an
    accidentally zero state cannot make this gate pass.
    """
    state = _warped_rigid_motion_state()
    disp = np.asarray(hm.nodal_displacement(state).value)
    rotation = np.asarray(hm.nodal_rotation(state).value)
    assert np.max(np.abs(disp)) > 0.1 and np.max(np.abs(rotation)) > 0.01

    mid, curvature, _ = hm.strain_fields(
        state, space=("DG", 2), method="interpolate", frame="local")
    eps_top = (_field_values(mid) - 0.05 * _field_values(curvature))
    vm_top = _field_values(hm.stress_field(
        state, space=("DG", 2), method="interpolate", surface="top"))

    strain_squared = np.max(np.sum(eps_top**2, axis=1))
    peak_vm = np.max(np.abs(vm_top))
    print(f"\nwarped-quad rigid-motion top-strain^2={strain_squared:.16e} "
          f"peak-von-Mises={peak_vm:.16e}")
    assert strain_squared < 1e-26, f"warped quad RBM top strain^2 {strain_squared:.6e}"
    assert peak_vm < 1e-11, f"warped quad RBM recovered von Mises {peak_vm:.6e}"


def test_prescribed_gauge_value_is_honoured(tri_mesh, recorder):
    """A gauge set consistently with the boundary motion must not fight it.

    This fails loudly if ``gauge`` ignores ``value=``: the interior point would be
    held at the origin while the whole boundary moves rigidly, which is a large
    strain state, not a vanishing one.  ``gauge`` is the one strong builder whose
    target is located by proximity rather than by a ``where`` predicate, so it does
    not share ``_strong_bcs_where``'s coverage.
    """
    dom, mat, loads = _problem(tri_mesh)
    translation = np.array([0.3, -0.2, 0.1])
    omega = np.array([0.04, -0.05, 0.03])

    def rigid_motion(x):
        out = np.empty((len(x), 6))
        out[:, :3] = translation + np.cross(omega, x)
        out[:, 3:] = omega
        return out

    bcs = (hm.clamp(dom, where=_all_boundary, method="strong", value=rigid_motion)
           + hm.gauge(dom, at=np.array([5.0, 1.0, 0.0]),
                      dofs=("ux", "uy", "uz", "rx", "ry", "rz"), value=rigid_motion))
    state = hm.solve(dom, mat, loads, bcs)
    energy = float(np.asarray(hm.elastic_energy(state).value).reshape(-1)[0])
    assert abs(energy) < 1e-18
