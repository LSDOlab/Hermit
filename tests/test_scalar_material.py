"""Material inputs may be a single value instead of a per-node / per-cell array."""

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


def _iso(dom, E, nu, thickness, density):
    return hmat.isotropic(dom, E=E, nu=nu, thickness=thickness, density=density,
                          constitutive_space=("Lagrange", 1))


def _run(plate_mesh, ref, material_fn):
    dom = ShellDomain(plate_mesh, element="CG2CG1")
    return solve(dom, material_fn(dom), hld.pressure(dom, float(ref["pressure_z"])),
                 hbc.clamp(dom, where=clamped_at_x0))


def test_scalar_equals_expanded(plate_mesh, cantilever_ref, recorder):
    ref = cantilever_ref
    E, nu, h, rho = (float(ref[k]) for k in ("E_val", "nu_val", "h_val", "rho_val"))
    nn = int(ref["n_nodes"])

    scal = _run(plate_mesh, ref, lambda d: _iso(d, E, nu, h, rho))
    arr = _run(plate_mesh, ref, lambda d: _iso(
        d, E * np.ones(nn), nu * np.ones(nn), h * np.ones(nn), rho * np.ones(nn)))

    assert float(np.ravel(out.compliance(scal).value)[0]) == pytest.approx(
        float(np.ravel(out.compliance(arr).value)[0]), rel=1e-12)
    assert float(np.ravel(out.mass(scal).value)[0]) == pytest.approx(
        float(np.ravel(out.mass(arr).value)[0]), rel=1e-12)


def test_scalar_and_array_mixed(plate_mesh, cantilever_ref, recorder):
    """A per-node thickness field with scalar E / nu / density."""
    ref = cantilever_ref
    E, nu, h, rho = (float(ref[k]) for k in ("E_val", "nu_val", "h_val", "rho_val"))
    nn = int(ref["n_nodes"])
    t = h * np.linspace(0.8, 1.2, nn)

    mixed = _run(plate_mesh, ref, lambda d: _iso(d, E, nu, t, rho))
    full = _run(plate_mesh, ref, lambda d: _iso(
        d, E * np.ones(nn), nu * np.ones(nn), t, rho * np.ones(nn)))
    assert float(np.ravel(out.compliance(mixed).value)[0]) == pytest.approx(
        float(np.ravel(out.compliance(full).value)[0]), rel=1e-12)


def test_inconsistent_lengths_raise(plate_mesh, recorder):
    """Each field carries its own space now, so there is no cross-field length check --
    a length matching neither n_nodes, n_cells nor an explicit space is rejected by
    ``as_field`` instead."""
    dom = ShellDomain(plate_mesh, element="CG2CG1")
    with pytest.raises(ValueError, match="cannot resolve"):
        _iso(dom, np.ones(5), 0.3, np.ones(7), 1.0)


def test_scalar_thickness_derivative(plate_mesh, cantilever_ref):
    """d(compliance)/d(scalar thickness) through the broadcast vs central difference."""
    ref = cantilever_ref
    E, nu, h, rho = (float(ref[k]) for k in ("E_val", "nu_val", "h_val", "rho_val"))

    def run(hval, grad):
        rec = csdl.Recorder(inline=True); rec.start()
        t = csdl.Variable(value=np.atleast_1d(hval), name="t")
        o = out.compliance(_run(plate_mesh, ref, lambda d: _iso(d, E, nu, t, rho)))
        val = float(np.ravel(o.value)[0])
        g = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([o], [t])[o, t])[0]) if grad else None
        rec.stop()
        return val, g

    _, ana = run(h, True)
    d = 1e-4 * h
    vp, _ = run(h + d, False)
    vm, _ = run(h - d, False)
    fd = (vp - vm) / (2 * d)
    rel = abs(ana - fd) / abs(fd)
    print(f"\nd(compliance)/dt: ana={ana:+.6e} fd={fd:+.6e} rel={rel:.2e}")
    assert rel < 1e-5
