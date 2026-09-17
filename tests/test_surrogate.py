"""The solve/surrogate seam: a surrogate returns a ``ShellState``."""

import numpy as np
import pytest

import csdl_alpha as csdl
import hermit.bcs as hbc
import hermit.loads as hld
import hermit.material as hmat
import hermit.outputs as out
from hermit.domain import ShellDomain
from hermit._solve import ShellState, solve
from conftest import clamped_at_x0


def _inputs(mesh, ref):
    E, nu, rho, h, pz = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val", "h_val", "pressure_z"))
    domain = ShellDomain(mesh, element="CG2CG1")
    bcs = hbc.clamp(domain, where=clamped_at_x0)
    loads = hld.pressure(domain, pz)
    material = hmat.isotropic(domain, E=E * np.ones(domain.n_nodes), nu=nu * np.ones(domain.n_nodes),
                              thickness=h * np.ones(domain.n_nodes), density=rho * np.ones(domain.n_nodes),
                              constitutive_space=("Lagrange", 1))
    return domain, material, loads, bcs


def test_trivial_surrogate_feeds_the_real_forms_op(plate_mesh, cantilever_ref, recorder):
    """A replayed state still drives the real scalar-form postprocess operations."""
    domain, material, loads, bcs = _inputs(plate_mesh, cantilever_ref)
    known = np.asarray(cantilever_ref["disp_solid"])

    def surrogate(domain, material, loads, bcs, *, geometry=None):
        return ShellState(domain, geometry, material, loads, bcs,
                          disp_solid=csdl.Variable(value=known.copy()))

    state = surrogate(domain, material, loads, bcs)
    assert float(np.ravel(out.compliance(state).value)[0]) == pytest.approx(
        float(cantilever_ref["compliance"]), rel=1e-6)
    assert float(np.ravel(out.mass(state).value)[0]) == pytest.approx(float(cantilever_ref["mass"]), rel=1e-10)


class _ThicknessSurrogate:
    """Toy surrogate consuming thickness and pressure, never A/B/D/As."""

    def __init__(self, domain):
        self.ndof = domain.W.dofmap.index_map.size_local * domain.W.dofmap.index_map_bs

    def __call__(self, domain, material, loads, bcs, *, geometry=None):
        t = material.thickness.coeffs
        pz = csdl.sum(loads.pressure_terms[0].coeffs) / loads.pressure_terms[0].coeffs.shape[0]
        disp = csdl.expand(pz / (csdl.sum(t) / t.shape[0]) ** 3, (self.ndof,))
        return ShellState(domain, geometry, material, loads, bcs, disp_solid=disp)


def test_surrogate_with_reduced_inputs(plate_mesh, cantilever_ref, recorder):
    domain, _, loads, bcs = _inputs(plate_mesh, cantilever_ref)
    rho, h = (float(cantilever_ref[k]) for k in ("rho_val", "h_val"))
    material = hmat.thickness_only(domain, thickness=h * np.ones(domain.n_nodes),
                                   density=rho * np.ones(domain.n_nodes))
    surrogate = _ThicknessSurrogate(domain)
    state = surrogate(domain, material, loads, bcs)
    assert float(np.ravel(out.mass(state).value)[0]) == pytest.approx(float(cantilever_ref["mass"]), rel=1e-10)

    # own recorder for compute_totals: everything it reads has to live in *this*
    # graph, so the loads are rebuilt inside it (the outer fixture recorder's
    # Variables are not nodes here).
    rec = csdl.Recorder(inline=True); rec.start()
    s = csdl.Variable(value=1.0, name="s")
    loads2 = hld.pressure(domain, float(cantilever_ref["pressure_z"]))
    mat2 = hmat.thickness_only(domain, thickness=s * csdl.Variable(value=h * np.ones(domain.n_nodes)),
                               density=rho * np.ones(domain.n_nodes))
    compliance = out.compliance(surrogate(domain, mat2, loads2, bcs))
    ana = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([compliance], [s])[compliance, s])[0])
    rec.stop()
    assert np.isfinite(ana) and ana != 0.0


def test_surrogate_only_material_rejects_stress_outputs(plate_mesh, cantilever_ref, recorder):
    domain, _, loads, bcs = _inputs(plate_mesh, cantilever_ref)
    material = hmat.thickness_only(domain, thickness=float(cantilever_ref["h_val"]) * np.ones(domain.n_nodes),
                                   density=float(cantilever_ref["rho_val"]) * np.ones(domain.n_nodes))
    with pytest.raises(ValueError, match="isotropic-only.*failure_index"):
        out.aggregated_stress(_ThicknessSurrogate(domain)(domain, material, loads, bcs))


def test_real_solve_rejects_surrogate_only_material(plate_mesh, cantilever_ref, recorder):
    domain, _, loads, bcs = _inputs(plate_mesh, cantilever_ref)
    material = hmat.thickness_only(domain, thickness=float(cantilever_ref["h_val"]) * np.ones(domain.n_nodes),
                                   density=float(cantilever_ref["rho_val"]) * np.ones(domain.n_nodes))
    with pytest.raises(ValueError, match="A/B/D/As"):
        solve(domain, material, loads, bcs)
