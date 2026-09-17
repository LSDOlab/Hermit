"""Subdomain (mesh-tag) p-norm stress outputs.

Regions come from ``ShellDomain(mesh, cell_tags=..., regions=...)``; ``region=`` and
the aggregation ``rho`` / ``m`` are arguments of ``hm.pnorm_stress`` /
``hm.aggregated_stress``.
"""

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


def _cell_tags(mesh, split_x=5.0):
    """Tag 1 = root half (x < split), tag 2 = tip half."""
    import dolfinx

    tdim = mesh.topology.dim
    nc = mesh.topology.index_map(tdim).size_local
    cells = np.arange(nc, dtype=np.int32)
    midx = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)
    vals = np.where(midx[:, 0] < split_x, 1, 2).astype(np.int32)
    return dolfinx.mesh.meshtags(mesh, tdim, cells, vals)


def _domain(plate_mesh, tagged=True):
    tags = _cell_tags(plate_mesh) if tagged else None
    regions = {"root": 1, "tip": 2} if tagged else None
    return ShellDomain(plate_mesh, element="CG2CG1", cell_tags=tags, regions=regions)


def _state(domain, ref, thickness=None):
    nn = int(ref["n_nodes"])
    E, nu, rd, h = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val", "h_val"))
    pz = float(ref["pressure_z"])
    t = h * np.ones(nn) if thickness is None else thickness
    material = hmat.isotropic(domain, E=E * np.ones(nn), nu=nu * np.ones(nn),
                              thickness=t, density=rd * np.ones(nn),
                              constitutive_space=("Lagrange", 1))
    return solve(domain, material, hld.pressure(domain, pz),
                 hbc.clamp(domain, where=clamped_at_x0))


def _pnorm(state, region):
    return out.pnorm_stress(state, rho=4.0, m=1e-3, region=region)


def test_root_region_more_stressed_than_tip(plate_mesh, cantilever_ref, recorder):
    state = _state(_domain(plate_mesh), cantilever_ref)
    whole = float(np.ravel(_pnorm(state, None).value)[0])
    root = float(np.ravel(_pnorm(state, "root").value)[0])
    tip = float(np.ravel(_pnorm(state, 2).value)[0])
    print(f"\npnorm stress  whole={whole:.4e}  root={root:.4e}  tip={tip:.4e}")
    assert root > tip                      # bending moment peaks at the clamp
    assert tip < whole < root or root > whole > tip


def test_unknown_region_raises(plate_mesh, cantilever_ref, recorder):
    state = _state(_domain(plate_mesh), cantilever_ref)
    with pytest.raises(KeyError):
        _pnorm(state, "middle")
    plain = _state(_domain(plate_mesh, tagged=False), cantilever_ref)
    with pytest.raises(ValueError, match="cell_tags"):
        _pnorm(plain, "root")


def test_region_pnorm_adjoint_wrt_thickness(plate_mesh, cantilever_ref):
    h, nn = float(cantilever_ref["h_val"]), int(cantilever_ref["n_nodes"])

    def run(scale, want_grad):
        rec = csdl.Recorder(inline=True); rec.start()
        s = csdl.Variable(value=float(scale), name="s")
        t = s * csdl.Variable(value=h * np.ones(nn))
        o = _pnorm(_state(_domain(plate_mesh), cantilever_ref, thickness=t), "root")
        val = float(np.ravel(o.value)[0])
        g = None
        if want_grad:
            g = float(np.ravel(csdl.experimental.PySimulator(rec).compute_totals([o], [s])[o, s])[0])
        rec.stop()
        return val, g

    _, ana = run(1.0, True)
    d = 1e-4
    vp, _ = run(1.0 + d, False)
    vm, _ = run(1.0 - d, False)
    cd = (vp - vm) / (2 * d)
    rel = abs(ana - cd) / abs(cd)
    print(f"\nregion pnorm d/ds : analytic={ana:.6e}  cd={cd:.6e}  rel={rel:.2e}")
    assert cd < 0 and rel < 2e-3
