"""One ``ShellDomain`` / BC bundle / material serves independent load cases."""

import numpy as np
import pytest

import hermit.bcs as hbc
import hermit.loads as hld
import hermit.material as hmat
import hermit.outputs as out
from hermit.domain import ShellDomain
from hermit._solve import solve
from conftest import clamped_at_x0


def test_repeated_evaluate_on_one_model(plate_mesh, cantilever_ref, recorder):
    ref = cantilever_ref
    E, nu, rd, h = (float(ref[k]) for k in ("E_val", "nu_val", "rho_val", "h_val"))
    domain = ShellDomain(plate_mesh, element="CG2CG1")
    bcs = hbc.clamp(domain, where=clamped_at_x0)
    material = hmat.isotropic(domain, E=E * np.ones(domain.n_nodes), nu=nu * np.ones(domain.n_nodes),
                              thickness=h * np.ones(domain.n_nodes), density=rd * np.ones(domain.n_nodes),
                              constitutive_space=("Lagrange", 1))
    comps = [float(np.ravel(out.compliance(solve(domain, material, hld.pressure(domain, pz), bcs)).value)[0])
             for pz in (1.0, 2.0, 3.0)]
    assert comps[1] / comps[0] == pytest.approx(4.0, rel=1e-6)
    assert comps[2] / comps[0] == pytest.approx(9.0, rel=1e-6)
