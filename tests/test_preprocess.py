"""Preprocess FE inputs retain the independently captured femo gates."""

import numpy as np

import hermit.loads as hld
import hermit.material as hmat
from hermit.domain import ShellDomain


def test_fe_inputs_match_reference(plate_mesh, cantilever_ref, recorder):
    ref = cantilever_ref
    E, nu, h, rho, pz = (float(ref[k]) for k in ("E_val", "nu_val", "h_val", "rho_val", "pressure_z"))
    domain = ShellDomain(plate_mesh, element="CG2CG1")
    material = hmat.isotropic(domain, E=E * np.ones(domain.n_nodes), nu=nu * np.ones(domain.n_nodes),
                              thickness=h * np.ones(domain.n_nodes), density=rho * np.ones(domain.n_nodes),
                              constitutive_space=("Lagrange", 1))
    loads = hld.traction(domain, np.column_stack((np.zeros(domain.n_nodes), np.zeros(domain.n_nodes),
                                                   pz * np.ones(domain.n_nodes))))
    assert np.allclose(material.thickness.coeffs.value, ref["thickness_fe"], rtol=1e-12)
    assert np.allclose(material.A.coeffs.value.reshape(-1), ref["A_fe"], rtol=1e-10, atol=0)
    assert np.allclose(loads.traction_terms[0].coeffs.value, ref["F_solid_fe"], rtol=1e-10, atol=1e-12)
    assert np.allclose(domain.node_coords, ref["mesh_nodes_ref"])
