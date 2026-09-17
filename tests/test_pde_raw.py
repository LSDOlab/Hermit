"""Phase 1 gate: a raw (no-CSDL) shell solve must match the femo_alpha reference.

Builds ``ShellPDE`` directly, sets a constant isotropic ABD, applies penalty BCs, solves
with MUMPS, and compares the mixed displacement vector + tip deflection to
``tests/data/rmshell_cantilever.npz``.
"""

import numpy as np
import pytest

from hermit.fenics.assembly import assemble_scalar, get_array, set_array
from hermit.fenics.solvers import linear_solve
import hermit.bcs as hbc
from hermit.domain import ShellDomain
from hermit._solve import _pde_for

from conftest import clamped_at_x0


def _isotropic_abd(E, nu, h):
    C = (E / (1 - nu**2)) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1 - nu)]]
    )
    A = h * C
    D = h**3 / 12 * C
    G = E / (2 * (1 + nu))
    As = 0.833 * G * h * np.eye(2)
    return A, np.zeros((3, 3)), D, As


def _fill_tensor(func, mat):
    """Set every node/cell of a tensor Function to the constant matrix ``mat``."""
    n = func.x.array.size // mat.size
    func.x.array[:] = np.tile(mat.reshape(-1), n)
    func.x.scatter_forward()


def test_raw_solve_matches_reference(plate_mesh, cantilever_ref):
    ref = cantilever_ref
    E, nu, h = float(ref["E_val"]), float(ref["nu_val"]), float(ref["h_val"])
    rho, pz = float(ref["rho_val"]), float(ref["pressure_z"])

    domain = ShellDomain(plate_mesh, element="CG2CG1")
    pde = _pde_for(domain)
    bc = hbc.clamp(domain, where=clamped_at_x0).to_bc_data()

    A, B, D, As = _isotropic_abd(E, nu, h)
    _fill_tensor(pde.A, A)
    _fill_tensor(pde.B, B)
    _fill_tensor(pde.D, D)
    _fill_tensor(pde.As, As)
    pde.h.x.array[:] = h
    pde.E.x.array[:] = E
    pde.nu.x.array[:] = nu
    pde.density.x.array[:] = rho
    pde.f.x.array.reshape(-1, 3)[:, 2] = pz  # uniform +z pressure
    pde.f.x.scatter_forward()

    res = pde.residual_form(penalty=True, dss=bc.dss, dSS=bc.dSS)
    pde.w.x.array[:] = 0.0
    linear_solve(res, pde.w, bcs=bc.strong)

    disp = get_array(pde.w)
    ref_disp = ref["disp_solid"]
    assert disp.shape == ref_disp.shape, (disp.shape, ref_disp.shape)

    # mixed-space DOF ordering is identical (same mesh, same element) -> compare directly
    rel = np.linalg.norm(disp - ref_disp) / np.linalg.norm(ref_disp)
    print(f"\n||disp - ref|| / ||ref|| = {rel:.3e}")
    print(f"max|disp| hermit={np.abs(disp).max():.6e}  ref={np.abs(ref_disp).max():.6e}  "
          f"EB={float(ref['eb_tip_deflection']):.6e}")

    assert rel < 1e-8

    # scalar forms assembled directly off the solved state
    mass = assemble_scalar(pde.mass_form())
    assert mass == pytest.approx(float(ref["mass"]), rel=1e-10)

    compliance = assemble_scalar(pde.compliance_form())
    print(f"compliance hermit={compliance:.6e}  ref={float(ref['compliance']):.6e}")
    assert compliance == pytest.approx(float(ref["compliance"]), rel=1e-8)

    cgx, cgy, cgz, m_tot = (assemble_scalar(fx) for fx in pde.cg_forms())
    cg = np.array([cgx, cgy, cgz]) / m_tot
    assert cg == pytest.approx(np.asarray(ref["cg"]), rel=1e-8, abs=1e-9)

    ee = assemble_scalar(pde.elastic_energy_form())
    assert ee == pytest.approx(float(ref["elastic_energy"]), rel=1e-7)


if __name__ == "__main__":
    import pathlib, sys
    import dolfinx
    from mpi4py import MPI

    HERE = pathlib.Path(__file__).parent
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, str(HERE / "meshes" / "plate_2x10_quad_4x20.xdmf"), "r") as x:
        mesh = x.read_mesh(name="Grid")
    ref = dict(np.load(HERE / "data" / "rmshell_cantilever.npz"))
    sys.path.insert(0, str(HERE))
    test_raw_solve_matches_reference(mesh, ref)
    print("OK")
