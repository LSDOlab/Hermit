"""Capture RMShell (femo_alpha, dev_coupling branch) ground-truth results.

Run in the ``femo_ref`` conda env:

    source ../scripts/conda_init.sh && conda activate femo_ref
    python tests/capture_femo_reference.py

Writes ``tests/data/rmshell_cantilever.npz`` — the reference Hermit is validated against.

Case: 2 x 10 cantilever plate, quad_4_20 mesh, clamped at x = 0, uniform transverse
pressure. Isotropic, nodal (CG1) material. Penalty BCs.
"""
import pathlib

import numpy as np
import dolfinx
from mpi4py import MPI
import csdl_alpha as csdl

from femo_alpha.rm_shell.rm_shell_model import RMShellModel

HERE = pathlib.Path(__file__).parent
MESH = HERE / "meshes" / "plate_2x10_quad_4x20.xdmf"
OUT = HERE / "data" / "rmshell_cantilever.npz"

# --- case parameters -------------------------------------------------------
E_VAL = 4.32e8
NU_VAL = 0.0
H_VAL = 0.2
RHO_VAL = 1.0
PRESSURE_Z = 2.0  # force per unit area, +z
WIDTH, LENGTH = 2.0, 10.0
DOLFIN_EPS = 3e-16


def clamped_boundary(x):
    return np.less(x[0], 0.0 + DOLFIN_EPS)


def main():
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, str(MESH), "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    nn = mesh.topology.index_map(0).size_local
    nel = mesh.topology.index_map(mesh.topology.dim).size_local
    print(f"mesh: {nn} nodes, {nel} elements")

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    thickness = csdl.Variable(value=H_VAL * np.ones(nn), name="thickness")
    E = csdl.Variable(value=E_VAL * np.ones(nn), name="E")
    nu = csdl.Variable(value=NU_VAL * np.ones(nn), name="nu")
    density = csdl.Variable(value=RHO_VAL * np.ones(nn), name="density")

    pressure = csdl.Variable(value=np.zeros((nn, 3)), name="nodal_pressure")
    pressure.value[:, 2] = PRESSURE_Z

    shell = RMShellModel(
        mesh,
        shell_bc_func=clamped_boundary,
        element_wise_material=False,
        PENALTY_BC=True,
        record=False,
    )
    material = shell.material_inputs.from_isotropic(
        E=E, nu=nu, thickness=thickness, density=density
    )
    loads = shell.load_inputs.from_fields(nodal_pressure=pressure)

    state = shell.solve(material=material, loads=loads)
    out = shell.post.clear().add_default_outputs().compute(state=state)

    # FE-ordered / raw quantities straight off the solve FEA object
    fea = shell.fea
    w_func = fea.states_dict["disp_solid"]["function"]
    F_solid = fea.inputs_dict["F_solid"]["function"].x.array.copy()
    A_fe = fea.inputs_dict["A"]["function"].x.array.copy()
    thickness_fe = fea.inputs_dict["thickness"]["function"].x.array.copy()
    mesh_nodes_ref = shell.reference_mesh_nodes.copy()

    data = dict(
        # case
        E_val=E_VAL, nu_val=NU_VAL, h_val=H_VAL, rho_val=RHO_VAL,
        pressure_z=PRESSURE_Z, width=WIDTH, length=LENGTH,
        n_nodes=nn, n_elem=nel,
        mesh_nodes_ref=mesh_nodes_ref,
        material_input_indices=np.asarray(shell._material_indices(shell.shell_pde)),
        # solve outputs
        disp_solid=np.asarray(out.disp_solid.value),
        disp_solid_dofs=int(out.disp_solid.value.size),
        compliance=float(out.compliance.value),
        mass=float(out.mass.value),
        cg=np.asarray(out.cg.value),
        elastic_energy=float(out.elastic_energy.value),
        # Captured for the record, no longer gated against: femo's aggregate is its
        # softabs floor ln(2)/50 rescaled (958117.0189), a constant independent of the
        # displacement field -- see test_outputs_api.test_scalar_outputs_match_reference.
        # Same for its derivative, which is identically zero for the same reason.
        aggregated_stress=float(out.aggregated_stress.value),
        disp_extracted=np.asarray(out.disp_extracted.value),
        displacements=np.asarray(out.displacements.value),
        rotations=np.asarray(out.rotations.value),
        # FE-ordered intermediates
        F_solid_fe=F_solid,
        A_fe=A_fe,
        thickness_fe=thickness_fe,
        # analytic check
        eb_tip_deflection=float(
            PRESSURE_Z * WIDTH * LENGTH**4 / (8 * E_VAL * (WIDTH * H_VAL**3 / 12))
        ),
    )

    # --- total derivatives (analytic adjoint only; FD is too slow here) ----
    sim = csdl.experimental.PySimulator(recorder)
    for oname, ovar in [
        ("compliance", out.compliance),
        ("aggregated_stress", out.aggregated_stress),
        ("mass", out.mass),
    ]:
        jac = sim.compute_totals([ovar], [thickness])
        key = list(jac.keys())[0]
        data[f"dtot_{oname}_dthickness"] = np.asarray(jac[key]).ravel()

    recorder.stop()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **data)
    print(f"wrote {OUT}")
    print(f"  EB tip deflection      : {data['eb_tip_deflection']:.6e}")
    print(f"  RM max |disp_solid|     : {np.abs(data['disp_solid']).max():.6e}")
    print(f"  max nodal w (disp_extr) : {data['disp_extracted'][:, 2].max():.6e}")
    print(f"  compliance              : {data['compliance']:.6e}")
    print(f"  mass                    : {data['mass']:.6e}  (exact {RHO_VAL*H_VAL*WIDTH*LENGTH})")
    print(f"  aggregated_stress       : {data['aggregated_stress']:.6e}")
    print(f"  |d compliance/d thickness|: {np.linalg.norm(data['dtot_compliance_dthickness']):.6e}")


if __name__ == "__main__":
    main()
