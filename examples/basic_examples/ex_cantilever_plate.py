"""Cantilever plate forward analysis

A clamped cantilever plate under uniform transverse pressure. Mirrors the femo_alpha
``ex_simple_shell.py`` case (E=4.32e8, nu=0, h=0.2, 2x10 plate, clamped at x=0) and
prints the Hermit tip deflection next to Euler-Bernoulli beam theory.

    conda activate hermit
    python examples/basic_examples/ex_cantilever_plate.py
"""

import pathlib

import numpy as np
import csdl_alpha as csdl

import hermit as hm

MESH = pathlib.Path(__file__).parents[2] / "tests" / "meshes" / "plate_2x10_quad_4x20.xdmf"

E_VAL, NU_VAL, H_VAL, RHO_VAL = 4.32e8, 0.0, 0.2, 1.0
PRESSURE_Z = 2.0
WIDTH, LENGTH = 2.0, 10.0


def clamped_at_x0(x):
    return np.less(x[0], 1e-12)


def main():
    mesh = hm.read_mesh(MESH)

    rec = csdl.Recorder(inline=True)
    rec.start()

    domain = hm.ShellDomain(mesh, element="CG2CG1")
    material = hm.isotropic(domain, E=E_VAL, nu=NU_VAL, thickness=H_VAL, density=RHO_VAL)
    state = hm.solve(domain, material, hm.pressure(domain, PRESSURE_Z),
                     hm.clamp(domain, where=clamped_at_x0))
    compliance, mass = hm.compliance(state), hm.mass(state)
    cg, elastic_energy = hm.center_of_gravity(state), hm.elastic_energy(state)

    rec.stop()

    Ix = WIDTH * H_VAL**3 / 12.0
    eb = PRESSURE_Z * WIDTH * LENGTH**4 / (8.0 * E_VAL * Ix)
    print(f"  nodes / elements        : {domain.n_nodes} / {domain.n_cells}")
    print(f"  Euler-Bernoulli tip w    : {eb:.6e}")
    print(f"  Reissner-Mindlin max|w|  : {np.abs(state.disp_solid.value).max():.6e}")
    print(f"  compliance               : {float(np.ravel(compliance.value)[0]):.6e}")
    print(f"  mass                     : {float(np.ravel(mass.value)[0]):.6e}"
          f"  (exact {RHO_VAL * H_VAL * WIDTH * LENGTH})")
    print(f"  cg                       : {np.ravel(cg.value)}")
    print(f"  elastic energy           : {float(np.ravel(elastic_energy.value)[0]):.6e}")


if __name__ == "__main__":
    main()
