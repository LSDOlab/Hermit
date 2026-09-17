"""Fibre orientation and field output

A carbon/epoxy cantilever plate whose fibres run at a chosen angle to the beam axis.
Shows: aligning the laminate with a global ``fiber_direction``, reading the per-cell
strain field back as a ``Field``, and re-expressing it in the global frame.

    conda activate hermit
    python examples/basic_examples/ex_fiber_orientation.py
"""

import pathlib

import numpy as np
import csdl_alpha as csdl

import hermit as hm
from hermit import Layup
from caddee_materials import TransverseMaterial

MESH = pathlib.Path(__file__).parents[2] / "tests" / "meshes" / "plate_2x10_quad_4x20.xdmf"


def clamped_at_x0(x):
    return np.less(x[0], 1e-12)


def main():
    mesh = hm.read_mesh(MESH)

    rec = csdl.Recorder(inline=True)
    rec.start()

    domain = hm.ShellDomain(mesh, element="CG2CG1")
    ud = TransverseMaterial(name="ud", EA=138e9, ET=10e9, vA=0.34, vT=0.67, GA=7e9, density=1.6e3)
    ud.set_strength(F1t=1500e6, F1c=1200e6, F2t=50e6, F2c=200e6, F12=70e6, F23=40e6)
    layup = Layup(ud, csdl.Variable(value=np.radians([0.0, 0.0, 0.0])),
                  np.full(3, 0.02 / 3), num_plies=3)
    loads = hm.pressure(domain, 1.0e3)
    bcs = hm.clamp(domain, where=clamped_at_x0)

    print("  fibre angle (from beam axis) | compliance | Tsai-Wu FI")
    for deg in (0, 15, 30, 45, 90):
        d = [np.cos(np.radians(deg)), np.sin(np.radians(deg)), 0.0]
        material = hm.laminate(domain, layup=layup, density=1.6e3,
                               orientation=hm.fiber_direction(domain, d))
        state = hm.solve(domain, material, loads, bcs)
        print(f"  {deg:>6d} deg                     | {float(np.ravel(hm.compliance(state).value)[0]):.4e} "
              f"| {float(np.ravel(hm.failure_index(state).value)[0]):.4e}")

    # strain field for the 30-degree case, local frame vs global
    d30 = [np.cos(np.radians(30)), np.sin(np.radians(30)), 0.0]
    material = hm.laminate(domain, layup=layup, density=1.6e3,
                           orientation=hm.fiber_direction(domain, d30))
    state = hm.solve(domain, material, loads, bcs)
    _, kap, _ = hm.strain_fields(state)
    print(f"\n  curvature field: {kap.values.shape} (n_cells, [xx, yy, 2xy])")
    print(f"    root cell, element frame : {np.round(kap.values[0], 6)}")
    print(f"    root cell, fibre frame   : {np.round(kap.to_frame(d30).values[0], 6)}")
    print(f"    root cell, global frame  : {np.round(kap.to_global().values[0], 6)}")
    print(f"    eval at cell 8 centroid  : {np.round(np.ravel(kap.eval([8], [[0.5, 0.5]]).value), 6)}")

    rec.stop()


if __name__ == "__main__":
    main()
