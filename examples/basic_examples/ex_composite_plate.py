"""Composite cantilever plate

A cantilever plate with a carbon/epoxy laminate (classical lamination theory).
Sweeps a few layups to show the fiber-angle effect on tip compliance and Tsai-Wu
failure index, and takes the total derivative of each w.r.t. the ply angles.

    conda activate hermit
    python examples/basic_examples/ex_composite_plate.py
"""

import pathlib

import numpy as np
import csdl_alpha as csdl

import hermit as hm
from hermit import Layup
from caddee_materials import TransverseMaterial

MESH = pathlib.Path(__file__).parents[2] / "tests" / "meshes" / "plate_2x10_quad_4x20.xdmf"
PRESSURE_Z, TOTAL_H, DENSITY = 1.0e3, 0.02, 1.6e3   # thin laminate, loaded near its limit


def clamped_at_x0(x):
    return np.less(x[0], 1e-12)


def main():
    mesh = hm.read_mesh(MESH)

    rec = csdl.Recorder(inline=True)
    rec.start()

    domain = hm.ShellDomain(mesh, element="CG2CG1")
    ud = TransverseMaterial(name="ud", EA=138e9, ET=10e9, vA=0.34, GA=7e9, vT=0.67, density=DENSITY)
    ud.set_strength(F1t=1500e6, F1c=1200e6, F2t=50e6, F2c=200e6, F12=70e6, F23=40e6)
    loads = hm.pressure(domain, PRESSURE_Z)
    bcs = hm.clamp(domain, where=clamped_at_x0)

    for tag, plies in [("[0/0/0]", [0, 0, 0]), ("[0/90/0]", [0, 90, 0]),
                       ("[45/-45/45]", [45, -45, 45]), ("[90/90/90]", [90, 90, 90])]:
        angles = csdl.Variable(value=np.radians(np.array(plies, dtype=float)), name="angles")
        layup = Layup(ud, angles, np.full(len(plies), TOTAL_H / len(plies)), num_plies=len(plies))
        material = hm.laminate(domain, layup=layup, density=DENSITY)
        state = hm.solve(domain, material, loads, bcs)
        compliance, failure_index = hm.compliance(state), hm.failure_index(state)
        c = float(np.ravel(compliance.value)[0])
        fi = float(np.ravel(failure_index.value)[0])
        sim = csdl.experimental.PySimulator(rec)
        dc = np.asarray(sim.compute_totals([compliance], [angles])[compliance, angles]).ravel()
        dfi = np.asarray(sim.compute_totals([failure_index], [angles])[failure_index, angles]).ravel()
        print(f"  {tag:14s}  compliance={c:.4e}  Tsai-Wu FI={fi:.4e}")
        print(f"  {'':14s}  d(compliance)/d(deg)={np.round(np.radians(1) * dc, 6)}")
        print(f"  {'':14s}  d(FI)/d(deg)        ={np.round(np.radians(1) * dfi, 6)}")

    rec.stop()


if __name__ == "__main__":
    main()
