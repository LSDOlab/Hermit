"""An isotropic-ply laminate matches an isotropic shell

An isotropic material split into arbitrarily oriented plies has exactly the same
ABD and transverse-shear stiffness as a single isotropic layer of the same total
thickness.  This script solves both material paths on the same cantilever mesh and
load, using the isotropic result computed in the same run as its reference.  Both
constructors use Hermit's ``0.833`` shear correction; specifying it here documents
the convention required for the equality.
"""

import pathlib
import sys

import numpy as np
import csdl_alpha as csdl
from caddee_materials import IsotropicMaterial

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import rect_plate  # noqa: E402
from _harness import Case, main, node_nearest  # noqa: E402

E, NU, H, RHO, LOAD = 70e9, 0.29, 0.02, 2700.0, 2.0e4


def solve_at(n):
    """Largest relative disagreement of compliance, mass, and tip displacement."""
    rec = csdl.Recorder(inline=True); rec.start()
    mesh = rect_plate(1.0, 0.25, nx=n, ny=max(2, n // 4))
    domain = hm.ShellDomain(mesh, element="CG2CG1")
    bcs = hm.clamp(domain, where=lambda x: np.isclose(x[0], 0.0))
    loads = hm.pressure(domain, LOAD)
    iso = hm.isotropic(domain, E=E, nu=NU, thickness=H, density=RHO)
    ply = IsotropicMaterial(name="aluminium", E=E, nu=NU, G=E / (2 * (1 + NU)), density=RHO)
    layup = hm.Layup(ply, np.radians([0.0, 37.0, -62.0, 90.0]), np.full(4, H / 4), num_plies=4)
    lam = hm.laminate(domain, layup=layup, density=RHO, shear_correction=0.833)
    si, sl = hm.solve(domain, iso, loads, bcs), hm.solve(domain, lam, loads, bcs)
    k, _ = node_nearest(domain, [1.0, 0.125, 0.0])
    values_i = np.array([float(hm.compliance(si).value[0]), float(hm.mass(si).value[0]),
                         float(hm.nodal_displacement(si).value.reshape(-1, 3)[k, 2])])
    values_l = np.array([float(hm.compliance(sl).value[0]), float(hm.mass(sl).value[0]),
                         float(hm.nodal_displacement(sl).value.reshape(-1, 3)[k, 2])])
    rel = np.abs(values_l - values_i) / np.maximum(np.abs(values_i), 1.0)
    print("    iso       ", " ".join(f"{x:.6e}" for x in values_i))
    print("    laminate  ", " ".join(f"{x:.6e}" for x in values_l))
    rec.stop()
    return float(rel.max())


CASE = Case(
    name="Isotropic laminate equivalence",
    quantity="maximum relative compliance, mass, or tip-displacement difference",
    reference=0.0, tolerance=1e-8,
    citation="same-run hm.isotropic solution; identical isotropic CLT constitutive law",
    levels=(4, 8), quick_level=4, solve=solve_at, monotone=False,
    notes="laminate shear_correction=0.833 matches hm.isotropic's convention",
)

if __name__ == "__main__":
    main(CASE)
