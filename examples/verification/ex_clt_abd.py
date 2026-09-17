"""Classical-lamination-theory ABD assembly

This is an algebraic verification of Hermit's laminate preprocessing, rather than a
finite-element benchmark.  It independently forms each ply's plane-stress reduced
stiffness and transformed stiffness, then sums the classical ``A``, ``B`` and ``D``
integrals through the thickness.  Both an unsymmetric stack (which has nonzero
extension--bending coupling ``B``) and a symmetric stack are checked.  The reference
is the NumPy calculation below; ``hermit._laminate`` is intentionally not used to
form it.
"""

import pathlib
import sys

import numpy as np
import csdl_alpha as csdl
from caddee_materials import TransverseMaterial

import hermit as hm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _geometry import rect_plate  # noqa: E402
from _harness import Case, main  # noqa: E402


E1, E2, NU12, G12 = 138e9, 10e9, 0.31, 7e9
STACKS = {"unsymmetric [0/45/90]": (0.0, 45.0, 90.0),
          "symmetric [0/45/90]s": (0.0, 45.0, 90.0, 90.0, 45.0, 0.0)}


def qbar(theta):
    """Plane-stress ``Qbar`` from first principles in engineering Voigt order."""
    nu21 = NU12 * E2 / E1
    d = 1.0 - NU12 * nu21
    q = np.array([[E1 / d, NU12 * E2 / d, 0.0],
                  [NU12 * E2 / d, E2 / d, 0.0],
                  [0.0, 0.0, G12]])
    c, s = np.cos(theta), np.sin(theta)
    teps = np.array([[c * c, s * s, s * c],
                     [s * s, c * c, -s * c],
                     [-2 * s * c, 2 * s * c, c * c - s * s]])
    tsig_inv = np.array([[c * c, s * s, -2 * s * c],
                         [s * s, c * c, 2 * s * c],
                         [s * c, -s * c, c * c - s * s]])
    return tsig_inv @ q @ teps


def hand_abd(angles, h=0.006):
    """Direct definitions of the CLT thickness integrals, bottom ply first."""
    z = np.linspace(-h / 2, h / 2, len(angles) + 1)
    a = np.zeros((3, 3)); b = np.zeros((3, 3)); d = np.zeros((3, 3))
    for theta, z0, z1 in zip(np.radians(angles), z[:-1], z[1:]):
        qb = qbar(theta)
        a += qb * (z1 - z0)
        b += qb * (z1**2 - z0**2) / 2
        d += qb * (z1**3 - z0**3) / 3
    return a, b, d


def solve_at(_):
    """Maximum normalized difference of Hermit's and independently assembled ABD."""
    rec = csdl.Recorder(inline=True); rec.start()
    domain = hm.ShellDomain(rect_plate(1.0, 1.0, nx=1, ny=1))
    ply = TransverseMaterial(name="ud", EA=E1, ET=E2, vA=NU12, vT=0.4,
                              GA=G12, density=1600.0)
    errors = []
    for name, angles in STACKS.items():
        layup = hm.Layup(ply, np.radians(angles), np.full(len(angles), 0.006 / len(angles)),
                          num_plies=len(angles))
        material = hm.laminate(domain, layup=layup, density=1600.0)
        want = hand_abd(angles)
        got = tuple(np.asarray(field.cell_values().value).reshape(3, 3) for field in
                    (material.A, material.B, material.D))
        errors.extend(np.max(np.abs(x - y)) / np.max(np.abs(y)) for x, y in zip(got, want))
        print(f"    {name:24s}  ||B||/||A||={np.linalg.norm(want[1]) / np.linalg.norm(want[0]):.3e}")
    rec.stop()
    return max(errors)


CASE = Case(
    name="CLT ABD assembly",
    quantity="maximum normalized A/B/D assembly discrepancy",
    reference=0.0,
    tolerance=1e-10,
    citation="independent NumPy evaluation of the CLT thickness integrals in this file",
    levels=(1,), quick_level=1, solve=solve_at, monotone=False,
    notes="one algebraic level; the unsymmetric stack deliberately exercises B",
)

if __name__ == "__main__":
    main(CASE)
