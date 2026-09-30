"""STW wingbox thickness optimization

Minimise the compliance of the Simple Transonic Wing (STW) benchmark wingbox at fixed
mass. The same problem as ``ex_thickness_opt.py`` -- compliance objective, equality
mass constraint at the baseline mass, PySLSQP -- on a real wingbox instead of a plate:

* **Mesh**: the benchmark's shell wingbox (upper/lower skins, front/rear spars, 23
  ribs; full semispan including the centre box), read straight from its NASTRAN BDF.
  ``LEVEL`` (or ``--level``) picks the resolution, 4 (coarsest) to 1 (finest):

      level   quads    elements between ribs / spars / skins
        1    71,200     20 / 40 / 20
        2    17,800     10 / 20 / 10
        3     4,450      5 / 10 /  5
        4     1,401      3 /  5 /  3

  The BDF is downloaded once from MDOBenchmarks/MDOAeroelasticBenchmark at a pinned
  revision and cached under ``~/.cache/hermit/stw``; ``--bdf`` points at a local copy.
* **Design variables**: one thickness per panel -- each skin bay, each spar segment
  between adjacent ribs, each rib. These are the BDF's 111 property families (22 +
  22 skin bays, 22 + 22 spar segments, 23 ribs), identical at every level, so the
  design space does not change with the mesh. The thickness field is DG0: the
  panel thicknesses scattered to cells. Piecewise-constant panels are also what
  keeps this well posed, unlike the free nodal field of ``ex_thickness_opt.py``.
* **Loads**: the benchmark's structural load case -- a uniform 30 kPa pressure on
  the lower skin, pushing up into the box, plus the 2.5 g inertial load of the
  structure itself, ``2.5 g * density * thickness`` per unit area, downward. The
  inertial load depends on the thickness design variables, and the compliance
  (work of both loads) accounts for that.
* **Winding**: ``hm.pressure`` acts along the cell normal, so the cells are re-wound
  to point out of the box (skins up/down, spars fore/aft, the root rib inboard,
  every other rib outboard). The BDF's own winding is not consistent;
  ``hm.pressure`` checks this and would refuse the mesh.
* **Boundary conditions**: the benchmark's own SPC cards -- ``uy, rx, rz`` on the
  root-rib perimeter (symmetry plane) and ``ux, uz`` on the side-of-body perimeter,
  where each edge joins three panels (inboard and outboard skin or spar, and the
  side-of-body rib).
* **Material**: isotropic aluminium, as in ``ex_thickness_opt.py``, not the
  benchmark's stiffened composite panels. Baseline thickness is the benchmark's
  6.5 mm panel thickness everywhere.

Design variables, objective and constraint are scaled to O(1) for SLSQP. The
optimization runs on ``csdl.experimental.JaxSimulator``, which compiles the CSDL
graph with JAX; Hermit's FEniCSx operations enter it as callbacks.

**Output**: ``stw_L<level>_design.xdmf`` (``--out``), for ParaView, with the baseline
at time 0 and the optimized design at time 1: ``displacement`` (vertex values,
use *Warp By Vector*), and per cell ``thickness``, ``von_mises_top`` /
``von_mises_bottom`` (Pa) and ``panel`` (design-variable index, 0..110).

    conda activate hermit
    pip install jax                  # or: pip install -e ".[opt]"
    python examples/advanced_examples/ex_stw.py [--level 3] [--out design.xdmf]
"""

import argparse
import inspect
import pathlib
import re
import urllib.request

import basix.ufl
import csdl_alpha as csdl
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import ufl
from dolfinx.mesh import create_mesh
from mpi4py import MPI

import hermit as hm

LEVEL = 4   # mesh resolution, 4 (coarsest) .. 1 (finest)

BENCHMARK_REV = "48e54b4dcc1c6c696ffc6625c01714f3c3c1244e"
BDF_URL = ("https://raw.githubusercontent.com/MDOBenchmarks/MDOAeroelasticBenchmark/"
           "{rev}/STW-Files/struct/wingbox-L{level}-Order2.bdf")
CACHE = pathlib.Path.home() / ".cache" / "hermit" / "stw"

E_VAL, NU_VAL, H_VAL, RHO_VAL = 70.0e9, 0.33, 6.5e-3, 2780.0   # aluminium, SI
PRESSURE = 30.0e3                                             # on the lower skin [Pa]
LOAD_FACTOR, G = 2.5, 9.81                                    # inertial load [g], [m/s^2]
H_MIN, H_MAX = 1.0e-3, 5.0e-2

_DOF_NAMES = {"1": "ux", "2": "uy", "3": "uz", "4": "rx", "5": "ry", "6": "rz"}
# BDF family prefix -> outward (axis, sign); first match wins, so the root rib precedes RIBS.
_OUTWARD = [("U_SKIN", 2, +1), ("L_SKIN", 2, -1), ("SPARS/SPAR.00", 0, -1),
            ("SPARS/SPAR.01", 0, +1), ("RIBS/RIB.00/", 1, -1), ("RIBS", 1, +1)]


def fetch_bdf(level):
    path = CACHE / f"wingbox-L{level}-Order2.bdf"
    if not path.exists():
        url = BDF_URL.format(rev=BENCHMARK_REV, level=level)
        print(f"downloading {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def read_bdf(path):
    """Nodes, CQUAD4 cells, per-cell family index, family names and SPC sets of a BDF.

    Handles the cards pyLayout writes: ``GRID*`` (large field, continued on a ``*``
    line), ``CQUAD4`` and ``SPC`` (small field), and the ``$ Shell element data for
    family <name>`` comment preceding each property's elements. Node and cell
    order is file order.
    """
    lines = pathlib.Path(path).read_text().splitlines()
    grids, quads, pids, spcs, names = {}, [], [], {}, {}
    family = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("GRID*"):
            grids[int(line[8:24])] = (float(line[40:56]), float(line[56:72]),
                                      float(lines[i + 1][8:24]))
            i += 2
            continue
        match = re.search(r"Shell element data for family\s+(\S+)", line)
        if match:
            family = match.group(1)
        elif line.startswith("CQUAD4"):
            pid = int(line[16:24])
            names.setdefault(pid, family)
            pids.append(pid)
            quads.append([int(line[k:k + 8]) for k in (24, 32, 40, 48)])
        elif line.startswith("SPC "):
            spcs.setdefault(line[24:32].strip(), []).append(int(line[16:24]))
        i += 1

    grid_ids = np.array(sorted(grids))
    points = np.array([grids[g] for g in grid_ids])
    index = {g: k for k, g in enumerate(grid_ids)}
    cells = np.vectorize(index.__getitem__)(np.array(quads))
    pid_list = sorted(names)
    family_idx = np.searchsorted(pid_list, pids)
    spc_nodes = {comp: np.array([index[g] for g in nodes]) for comp, nodes in spcs.items()}
    return points, cells, family_idx, [names[p] for p in pid_list], spc_nodes


def build_mesh(points, cells):
    """A linear-quad shell mesh from file-order points and cyclic quads.

    ``create_mesh`` wants basix (tensor-product) vertex order, ``0, 1, 3, 2`` of a
    cyclic quad, and swaps its last two arguments between DOLFINx 0.9 and 0.11.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    cells = np.ascontiguousarray(cells[:, [0, 1, 3, 2]], dtype=np.int64)
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(3,)))
    if list(inspect.signature(create_mesh).parameters)[2] == "e":   # DOLFINx 0.11
        return create_mesh(MPI.COMM_WORLD, cells, domain, points)
    return create_mesh(MPI.COMM_WORLD, cells, points, domain)       # DOLFINx 0.9


def wind_outward(points, cells, family_idx, families):
    """Reverse the cells whose normal points into the box, per ``_OUTWARD``."""
    p = points[cells]
    n = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])      # normal of the cyclic loop
    axis, sign = np.array([next((a, sg) for pre, a, sg in _OUTWARD if f.startswith(pre))
                           for f in families]).T
    flip = sign[family_idx] * n[np.arange(len(cells)), axis[family_idx]] < 0.0
    cells = cells.copy()
    cells[flip] = cells[flip, ::-1]
    return cells


def at_nodes(nodes, tol=1e-8):
    """``where`` predicate selecting the given points; a facet is selected when all of
    its vertices are."""
    tree = cKDTree(nodes)
    return lambda x: tree.query(x.T, distance_upper_bound=tol)[0] <= tol


def snapshot(fields):
    """Current coefficient values of each output field (a copy)."""
    return {name: np.array(field.coeffs.value, dtype=float).ravel()
            for name, field in fields.items()}


def export(path, domain, fields, snapshots):
    """Write ``(time, snapshot)`` pairs of ``fields`` to an XDMF file for ParaView.

    XDMF holds vertex or cell values only, so higher-order Lagrange fields (the CG2
    displacement) are interpolated to CG1 first.
    """
    import dolfinx

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, str(path), "w") as xdmf:
        xdmf.write_mesh(domain.mesh)
        for time, values in snapshots:
            for name, field in fields.items():
                f = dolfinx.fem.Function(domain.function_space(field.space), name=name)
                f.x.array[:] = values[name]
                family, degree, shape = field.space
                if family in ("Lagrange", "CG", "P") and degree > 1:
                    f1 = dolfinx.fem.Function(domain.function_space(("Lagrange", 1, shape)),
                                              name=name)
                    f1.interpolate(f)
                    f = f1
                xdmf.write_function(f, time)


def main(level=LEVEL, bdf=None, out=None):
    points, cells, family_idx, families, spc_nodes = read_bdf(bdf or fetch_bdf(level))
    n_cells, n_panels = len(cells), len(families)
    kind = np.array([f.split("/")[0] for f in families])      # U_SKIN, L_SKIN, SPARS, RIBS
    lower = kind[family_idx] == "L_SKIN"
    cells = wind_outward(points, cells, family_idx, families)
    print(f"STW wingbox L{level}: {len(points)} nodes, {n_cells} quads, {n_panels} panels")

    rec = csdl.Recorder(inline=True)
    rec.start()

    domain = hm.ShellDomain(build_mesh(points, cells), element="CG2CG1")
    scatter = sp.csr_matrix((np.ones(n_cells), (np.arange(n_cells), family_idx)),
                            shape=(n_cells, n_panels))
    thickness = csdl.Variable(value=H_VAL * np.ones(n_panels), name="thickness")
    cell_thickness = hm.from_cells(domain, csdl.sparse.matvec(scatter, thickness))
    material = hm.isotropic(domain, E=E_VAL, nu=NU_VAL, density=RHO_VAL,
                            thickness=cell_thickness)

    # the outward normal of the lower skin points down, so pushing up is negative p
    pressure = hm.pressure(domain, hm.from_cells(domain, np.where(lower, -PRESSURE, 0.0)))
    # per-cell (0, 0, -n g rho t), flattened row-major, straight from the panel thicknesses
    weight = sp.kron(scatter, np.array([[0.0], [0.0], [-LOAD_FACTOR * G * RHO_VAL]]))
    inertial = hm.traction(domain, hm.from_cells(
        domain, csdl.sparse.matvec(weight.tocsr(), thickness).reshape((n_cells, 3))))
    loads = pressure + inertial

    pins = [hm.pin(domain, where=at_nodes(points[nodes]), dofs=[_DOF_NAMES[c] for c in comp])
            for comp, nodes in spc_nodes.items()]
    bcs = pins[0]
    for bc in pins[1:]:
        bcs = bcs + bc
    state = hm.solve(domain, material, loads, bcs)
    compliance, mass = hm.compliance(state), hm.mass(state)

    compliance_0 = float(np.ravel(compliance.value)[0])
    mass_0 = float(np.ravel(mass.value)[0])
    print(f"baseline: compliance={compliance_0:.6e}  mass={mass_0:.6e}")

    fields = {
        "displacement": hm.displacement_field(state),
        "thickness": cell_thickness,
        "von_mises_top": hm.stress_field(state, space=("DG", 0), method="average", surface="top"),
        "von_mises_bottom": hm.stress_field(state, space=("DG", 0), method="average",
                                            surface="bottom"),
        "panel": hm.from_cells(domain, family_idx.astype(float)),
    }
    snapshots = [(0.0, snapshot(fields))]

    thickness.set_as_design_variable(lower=H_MIN, upper=H_MAX, scaler=1.0 / H_VAL)
    mass.set_as_constraint(equals=mass_0, scaler=1.0 / mass_0)
    compliance.set_as_objective(scaler=1.0 / compliance_0)

    from modopt import CSDLAlphaProblem, PySLSQP

    sim = csdl.experimental.JaxSimulator(rec, gpu=False)
    prob = CSDLAlphaProblem(problem_name=f"hermit_stw_L{level}_thickness", simulator=sim)
    optimizer = PySLSQP(prob, solver_options={"maxiter": 200, "acc": 1e-9})
    optimizer.solve()
    optimizer.print_results()

    # After a JAX run only the simulator's own values are reliably current (callbacks
    # overwrite some graph values, the compiled graph none of the rest): copy the optimum
    # back and re-run the graph inline to refresh every variable for the export.
    thickness.value = sim[thickness]
    rec.execute()
    rec.stop()
    snapshots.append((1.0, snapshot(fields)))
    out = pathlib.Path(out or f"stw_L{level}_design.xdmf")
    export(out, domain, fields, snapshots)

    t = thickness.value
    print("optimized:")
    print(f"  compliance : {float(np.ravel(compliance.value)[0]):.6e}  (baseline {compliance_0:.6e})")
    print(f"  mass       : {float(np.ravel(mass.value)[0]):.6e}  (target {mass_0:.6e})")
    for k in ("U_SKIN", "L_SKIN", "SPARS", "RIBS"):
        tk = 1e3 * t[kind == k]
        print(f"  {k:<7}thickness [mm]: min {tk.min():.2f}  max {tk.max():.2f}")
    print(f"wrote {out} (+ .h5): baseline at t=0, optimized at t=1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4), default=LEVEL)
    parser.add_argument("--bdf", type=pathlib.Path, help="local BDF instead of the download")
    parser.add_argument("--out", type=pathlib.Path,
                        help="XDMF output (default stw_L<level>_design.xdmf)")
    args = parser.parse_args()
    main(args.level, args.bdf, args.out)
