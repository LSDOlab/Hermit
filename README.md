# Hermit

[![Tests](https://github.com/LSDOlab/Hermit/actions/workflows/actions.yml/badge.svg)](https://github.com/LSDOlab/Hermit/actions/workflows/actions.yml)

**Reissner-Mindlin shell analysis in FEniCSx, connected to [CSDL](https://github.com/LSDOlab/CSDL_alpha) for gradient-based design.**

Hermit is a linear-static shell finite-element code built on
[FEniCSx](https://fenicsproject.org/) (DOLFINx 0.9 or 0.11). It wraps the FE solve
and its output functionals as CSDL custom operations, so shell compliance, mass,
centre of gravity, stress, and nodal fields are all differentiable with respect
to thickness, composite ply layup, applied loads, and the mesh coordinates. The mechanics are based on `RMShell` model in
[`femo_alpha`](https://github.com/LSDOlab/femo_alpha).

## Design

Build the domain and inputs, solve for the state, and compute outputs from the state:

```python
import hermit as hm

domain = hm.ShellDomain(mesh, element="CG2CG1")          # mesh + state space, built once
bcs    = hm.clamp(domain, where=root)

mat    = hm.isotropic(domain, E=E, nu=nu, thickness=t, density=rho)
loads  = hm.pressure(domain, p)

state  = hm.solve(domain, mat, loads, bcs)
c, m   = hm.compliance(state), hm.mass(state)
# csdl.Variables -> set_as_objective / _constraint, compute_totals
```

`ShellState` back-references its `domain / geometry / material / loads / bcs`, so every
postprocess function needs only `state`.

| stage | what it is | CSDL custom op? |
|-------|-----------|-----------------|
| `ShellDomain` + `Material` / `Loads` / `BoundaryConditions` / `Geometry` | composable inputs, each a `Field` on whatever space you give it | no — plain CSDL |
| `solve(domain, material, loads, bcs, *, geometry=None)` → `ShellState` | assemble + linear solve + adjoint | **yes** — `ShellSolveOp` (implicit) |
| `compliance(state)`, `strain_fields(state)`, … | output functionals and fields | **yes** — `ShellScalarFormsOp` / `ShellFieldFormsOp` (explicit); nodal fields are plain CSDL |

`Loads` and `BoundaryConditions` compose with `+`. The solve is **swappable**: any
`Callable[(domain, material, loads, bcs), ShellState]` is a drop-in surrogate, and may
consume fewer inputs than the FE solve — build a thickness-only material with
`hm.thickness_only`.

## Capabilities

- **Elements:** mixed CG2×CG1 (displacement × rotation), CG1×CG1, or CG2×CR1. Quad or tri meshes.
- **Fields:** `hm.from_coeffs` / `constant` / `from_nodal` / `from_cells` / `from_function` build
  a `Field` on any FE space; `interpolate` / `project` move between spaces. Every material,
  load and orientation input is a `Field`, and they need not share a space.
- **Materials:** `isotropic`, composite laminate via classical lamination theory
  (`laminate(*, layup, ...)`), pre-computed per-point ABD (`composite`), or
  `thickness_only` for a surrogate. `fiber_direction` /
  `fiber_angle` orient the laminate; the `Tε(θ)` rotation happens **inside the form**, so
  it is differentiable and correct on a continuous orientation field.
- **Loads:** `pressure` (follows the shell normal), `traction`, `moment`, `point_load`,
  or a direct generalized `load_vector`; `edge_pressure` / `edge_traction` /
  `edge_moment` apply the same loads per unit length on exterior facets a `where=`
  predicate selects.
- **Boundary conditions:** `clamp` / `pin` / `symmetry` / `gauge`, penalty (default) or
  strong Dirichlet, on a region predicate; `near` / `on_plane` build the predicates.
  Every builder takes `value=` for a non-zero prescribed displacement or rotation
  (a scalar, a 6-vector, or a callable of the coordinates).
- **Outputs:** `compliance`, `mass`, `center_of_gravity`, `elastic_energy`,
  `aggregated_stress` / `pnorm_stress` (optionally per `region`), `stress_field`,
  `strain_fields`, `displacement_field`, `rotation_field`, `nodal_displacement`,
  `nodal_rotation`, and — for a laminate — `failure_index` / `failure_field` (Tsai-Wu).
- **Derivatives:** reverse-mode through everything, including
  `d(output)/d(mesh_nodes)` — pass `geometry=hm.geometry(domain, node_disp=...)`.

## Installation

Hermit and its FEniCSx / CSDL dependencies are distributed on the `HgXe` conda
channel:

```sh
conda create -n hermit -c HgXe/label/test -c HgXe -c conda-forge hermit
conda activate hermit
```

See `docs/src/getting_started.md` for an editable-development setup.

Hermit runs **serially** — `ShellDomain` requires a mesh whose vertex numbering is a
permutation of the mesh-file order, which an MPI-partitioned mesh is not.

## Example

`examples/basic_examples/ex_cantilever_plate.py` (forward),
`examples/advanced_examples/ex_thickness_opt.py` (thickness optimization with
modopt/SLSQP), `examples/basic_examples/ex_composite_plate.py` (ply-angle sweep).

## Tests

```sh
cd Hermit && python -m pytest
```

The captured `femo_alpha` reference (`tests/data/rmshell_cantilever.npz`) covers the
forward quantities the two codes share --- compliance, mass, CG, elastic energy,
aggregated stress, displacements and rotations --- plus the thickness adjoints of
compliance, mass and aggregated stress. Everything added since (edge and point loads,
ply-angle and shape derivatives, stress and strain fields, Tsai-Wu failure) is
validated against closed-form or published references in
`examples/verification/`, not against femo.

## License

GNU Lesser General Public License v3.0 or later. See `LICENSE.txt`.


## Acknowledgments

The development of this software was supported, in part, by the Air Force Research Laboratory through the Collaborative Center for the Design and Research Of InterDisciplinary Systems (CC DROIDS). Distribution Statement A. Approved for public release: distribution is unlimited. Approved AFRL-2026-1633 17-09-2026. This authorization applies to Git commit ee5a55f0afaa04ad25f651a1f05738c9a4bbd08f.
