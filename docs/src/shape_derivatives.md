# Shape (mesh-coordinate) derivatives

Hermit differentiates the shell solve and every output functional with respect to the
mesh node coordinates. This is the sensitivity a shape optimizer, or the chain rule
back to a geometry parameterisation, consumes.

## Using it

Build a geometry from a `csdl.Variable` as either:

- `node_disp=` --- a displacement field added to the reference coordinates
  (`nodes = reference + node_disp`), or
- `nodes=` --- absolute node coordinates.

```python
nd = csdl.Variable(value=np.zeros((domain.n_nodes, 3)), name="node_disp")
state = hm.solve(domain, material, loads, bcs,
                 geometry=hm.geometry(domain, node_disp=nd))
compliance = hm.compliance(state)

dC_dX = sim.compute_totals([compliance], [nd])[compliance, nd].reshape(domain.n_nodes, 3)
```

`dC_dX[k]` is $\partial(\text{compliance})/\partial\mathbf{x}_k$ at mesh node $k$, in
your node ordering.

Without a live `node_disp` / `nodes` variable the geometry is treated as constant and
no mesh-derivative forms are built --- there is no cost to leaving it out.

## How it works

The residual and output forms are differentiated with respect to
`ufl.SpatialCoordinate` (a UFL `CoordinateDerivative`). The resulting sensitivity
lives on the mesh coordinate space; Hermit scatters it back to the user's node
ordering via `mesh.geometry.input_global_indices`.

`mesh.geometry.x` is used as a scratch buffer during assembly and restored afterwards,
so a solve never leaves the mesh moved for a later `ShellDomain` solve.

## Requirements and caveats

- **Serial mesh** whose `input_global_indices` form a node permutation (true for a
  normally-read serial mesh). This is not specific to shape derivatives ---
  `ShellDomain` enforces it for every use, so an MPI-partitioned mesh raises before
  any solve.
- **UFL patch.** UFL's `CoordinateDerivative` handler crashes on the shell bending
  term (`grad(cross(CellNormal, theta))`) --- in both UFL 2024.2 (DOLFINx 0.9) and
  2026.1 (DOLFINx 0.11). Hermit applies a small runtime monkeypatch
  (`hermit._ufl_compat`) on `import hermit`, adapting to whichever UFL ruleset API is
  present; the installed UFL files are untouched.
- **Non-smooth stabilization scale.** UFL drops the (non-differentiable) subgradient
  of `CellDiameter`, which appears only in the drilling / penalty *stabilization
  scale*. It is a stabilization term rather than a physical one, so the effect on
  outputs is small, but the size has not been quantified against a benchmark.
- **Penalty conditioning.** With a very large geometry perturbation the
  penalty-BC system becomes ill-conditioned and finite-difference checks stop
  converging --- a conditioning limit, not a bug. Use the strong-BC path or a smaller
  step for verification.

The mesh-coordinate adjoints are finite-difference validated in
`tests/test_mesh_coord_deriv.py`, which measures a few parts in $10^{5}$ relative
error and gates at $3\times10^{-3}$; the `ex_shape_derivative_fd.py` benchmark gates
at $5\times10^{-3}$. Those gates are set by finite-difference truncation and
cancellation, not by the adjoint.
