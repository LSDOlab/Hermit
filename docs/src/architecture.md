# Architecture

Hermit separates reusable finite-element setup from differentiable problem data.
`ShellDomain` owns the mesh, the mixed shell space and the ordering maps, and caches
the compiled `ShellPDE` on first solve; `Field` is the common representation for
material, load, geometry, and output fields.

The mesh must be **serial**: `ShellDomain` requires its vertex numbering to be a
permutation of the mesh-file order, and raises on an MPI-partitioned mesh. This
applies to every use, not only to shape derivatives.

```python
domain = hm.ShellDomain(mesh, element="CG2CG1")
bcs = hm.clamp(domain, where=root)
mat = hm.isotropic(domain, E=E, nu=nu, thickness=t, density=rho)
state = hm.solve(domain, mat, hm.pressure(domain, pz), bcs)
compliance = hm.compliance(state)
```

The domain and boundary conditions are constructed once and reused across cases.
`solve` returns a `ShellState` that retains its domain, geometry, material, loads, and
boundary conditions, so each output requires only that state. The FEniCSx assembly and
linear/adjoint solves remain behind custom operations; CSDL sees differentiable inputs
and outputs.

## Fields and ordering

`Field.coeffs` are in FE-dof order. `hm.from_coeffs` accepts that order directly, while
`hm.from_nodal` and `hm.from_cells` accept mesh-file vertex and cell order respectively.
Use `domain.dof_coords(space)` with `hm.from_function` or `hm.from_coeffs` when constructing
coefficients on other spaces. Input fields can use independent spaces; no domain-wide
material or load space exists.

`hm.interpolate` transfers between spaces on the same mesh by reference-element
collocation. `hm.project` is the geometry-dependent L2 alternative.

## Materials, loads, and boundary conditions

`hm.isotropic`, `hm.laminate`, and `hm.composite` construct a `Material`. Composite
ABD matrices remain in laminate axes and an `Orientation` from `hm.fiber_angle` or
`hm.fiber_direction` is consumed inside the shell form.

`hm.pressure`, `hm.traction`, `hm.moment`, `hm.point_load`, and `hm.load_vector` each
return composable `Loads`; `hm.edge_pressure`, `hm.edge_traction`, and
`hm.edge_moment` apply the same loads per unit length on exterior facets selected by
a `where=` predicate. Every load term reaches the residual and the compliance form on
its own space, so mixing spaces costs no interpolation.

Use `hm.clamp`, `hm.pin`, `hm.symmetry`, and `hm.gauge` for boundary conditions. Each
takes `value=` for a non-zero prescribed displacement or rotation, and `+` merges
them with later terms winning on overlapping entities. A prescribed value is
interpolated once rather than carried as a solve input, so unlike a load it cannot be
a `csdl.Variable`. See {doc}`background`.

## Outputs and frames

Scalar outputs are individual functions: `hm.compliance`, `hm.mass`,
`hm.center_of_gravity`, `hm.elastic_energy`, stress aggregates, and composite
`hm.failure_index`. `hm.strain_fields(state, space=..., method=..., frame=...)`
returns membrane, bending, and shear `Field`s; `hm.displacement_field` and
`hm.rotation_field` return state fields.

A DG strain field defaults to an element-local frame, while a CG field defaults to a
global Cartesian frame. Use `to_global()` or `to_frame(...)` to change representation.
Local-frame components require DG because neighbouring element frames need not agree;
global-frame components can be continuous.

## Geometry and surrogates

The reference geometry is used by default. Pass
`geometry=hm.geometry(domain, node_disp=shape)` (or `nodes=...`) to make geometry a
differentiable input. A surrogate can use the same arguments as `hm.solve` and return a
`ShellState`; state-based output functions can then consume that result.
