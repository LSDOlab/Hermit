# API reference

Auto-generated from docstrings.

The public workflow is `ShellDomain` → material / load / BC builders → `solve` →
state-based output functions.

| stage | functions |
|---|---|
| domain | `ShellDomain`, `read_mesh` |
| fields | `constant`, `from_nodal`, `from_cells`, `from_function`, `from_coeffs`, `as_field`, `field_fn`, `interpolate`, `project` |
| material | `isotropic`, `laminate`, `composite`, `thickness_only`; `fiber_angle`, `fiber_direction`; `Layup` |
| loads | `pressure`, `traction`, `moment`; `edge_pressure`, `edge_traction`, `edge_moment`; `point_load`, `load_vector` |
| boundary conditions | `clamp`, `pin`, `symmetry`, `gauge`; `near`, `on_plane` |
| geometry | `geometry` |
| solve | `solve` → `ShellState` |
| scalar outputs | `compliance`, `mass`, `center_of_gravity`, `elastic_energy`, `pnorm_stress`, `aggregated_stress`, `stress_scaling`, `failure_index` |
| field outputs | `strain_fields`, `stress_field`, `displacement_field`, `rotation_field`, `failure_field`, `nodal_displacement`, `nodal_rotation` |

`Loads` and `BoundaryConditions` compose with `+`.

```{toctree}
:maxdepth: 2

autoapi/hermit/index
```
