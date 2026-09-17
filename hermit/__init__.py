"""Hermit -- Reissner-Mindlin shell analysis in FEniCSx, connected to CSDL.

The public API is a ``ShellDomain`` (mesh + state space + index maps) plus free
functions that build its inputs -- ``Field`` builders, ``BoundaryConditions``,
``Geometry``, ``Material`` / ``Orientation``, ``Loads`` -- fed to :func:`solve`,
whose ``ShellState`` the postprocess functions (``compliance`` / ``mass`` /
``stress_field`` / ...) consume. The FEniCSx custom operations live in
``hermit.fenics.ops`` and are called only from ``hermit._solve`` /
``hermit.outputs`` / ``hermit.transfer``.

Note on module names: several implementation modules are underscore-prefixed
(``_solve`` / ``_field`` / ``_geometry`` / ``_laminate``) precisely so they cannot
shadow the public function of the same name. ``import hermit.solve`` would otherwise
rebind ``hermit.solve`` from the function to the module -- silently, with no error.
"""

__version__ = "1.0.0"

from . import _compat as _compat  # noqa: E402  (DOLFINx 0.9 / 0.11 shims)
from . import _ufl_compat as _ufl_compat  # noqa: E402

_ufl_compat.apply()  # UFL mesh-coordinate-derivative fix; see the module docstring

from ._laminate import Layup  # noqa: E402
from .domain import ShellDomain, read_mesh  # noqa: E402
from ._field import (  # noqa: E402
    Field,
    as_field,
    constant,
    field_fn,
    from_cells,
    from_coeffs,
    from_function,
    from_nodal,
)
from .transfer import interpolate, project  # noqa: E402
from .bcs import (  # noqa: E402
    BoundaryConditions,
    clamp,
    gauge,
    near,
    on_plane,
    pin,
    symmetry,
)
from ._geometry import Geometry, geometry  # noqa: E402
from .material import (  # noqa: E402
    Material,
    Orientation,
    composite,
    fiber_angle,
    fiber_direction,
    isotropic,
    laminate,
    thickness_only,
)
from .loads import (  # noqa: E402
    Loads,
    edge_moment,
    edge_pressure,
    edge_traction,
    load_vector,
    moment,
    point_load,
    pressure,
    traction,
)
from ._solve import ShellState, solve  # noqa: E402
from .outputs import (  # noqa: E402
    aggregated_stress,
    center_of_gravity,
    compliance,
    displacement_field,
    elastic_energy,
    failure_field,
    failure_index,
    mass,
    nodal_displacement,
    nodal_rotation,
    pnorm_stress,
    rotation_field,
    strain_fields,
    stress_field,
    stress_scaling,
)

__all__ = [
    "BoundaryConditions",
    "Field",
    "Geometry",
    "Layup",
    "Loads",
    "Material",
    "Orientation",
    "ShellDomain",
    "ShellState",
    "aggregated_stress",
    "as_field",
    "center_of_gravity",
    "clamp",
    "compliance",
    "composite",
    "constant",
    "displacement_field",
    "edge_moment",
    "edge_pressure",
    "edge_traction",
    "elastic_energy",
    "failure_field",
    "failure_index",
    "fiber_angle",
    "fiber_direction",
    "field_fn",
    "from_cells",
    "from_coeffs",
    "from_function",
    "from_nodal",
    "gauge",
    "geometry",
    "interpolate",
    "isotropic",
    "laminate",
    "load_vector",
    "mass",
    "moment",
    "near",
    "nodal_displacement",
    "nodal_rotation",
    "on_plane",
    "pin",
    "pnorm_stress",
    "point_load",
    "pressure",
    "project",
    "read_mesh",
    "rotation_field",
    "solve",
    "strain_fields",
    "stress_field",
    "stress_scaling",
    "symmetry",
    "thickness_only",
    "traction",
]
