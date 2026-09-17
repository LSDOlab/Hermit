"""Space-descriptor plumbing -- normalise / build ``dolfinx`` ``FunctionSpace``\\s from
a compact ``(family, degree[, value_shape])`` tuple, plus the single definition of the
RM-shell mixed state space.

Pure FEniCSx (DOLFINx 0.9 / 0.11), no CSDL. ``_ELEMENTS`` / ``state_space`` used to
live on ``ShellPDE`` directly; they are factored out here so ``ShellPDE`` and the new
``ShellDomain`` both call the same code instead of each building the mixed element by
hand.
"""

import basix.ufl
from dolfinx.fem import functionspace

_ELEMENTS = {
    # name: ((disp family, degree), (rotation family, degree))
    "CG2CG1": (("Lagrange", 2), ("Lagrange", 1)),
    "CG1CG1": (("Lagrange", 1), ("Lagrange", 1)),
    "CG2CR1": (("Lagrange", 2), ("Crouzeix-Raviart", 1)),  # simplex meshes only
}
_SIMPLEX_CELLS = {"interval", "triangle", "tetrahedron"}

_FAMILY_ALIASES = {"CG": "Lagrange"}


def state_space(mesh, element="CG2CG1"):
    """The mixed (displacement, rotation) state space for the R-M shell -- e.g. CG2xCG1
    for ``element="CG2CG1"``. Ported verbatim from ``ShellPDE.__init__``."""
    if element not in _ELEMENTS:
        raise ValueError(f"unsupported element {element!r}; choose from {list(_ELEMENTS)}")
    (du_fam, du_deg), (th_fam, th_deg) = _ELEMENTS[element]

    cell = mesh.basix_cell()
    if th_fam == "Crouzeix-Raviart" and mesh.topology.cell_name() not in _SIMPLEX_CELLS:
        raise ValueError(
            f"element {element!r} (Crouzeix-Raviart rotation) needs a simplex mesh; "
            f"this mesh is {mesh.topology.cell_name()!r} -- use 'CG2CG1' or 'CG1CG1'"
        )
    disp_el = basix.ufl.element(du_fam, cell, du_deg, shape=(3,))
    rot_el = basix.ufl.element(th_fam, cell, th_deg, shape=(3,))
    return functionspace(mesh, basix.ufl.mixed_element([disp_el, rot_el]))


def normalize_space(space):
    """Normalise a space descriptor to ``(family: str, degree: int, value_shape: tuple)``.

    Accepted input: ``("Lagrange", 1)`` / ``("CG", 2)`` / ``("DG", 0)`` / ``("DQ", 1)``
    / ``("Quadrature", 2)``, each optionally with a third element giving the value
    shape, e.g. ``("DG", 0, (3, 3))`` / ``("Lagrange", 1, (3,))``. A scalar space
    normalises to ``value_shape == ()``. ``"CG"`` normalises to ``"Lagrange"`` so
    ``("CG", 1)`` and ``("Lagrange", 1)`` are the same space -- equal tuples, one
    cache entry.
    """
    space = tuple(space)
    if len(space) == 2:
        family, degree = space
        value_shape = ()
    elif len(space) == 3:
        family, degree, value_shape = space
        value_shape = tuple(value_shape)
    else:
        raise ValueError(
            f"space descriptor must be (family, degree[, value_shape]), got {space!r}")
    family = _FAMILY_ALIASES.get(str(family), str(family))
    return (family, int(degree), value_shape)


def make_space(mesh, space):
    """Build the ``dolfinx`` ``FunctionSpace`` for a (possibly un-normalised) space
    descriptor.

    Ordinary families go through the plain ``functionspace`` element tuple.
    ``"Quadrature"`` must go through ``basix.ufl.quadrature_element`` -- it is not a
    valid plain family string for ``functionspace``. Note: any UFL measure that later
    touches a quadrature-space coefficient needs a matching
    ``metadata={"quadrature_degree": d}`` or dolfinx errors; that is the caller's job,
    not this constructor's.
    """
    family, degree, value_shape = normalize_space(space)
    if family == "Quadrature":
        el = basix.ufl.quadrature_element(mesh.basix_cell(), degree=degree,
                                          value_shape=value_shape)
        return functionspace(mesh, el)
    elt = (family, degree) if value_shape == () else (family, degree, value_shape)
    return functionspace(mesh, elt)


def is_discontinuous(space) -> bool:
    """True for DG / DQ / Quadrature spaces -- the single definition, which
    ``hermit.fenics.ops._space_is_dg`` delegates to. ``"Quadrature"`` counts:
    its dofs are per-quadrature-point values, with no continuity across cells.
    """
    family = str(space[0]).upper() if not isinstance(space, str) else space.upper()
    return family in ("DG", "DQ", "QUADRATURE") or "DISCONTINUOUS" in family
