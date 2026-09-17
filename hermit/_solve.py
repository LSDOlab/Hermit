"""``solve`` / ``ShellState`` -- the FE solve and the object it returns.

Each load term reaches the residual on its own space, with no interpolation between
terms, and ``pressure`` uses the live shape-differentiable ``ufl.CellNormal``.
Likewise the ABD material fields reach the form on whatever ``constitutive_space``
they were built on.

The ``ShellPDE`` behind a domain is built once and cached on the domain itself, so
every solve and every ``BoundaryConditions`` for that domain shares one state space.
That matters because a strong BC located against a different, merely
structurally-identical ``FunctionSpace`` is silently dropped by dolfinx.
"""

from __future__ import annotations

from types import SimpleNamespace

from .fenics.ops import ShellSolveOp
from .fenics.shell_pde import ShellPDE
from ._geometry import geometry as _geometry

_ORIENT_ARG_NAME = {"angle": "fiber_angle", "direction": "fiber_direction"}


def _pde_for(domain) -> ShellPDE:
    """The ``ShellPDE`` behind a ``ShellDomain``, built once and cached **on the
    domain itself**, sharing ``domain.W``. Every ``solve`` for this ``domain`` (and
    every ``BoundaryConditions`` built against it) therefore shares the *one*
    ``FunctionSpace`` object; a strong BC located against a different, merely
    structurally-identical instance is silently dropped by dolfinx (``max|w|`` blows
    up from 8.7e-3 to 1.7e8, with no exception).

    Reusing this cached PDE across every ``solve`` for the domain is also what makes
    ``ShellSolveOp``'s structural form cache (stored alongside it, in
    ``pde._solve_form_cache``) actually save repeat compile cost for a fixed
    composition.
    """
    pde = getattr(domain, "_pde", None)
    if pde is None:
        pde = ShellPDE(domain.mesh, element=domain.element, W=domain.W, quadrature_degree=domain.quadrature_degree)
        pde._solve_form_cache = {}
        domain._pde = pde
    return pde


class ShellState:
    """The solved shell state, and back-references to everything that produced it.

    Returned by :func:`solve`. Because it carries its own inputs, every output
    function in :mod:`hermit.outputs` needs only the state. A surrogate solve should
    return one of these to be a drop-in replacement.

    Parameters
    ----------
    domain : ShellDomain
    geometry : Geometry or None
        ``None`` is normalised to the reference configuration, so
        ``state.geometry`` is never ``None``.
    material : Material
    loads : Loads
    bcs : BoundaryConditions
    disp_solid : csdl.Variable
        The raw mixed-space dof vector, in ``domain.W`` dof order.
    """

    def __init__(self, domain, geometry, material, loads, bcs, *, disp_solid):
        if geometry is None:
            geometry = _geometry(domain)
        self.domain = domain
        self.geometry = geometry
        self.material = material
        self.loads = loads
        self.bcs = bcs
        self.disp_solid = disp_solid


def _material_terms(material):
    """``({"A": space, ...}, {"A": coeffs, ...})`` from a ``Material``'s ``A``/``B``/
    ``D``/``As`` fields. Raises for a ``thickness_only`` material (no ABD to solve
    with -- surrogate-only, per its own docstring)."""
    names = ("A", "B", "D", "As")
    fields = (material.A, material.B, material.D, material.As)
    if any(f is None for f in fields):
        raise ValueError(
            "hm.solve needs a Material with A/B/D/As (hm.isotropic / hm.laminate / "
            "hm.composite) -- got a thickness_only material, which is surrogate-only "
            "(pass a real Material, or use a surrogate callable instead of hm.solve)."
        )
    spaces = {n: f.space for n, f in zip(names, fields)}
    values = {n: f.coeffs for n, f in zip(names, fields)}
    return spaces, values


def _load_terms(loads):
    """``(arg_names, spec, values)`` for every distributed term in a ``Loads`` --
    one declared arg per term (its own kind, its own space, no combining/interpolation).
    ``spec`` is the ``(argname, kind, space)`` list ``ShellSolveOp`` takes; ``values``
    maps each ``argname`` to its ``coeffs``."""
    arg_names, spec, values = [], [], {}

    def _add(terms, kind):
        for i, f in enumerate(terms):
            name = f"{kind}_{i}"
            arg_names.append(name)
            spec.append((name, kind, f.space))
            values[name] = f.coeffs

    _add(loads.traction_terms, "traction")
    _add(loads.moment_terms, "moment")
    _add(loads.pressure_terms, "pressure")
    # The tagged exterior measure is structural form data; the Field coefficient
    # remains an ordinary differentiable solve argument.
    def _add_edge(terms, kind):
        for i, term in enumerate(terms):
            name = f"edge_{kind}_{i}"
            arg_names.append(name)
            spec.append((name, kind, term.field.space, term.ds, term.facets))
            values[name] = term.field.coeffs

    _add_edge(loads.edge_traction_terms, "traction")
    _add_edge(loads.edge_moment_terms, "moment")
    _add_edge(loads.edge_pressure_terms, "pressure")
    return arg_names, spec, values


def solve(domain, material, loads, bcs, *, geometry=None) -> ShellState:
    """Solve the linear shell problem.

    Parameters
    ----------
    domain : ShellDomain
    material : Material
        Must carry ``A``/``B``/``D``/``As``; a :func:`~hermit.thickness_only`
        material is surrogate-only.
    loads : Loads
    bcs : BoundaryConditions
    geometry : Geometry, optional
        Defaults to the reference configuration. Pass
        ``hm.geometry(domain, node_disp=...)`` to make the mesh coordinates a
        differentiable input.

    Returns
    -------
    ShellState

    Raises
    ------
    ValueError
        If ``material``, ``loads`` or ``bcs`` was built against a different
        ``ShellDomain``, or if the material has no ABD stiffness.

    Notes
    -----
    The formulation is linear and the solve is a direct MUMPS factorization,
    wrapped as a CSDL implicit custom operation. Because the shell tangent is
    symmetric, one factorization serves both the forward solve and the adjoint.

    Every input must be built against this exact ``domain``. DOLFINx silently
    drops a boundary condition located against a structurally identical but
    distinct function space, so this is checked rather than assumed.

    ``solve`` is swappable: any
    ``Callable[(domain, material, loads, bcs), ShellState]`` is a drop-in
    surrogate, and the output functions consume its result unchanged.

    Examples
    --------
    >>> state = hm.solve(domain, material, hm.pressure(domain, 2.0),
    ...                  hm.clamp(domain, where=hm.near("x", 0.0)))
    """
    if bcs.domain is not domain:
        raise ValueError(
            "hm.solve: bcs must be built against this exact ShellDomain -- a "
            "DirichletBC located against a structurally identical but distinct "
            "FunctionSpace is silently dropped. Build the material, loads and bcs "
            "against the same ShellDomain you pass here."
        )
    if material.domain is not domain:
        raise ValueError("hm.solve: material must be built against this exact ShellDomain")
    if loads.domain is not domain:
        raise ValueError("hm.solve: loads must be built against this exact ShellDomain")

    geometry = _geometry(domain) if geometry is None else geometry
    pde = _pde_for(domain)

    mat_spaces, mat_values = _material_terms(material)
    load_arg_names, load_spec, load_values = _load_terms(loads)

    orientation, orient_argname, orient_value = None, None, None
    if material.orientation is not None:
        orient_argname = _ORIENT_ARG_NAME[material.orientation.kind]
        orientation = (orient_argname, material.orientation.value.space)
        orient_value = material.orientation.value.coeffs

    direct = loads.direct_vector()   # never None -- zeros when there's nothing to add

    arg_names = ["A", "B", "D", "As"] + load_arg_names + ["load_vector"]
    if orientation is not None:
        arg_names.append(orient_argname)
    if geometry.is_differentiable:
        arg_names.append("mesh_nodes")

    op = ShellSolveOp(pde, bcs.to_bc_data(), arg_names, options=None,
                      form_cache=pde._solve_form_cache, orientation=orientation,
                      material=mat_spaces, loads=load_spec)

    fe_kwargs = dict(A=mat_values["A"], B=mat_values["B"], D=mat_values["D"],
                     As=mat_values["As"], load_vector=direct, mesh_nodes=geometry.nodes)
    fe_kwargs.update(load_values)
    if orientation is not None:
        fe_kwargs[orient_argname] = orient_value

    disp_solid = op.evaluate(SimpleNamespace(**fe_kwargs))
    return ShellState(domain, geometry, material, loads, bcs, disp_solid=disp_solid)
