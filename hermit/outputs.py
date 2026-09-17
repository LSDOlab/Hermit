"""Postprocess a :class:`hermit.ShellState` into scalar and field outputs.

Every function here takes the solved state and nothing else -- the state
back-references its domain, geometry, material, loads and boundary conditions.
Scalar outputs return ``csdl.Variable``\\s; field outputs return
:class:`hermit.Field`\\s. All of them are differentiable in reverse mode, including
with respect to the mesh coordinates when the solve was given a live geometry.
"""

from types import SimpleNamespace

import numpy as np
import csdl_alpha as csdl
import ufl

from . import csdl_helpers as H
from .failure import failure_field as _failure_field
from .failure import failure_index as _failure_index
from .fenics.ops import FieldSpec, ShellFieldFormsOp, ShellScalarFormsOp, _share_edge_and_penalty_ds
from .fenics.bcs import BCData
from .fenics.ops import strain_fields as _strain_specs
from .fenics.spaces import _ELEMENTS
from ._solve import _pde_for


def _pde(state):
    return _pde_for(state.domain)


def _geometry_is_live(state):
    return state.geometry.is_differentiable


def _state_values(state, **values):
    values.setdefault("disp_solid", state.disp_solid)
    values.setdefault("mesh_nodes", state.geometry.nodes)
    return SimpleNamespace(**values)


def _scalar(state, form, args, values, coefficients=None, orientation=None):
    """One scalar form, one ``ShellScalarFormsOp`` -- the deliberate API choice."""
    op = ShellScalarFormsOp(_pde(state), {"value": (form, tuple(args))},
                            differentiable_geometry=_geometry_is_live(state),
                            coefficients=coefficients, orientation=orientation)
    return op.evaluate(_state_values(state, **values)).value


def _material_coefficients(state, *names):
    material = state.material
    fields = {name: getattr(material, name) for name in names}
    missing = [name for name, field in fields.items() if field is None]
    if missing:
        raise ValueError(f"output needs material fields {missing}")
    return {name: field.space for name, field in fields.items()}, \
        {name: field.coeffs for name, field in fields.items()}


def _orientation(state):
    orient = state.material.orientation
    if orient is None:
        return None, {}, {}
    name = "fiber_" + orient.kind
    return (name, orient.value.space), {name: orient.value.coeffs}, {name: orient.value.space}


def _stress_material(state):
    material = state.material
    if material.E is None or material.nu is None:
        raise ValueError(
            "stress outputs are isotropic-only (the material has no E and nu); "
            "pass E= and nu= to hm.composite, or use failure_index for laminate "
            "failure recovery."
        )
    return _material_coefficients(state, "thickness", "E", "nu")


def _stress_measure(state, region, quadrature_degree=None):
    domain = state.domain
    qdeg = domain.quadrature_degree if quadrature_degree is None else quadrature_degree
    md = {"quadrature_degree": qdeg}
    if region is None:
        return ufl.Measure("dx", domain=domain.mesh, metadata=md)
    if domain.cell_tags is None:
        raise ValueError("region= needs ShellDomain(cell_tags=..., regions=...)")
    tag = domain.regions.get(region, region) if isinstance(region, str) else region
    if tag not in set(domain.cell_tags.values):
        raise KeyError(f"region {region!r} (tag {tag}) not in the mesh tags")
    return ufl.Measure("dx", domain=domain.mesh, subdomain_data=domain.cell_tags,
                       metadata=md)(tag)


def compliance(state):
    """Work done by the applied loads on the solution.

    Parameters
    ----------
    state : ShellState

    Returns
    -------
    csdl.Variable
        Scalar. Conjugate to the exact ``Loads`` object that was solved, including
        edge and point terms.
    """
    pde = _pde(state)
    args, values, coefficients, terms = ["disp_solid"], {}, {}, []
    for kind, fields in (("traction", state.loads.traction_terms),
                         ("moment", state.loads.moment_terms),
                         ("pressure", state.loads.pressure_terms)):
        for i, field in enumerate(fields):
            name = f"{kind}_{i}"
            args.append(name); values[name] = field.coeffs; coefficients[name] = field.space
            terms.append((kind, pde.coefficient(name, field.space)))
    edge_specs, edge_fields = [], {}
    for kind, edge_terms in (("traction", state.loads.edge_traction_terms),
                             ("moment", state.loads.edge_moment_terms),
                             ("pressure", state.loads.edge_pressure_terms)):
        for i, edge in enumerate(edge_terms):
            name = f"edge_{kind}_{i}"
            edge_specs.append((name, kind, edge.field.space, edge.ds, edge.facets))
            edge_fields[name] = edge.field
    # A compliance form has no BC terms, but several edge loads still need one
    # shared tagged ds object (the same DOLFINx constraint as the residual).
    _, edge_specs = _share_edge_and_penalty_ds(pde, BCData(penalty=False), edge_specs)
    for name, kind, space, ds, _ in edge_specs:
        edge = edge_fields[name]
        args.append(name); values[name] = edge.coeffs; coefficients[name] = edge.space
        terms.append((kind, pde.coefficient(name, edge.space), ds))
    form = pde.compliance_form(loads=terms)
    value = _scalar(state, form, args, values, coefficients)
    return value + csdl.vdot(state.loads.direct_vector(), state.disp_solid)


def mass(state):
    """Structural mass, ``int density * thickness dx``.

    Parameters
    ----------
    state : ShellState

    Returns
    -------
    csdl.Variable
        Scalar.
    """
    pde = _pde(state)
    coeffs, values = _material_coefficients(state, "thickness", "density")
    return _scalar(state, pde.mass_form(pde.coefficient("thickness", coeffs["thickness"]),
                                        pde.coefficient("density", coeffs["density"])),
                   ("thickness", "density"), values, coeffs)


def center_of_gravity(state):
    """Mass-weighted centre of gravity, in global Cartesian coordinates.

    Parameters
    ----------
    state : ShellState

    Returns
    -------
    csdl.Variable
        Shape ``(3,)``.
    """
    pde = _pde(state)
    coeffs, values = _material_coefficients(state, "thickness", "density")
    forms = pde.cg_forms(pde.coefficient("thickness", coeffs["thickness"]),
                         pde.coefficient("density", coeffs["density"]))
    raw = [_scalar(state, form, ("thickness", "density"), values, coeffs) for form in forms]
    return csdl.concatenate(tuple(raw[:3])) / raw[3]


def elastic_energy(state):
    """Stored shell strain energy.

    Parameters
    ----------
    state : ShellState

    Returns
    -------
    csdl.Variable
        Scalar. For a linear solve this is half the compliance, up to the
        round-off of the direct solve.

    Raises
    ------
    ValueError
        If the material carries no ``A``/``B``/``D``/``As`` (a
        :func:`~hermit.thickness_only` material).
    """
    pde = _pde(state)
    coeffs, values = _material_coefficients(state, "A", "B", "D", "As")
    orient, orient_values, orient_coeffs = _orientation(state)
    values.update(orient_values); coeffs.update(orient_coeffs)
    material = tuple(pde.coefficient(name, coeffs[name]) for name in ("A", "B", "D", "As"))
    orientation = None if orient is None else (orient[0], pde.coefficient(*orient))
    form = pde.elastic_energy_form(orientation=orientation, material=material)
    return _scalar(state, form, ("disp_solid", *values), values, coeffs, orient)


def pnorm_stress(state, *, rho=100, m=1e-6, region=None, surface="top", quadrature_degree=None):
    """Area-normalized p-norm stress integral, ``(1/A) int (m*vm)**rho dx``.

    Parameters
    ----------
    state : ShellState
    rho : float, optional
        Aggregation exponent. Default 100.
    m : float, optional
        Stress scaling; see Notes. Default 1e-6.
    region : str or int, optional
        Restrict the integral to a mesh-tagged region. Needs a domain built with
        ``cell_tags=`` and ``regions=``.
    surface : {'top', 'bottom', 'mid'}, optional
        Through-thickness station the stress is recovered at. Default ``'top'``.
    quadrature_degree : int, optional
        Overrides the domain's degree for this integral.

    Returns
    -------
    csdl.Variable
        Scalar.

    Raises
    ------
    ValueError
        If the material carries no ``E`` and ``nu`` (stress recovery is
        isotropic-only -- use :func:`failure_index` for laminates), or if
        ``region=`` is given without mesh tags.
    KeyError
        If ``region`` is not among the mesh tags.

    Notes
    -----
    ``m`` is a problem-specific scaling, not a smoothing knob. It enters as
    ``(m*vm)**rho``, so it must put ``m*max(vm)`` near 1 or the integrand runs off
    the end of double precision: with the default ``m`` and a peak von Mises of
    1.4e4, ``(m*vm)**100`` is 1e-186. Use :func:`stress_scaling` to compute one.

    ``m`` must also be **constant** with respect to the design variables. It is not
    differentiated and cancels out of the aggregate exactly; recomputing it inside
    the recorded graph each optimizer iteration silently makes the reported
    derivative wrong.

    See Also
    --------
    aggregated_stress : the same quantity reduced back to stress units.
    """
    pde = _pde(state)
    coeffs, values = _stress_material(state)
    form = pde.pnorm_stress_form(
        _stress_measure(state, region, quadrature_degree=quadrature_degree), rho=rho, m=m, surface=surface,
        thickness=pde.coefficient("thickness", coeffs["thickness"]),
        E=pde.coefficient("E", coeffs["E"]), nu=pde.coefficient("nu", coeffs["nu"]),
    )
    return _scalar(state, form, ("disp_solid", "thickness", "E", "nu"), values, coeffs)


def aggregated_stress(state, *, rho=100, m=1e-6, region=None, surface="top", quadrature_degree=None):
    """Smooth aggregate of isotropic von Mises stress, ``(1/m) * pnorm**(1/rho)``.

    A differentiable stand-in for the peak stress, suitable as an optimizer
    constraint.

    Parameters
    ----------
    state : ShellState
    rho : float, optional
        Aggregation exponent. Default 100.
    m : float, optional
        Stress scaling; see :func:`pnorm_stress`. Default 1e-6.
    region : str or int, optional
        Restrict to a mesh-tagged region.
    surface : {'top', 'bottom', 'mid'}, optional
        Through-thickness station. Default ``'top'``.
    quadrature_degree : int, optional
        Overrides the domain's degree for this integral.

    Returns
    -------
    csdl.Variable
        Scalar, in stress units.

    Warns
    -----
    UserWarning
        If ``m`` is degenerate for this problem, which would otherwise return a
        number that barely depends on the solution.

    Notes
    -----
    Approaches ``max(vm)`` **from below** as ``rho`` grows: this is the area-averaged
    integral p-norm, bounded above by the pointwise maximum, unlike the discrete
    log-sum-exp aggregate :func:`failure_index` uses.

    Examples
    --------
    >>> m = hm.stress_scaling(state)
    >>> peak = hm.aggregated_stress(state, m=m)
    """
    return H.aggregate(pnorm_stress(state, rho=rho, m=m, region=region, surface=surface, quadrature_degree=quadrature_degree), m, rho)


def stress_scaling(state, *, surface="top", space=("DG", 2), method="project"):
    """Suggest the stress scaling ``m = 1 / max|von Mises|`` for the aggregates.

    Parameters
    ----------
    state : ShellState
    surface : {'top', 'bottom', 'mid'}, optional
        Through-thickness station. Default ``'top'``.
    space : tuple, optional
        Space the stress field is recovered on. Default ``("DG", 2)``.
    method : {'project', 'interpolate', 'average'}, optional
        Field recovery method. Default ``'project'``.

    Returns
    -------
    float
        A plain Python float, deliberately: it is a snapshot taken outside the
        differentiated path, so it cannot contaminate the derivative of
        :func:`aggregated_stress`.

    Raises
    ------
    ValueError
        If the stress field has no value (run under an inline ``csdl.Recorder``),
        or if the peak is not finite and positive.

    Notes
    -----
    Costs one extra stress-field assembly. Call it once when setting a problem up,
    not inside an optimizer loop -- a design-dependent ``m`` makes the reported
    gradient wrong even though the value looks better scaled.
    """
    coeffs = stress_field(state, space=space, method=method, surface=surface).coeffs.value
    if coeffs is None:
        raise ValueError("stress_scaling needs the stress field evaluated: run it under "
                         "an inline csdl.Recorder")
    peak = float(np.abs(np.asarray(coeffs)).max())
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError(f"cannot scale to a peak von Mises stress of {peak}")
    return 1.0 / peak


def _field_values(state, fields, values=None, coefficients=None):
    op = ShellFieldFormsOp(_pde(state), fields, differentiable_geometry=_geometry_is_live(state),
                           coefficients=coefficients, quadrature_degree=state.domain.quadrature_degree)
    return op, op.evaluate(_state_values(state, **(values or {})))


def _field(state, space, coeffs, *, kind="scalar", frame=None, global_frame=False):
    from ._field import Field
    return Field(state.domain, space, csdl.reshape(coeffs, (coeffs.size,)), kind=kind,
                 frame=frame, global_frame=global_frame)


def strain_fields(state, *, space=("DG", 2), method="project", frame=None):
    """Membrane strain, bending curvature and transverse shear.

    Parameters
    ----------
    state : ShellState
    space : tuple, optional
        Target space for all three fields. Default ``("DG", 2)``.
    method : {'project', 'interpolate', 'average'}, optional
        Recovery method. ``'average'`` is the DG0 cell average. Default
        ``'project'``.
    frame : {'local', 'global'}, optional
        Component frame. Defaults to the element-local in-plane frame on a DG
        space, and to global Cartesian on a CG space, where per-cell frames would
        be ambiguous at shared nodes.

    Returns
    -------
    tuple of Field
        ``(mid_strain, curvature, shear_strain)``. The component layout follows the
        frame: in the element-local frame the first two are in-plane engineering
        Voigt ``[xx, yy, 2xy]`` and the third is ``[xz, yz]``; in the global frame
        they become full symmetric tensors, ``[xx, yy, zz, 2yz, 2xz, 2xy]`` and a
        3-vector.

    Raises
    ------
    ValueError
        If ``method`` or ``frame`` is not one of the accepted values, or if a
        local frame is requested on a continuous (CG) space.

    See Also
    --------
    hermit.Field.to_frame, hermit.Field.to_global : re-express the components.

    Examples
    --------
    >>> eps, kappa, gamma = hm.strain_fields(state)
    >>> kappa_global = kappa.to_global().values
    """
    specs = _strain_specs(_pde(state), space=space, method=method, frame=frame)
    op, raw = _field_values(state, specs)
    out = []
    for name in ("mid_strain", "curvature", "shear_strain"):
        spec = specs[name]
        out.append(_field(state, (*spec.space, (spec.n_components,)), getattr(raw, name),
                          kind=spec.kind,
                          frame=None if spec.frame == "global" else state.domain.local_frames(),
                          global_frame=spec.frame == "global"))
    return tuple(out)


def stress_field(state, *, space=("DG", 2), method="project", surface="top"):
    """Isotropic von Mises stress field.

    Parameters
    ----------
    state : ShellState
    space : tuple, optional
        Target space. Default ``("DG", 2)``.
    method : {'project', 'interpolate', 'average'}, optional
        Recovery method. Default ``'project'``.
    surface : {'top', 'bottom', 'mid'}, optional
        Through-thickness station. Default ``'top'``.

    Returns
    -------
    Field
        Scalar field.

    Raises
    ------
    ValueError
        If the material carries no ``E`` and ``nu``; von Mises recovery is
        isotropic-only. :func:`~hermit.composite` accepts ``E=`` / ``nu=`` for this.
    """
    pde = _pde(state)
    coeffs, values = _stress_material(state)
    expr = pde.von_mises_form(
        surface=surface, thickness=pde.coefficient("thickness", coeffs["thickness"]),
        E=pde.coefficient("E", coeffs["E"]), nu=pde.coefficient("nu", coeffs["nu"]),
    )
    fields = {"stress": FieldSpec(expr, 1, ("disp_solid", "thickness", "E", "nu"),
                                    space=tuple(space), method=method)}
    op = ShellFieldFormsOp(pde, fields, differentiable_geometry=_geometry_is_live(state),
                           coefficients=coeffs, quadrature_degree=state.domain.quadrature_degree)
    raw = op.evaluate(_state_values(state, thickness=values["thickness"], E=values["E"],
                                    nu=values["nu"]))
    return _field(state, tuple(space), raw.stress)


def _match_rows(a, b):
    """``perm`` with ``b[i] == a[perm[i]]`` for two orderings of the same point set."""
    ka, kb = np.lexsort(np.round(a, 10).T), np.lexsort(np.round(b, 10).T)
    perm = np.empty(len(b), dtype=np.int64)
    perm[kb] = ka
    return perm


def _subfield(state, sub):
    family, degree = _ELEMENTS[state.domain.element][sub]
    space = (family, degree, (3,))
    Vsub, dofs = state.domain.W.sub(sub).collapse()
    # A collapsed sub-space's dof order is NOT the order of a freshly built space of
    # the same element -- measured on a 4x20 quad plate, 349 of 369 CG2 blocks and 102
    # of 105 CG1 blocks move. ``Field`` tabulates against the canonical (freshly built)
    # space, so the coefficients have to be permuted into it or every consumer that
    # reads ``.coeffs`` by dof gets a scrambled field.
    perm = _match_rows(Vsub.tabulate_dof_coordinates(), state.domain.dof_coords(space))
    blocks = np.asarray(dofs).reshape(-1, 3)[perm].ravel()
    return _field(state, space, state.disp_solid[list(blocks)], kind="vector3",
                  global_frame=True)


def displacement_field(state):
    """Mid-surface displacement as a field.

    Parameters
    ----------
    state : ShellState

    Returns
    -------
    Field
        ``(3,)`` vector field on the displacement subspace of the element, in
        global Cartesian components.

    See Also
    --------
    nodal_displacement : the same quantity at mesh vertices, in file order.
    """
    return _subfield(state, 0)


def rotation_field(state):
    """Director rotation as a field.

    Parameters
    ----------
    state : ShellState

    Returns
    -------
    Field
        ``(3,)`` vector field on the rotation subspace of the element, in global
        Cartesian components.
    """
    return _subfield(state, 1)


def failure_field(state, *, sampling="average"):
    """Per-cell, per-ply-face Tsai-Wu failure indices.

    Parameters
    ----------
    state : ShellState
    sampling : {'average', 'midpoint'}, optional
        How the DG0 strain measures are sampled per cell. Default ``'average'``.

    Returns
    -------
    Field
        DG0 field with one component per ply face.

    Raises
    ------
    ValueError
        If the material carries no layup. Only :func:`~hermit.laminate` sets one.
    """
    if state.material.layup is None:
        raise ValueError("failure_field needs a composite ply layup, which only "
                         "hm.laminate carries; hm.composite takes ABD matrices and "
                         "cannot recover per-ply failure")
    specs = _strain_specs(_pde(state), space=("DG", 0), method=sampling)
    
    orient, orient_values, orient_coeffs = _orientation(state)
    if orient is not None and orient[0] == "fiber_direction":
        from .fenics.ops import orientation_cos_sin_spec
        specs["orientation_cs"] = orientation_cos_sin_spec(_pde(state), orient[0], coeff_space=orient[1], target_space=("DG", 0), method=sampling)
        
    _, raw = _field_values(state, specs, values=orient_values, coefficients=orient_coeffs)
    values = _failure_field(raw.mid_strain, raw.curvature, raw.shear_strain, state.material, orientation_cs=getattr(raw, "orientation_cs", None))
    return _field(state, ("DG", 0, (values.shape[1],)), values)


def failure_index(state, *, rho=100, sampling="average"):
    """KS-aggregated Tsai-Wu failure index over the whole laminate.

    Parameters
    ----------
    state : ShellState
    rho : float, optional
        KS aggregation exponent. Default 100.
    sampling : {'average', 'midpoint'}, optional
        How the DG0 strain measures are sampled per cell. Default ``'average'``.

    Returns
    -------
    csdl.Variable
        Scalar. Below 1 means no ply face has failed; the log-sum-exp KS aggregate
        bounds the true maximum from **above**.

    Raises
    ------
    ValueError
        If the material carries no layup. Only :func:`~hermit.laminate` sets one.
    """
    if state.material.layup is None:
        raise ValueError("failure_index needs a composite ply layup, which only "
                         "hm.laminate carries; hm.composite takes ABD matrices and "
                         "cannot recover per-ply failure")
    specs = _strain_specs(_pde(state), space=("DG", 0), method=sampling)
    
    orient, orient_values, orient_coeffs = _orientation(state)
    if orient is not None and orient[0] == "fiber_direction":
        from .fenics.ops import orientation_cos_sin_spec
        specs["orientation_cs"] = orientation_cos_sin_spec(_pde(state), orient[0], coeff_space=orient[1], target_space=("DG", 0), method=sampling)
        
    _, raw = _field_values(state, specs, values=orient_values, coefficients=orient_coeffs)
    return _failure_index(raw.mid_strain, raw.curvature, raw.shear_strain,
                          state.material, orientation_cs=getattr(raw, "orientation_cs", None), rho=rho)


def nodal_displacement(state):
    """Displacements at the mesh vertices.

    Parameters
    ----------
    state : ShellState

    Returns
    -------
    csdl.Variable
        Shape ``(n_nodes, 3)``, in **file** vertex order -- the ordering an
        external geometry pipeline uses.
    """
    from .fenics.maps import OrderingMaps
    maps = OrderingMaps(state.domain.mesh, _pde(state))
    return H.extract_nodal(state.disp_solid, maps.nodal_disp_map, maps.geom_shape,
                           state.domain.reverse_node_idx)


def nodal_rotation(state):
    """Director rotations at the mesh vertices.

    Parameters
    ----------
    state : ShellState

    Returns
    -------
    csdl.Variable
        Shape ``(n_nodes, 3)``, in **file** vertex order.
    """
    from .fenics.maps import OrderingMaps
    maps = OrderingMaps(state.domain.mesh, _pde(state))
    return H.extract_nodal(state.disp_solid, maps.nodal_rot_map, maps.geom_shape,
                           state.domain.reverse_node_idx)
