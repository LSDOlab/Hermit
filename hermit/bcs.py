"""``BoundaryConditions`` -- the BC builders ``clamp`` / ``pin`` / ``symmetry`` /
``gauge``, compiled once against a :class:`hermit.ShellDomain`.

No CSDL is involved, so a compiled ``BoundaryConditions`` survives across every
``solve`` for its domain. Two enforcement paths are available: a penalty term over
the selected facets (the default; it eliminates no dofs and supports several
independently masked regions in one form) and strong Dirichlet constraints
(``method="strong"``). The penalty scale is large, so under a large geometry
perturbation the strong path conditions better -- see the shape-derivative docs.

Selecting a region: ``where``
-----------------------------
``where`` is the raw DOLFINx entity locator, ``Callable[[ndarray], ndarray[bool]]``,
receiving coordinates of shape ``(3, N)`` and returning an ``(N,)`` boolean mask.
Two helpers build the common cases:

- :func:`near` -- ``x[axis] == value`` within an absolute tolerance.
- :func:`on_plane` -- points on a plane through a point with a given normal, which
  need not be axis-aligned.

Prescribed values
-----------------
Every builder takes ``value=`` -- a scalar, a ``(6,)`` vector over
``(ux, uy, uz, rx, ry, rz)``, or a callable returning ``(N, 6)`` for its ``(N, 3)``
coordinates. It defaults to zero. A ``csdl.Variable`` is deliberately **not**
accepted: an enforced displacement is interpolated once, not carried as a
differentiable solve input, so it cannot be a design variable. Loads and material
properties can.

The ``symmetry`` mask
---------------------
On a symmetry plane with unit normal ``n``, the constrained dofs are the
displacement *along* ``n`` and the two rotations *about the in-plane axes*; the
rotation about ``n`` itself is left free. For an axis-aligned plane with normal
index ``k`` that is ``mask = (m0, m1, m2, r0, r1, r2)`` with ``m_k = 1`` and
``r_i = 1`` for ``i != k`` -- normal ``+x`` gives ``(1, 0, 0, 0, 1, 1)``, the
standard "SPC on trans-x / rot-y / rot-z" convention. The penalty machinery supports
only an axis-aligned 6-component mask, so ``symmetry`` raises on a rotated normal.

Merging: later terms win on overlap
-----------------------------------
``a + b`` merges two ``BoundaryConditions`` on the same domain. Walking the merged
term list from last-added to first, each term's located entities are removed from
every earlier term before that term's measures are recompiled. So on a facet claimed
by two regions only the later region's mask governs, rather than the sum of both
masks; a term that loses all of its entities is dropped. Strong constraints are
concatenated in ``+`` order, and DOLFINx applies them in order, giving the same
"later wins" rule.

``penalty_beta``
----------------
Stored on the whole compiled object (default ``1e15``); the effective penalty is
``beta / h`` through the form's own ``CellDiameter``. Merging two objects that both
carry penalty terms with different explicit ``penalty_beta`` raises rather than
silently picking one.

One state space per domain
--------------------------
A ``BoundaryConditions`` is valid only for a solve sharing its ``domain.W`` object.
DOLFINx does not treat two independently built but structurally identical
``FunctionSpace``\\s as interchangeable for ``DirichletBC`` application: it does not
raise, it silently solves the wrong problem. See :class:`hermit.ShellDomain`.
"""

from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx import mesh as dmesh
from dolfinx.fem import Function, dirichletbc, locate_dofs_geometrical

_DOF_NAMES = ("ux", "uy", "uz", "rx", "ry", "rz")
_DEFAULT_PENALTY_BETA = 1.0e15
_BC_TAG = 100  # same tag hermit.fenics.bcs.build_bc uses -- each Measure owns its own
               # meshtags object, so tag reuse across separate BoundaryConditions is fine.


# -- where helpers ------------------------------------------------------------

_AXES = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}


def _axis_index(axis):
    try:
        return _AXES[axis]
    except (KeyError, TypeError):
        raise ValueError(f"axis must be 0/1/2 or 'x'/'y'/'z', got {axis!r}")


def near(axis, value, *, tol=1e-12):
    """Build a ``where`` predicate selecting an axis-aligned plane.

    Parameters
    ----------
    axis : {0, 1, 2, 'x', 'y', 'z'}
        Coordinate axis to test.
    value : float
        Coordinate value the plane sits at.
    tol : float, optional
        Absolute tolerance. Default 1e-12.

    Returns
    -------
    callable
        Suitable as ``where=`` for any BC or edge-load builder.

    Raises
    ------
    ValueError
        If ``axis`` is not one of the accepted names.

    Notes
    -----
    The comparison is purely absolute. A relative tolerance would swamp ``tol`` far
    from the origin -- ``near("x", 10.0)`` would select everything within 1e-4 of
    the plane.

    Examples
    --------
    >>> bcs = hm.clamp(domain, where=hm.near("x", 0.0))
    """
    idx = _axis_index(axis)
    value = float(value)
    return lambda x: np.isclose(x[idx], value, rtol=0.0, atol=tol)


def on_plane(point, normal, *, tol=1e-12):
    """Build a ``where`` predicate selecting an arbitrary plane.

    Parameters
    ----------
    point : array_like
        A ``(3,)`` point on the plane.
    normal : array_like
        A ``(3,)`` plane normal; need not be unit or axis-aligned.
    tol : float, optional
        Absolute distance tolerance. Default 1e-12.

    Returns
    -------
    callable
        Suitable as ``where=`` for any BC or edge-load builder.

    See Also
    --------
    near : the axis-aligned shorthand.
    """
    point = np.asarray(point, dtype=float).reshape(3)
    normal = np.asarray(normal, dtype=float).reshape(3)
    normal = normal / np.linalg.norm(normal)

    def _locator(x):
        d = normal[0] * (x[0] - point[0]) + normal[1] * (x[1] - point[1]) + normal[2] * (x[2] - point[2])
        return np.abs(d) < tol

    return _locator


# -- dof-name <-> mask ----------------------------------------------------

def _dofs_mask(dofs):
    dofs = tuple(dofs)
    unknown = sorted(set(dofs) - set(_DOF_NAMES))
    if unknown:
        raise ValueError(f"unknown dof name(s) {unknown}; choose from {_DOF_NAMES}")
    return tuple(1 if name in dofs else 0 for name in _DOF_NAMES)


_AXIS_ALIGN_TOL = 1e-8


def _axis_aligned_index(normal):
    n = np.asarray(normal, dtype=float).reshape(3)
    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError("symmetry(): normal must be nonzero")
    n = n / norm
    idx = int(np.argmax(np.abs(n)))
    off_axis = np.delete(n, idx)
    if not np.isclose(abs(n[idx]), 1.0, atol=_AXIS_ALIGN_TOL) or np.any(np.abs(off_axis) > _AXIS_ALIGN_TOL):
        raise ValueError(
            f"symmetry() needs an axis-aligned normal -- the penalty machinery only "
            f"supports a 6-component 0/1 dof mask expressed in the global x/y/z axes, "
            f"not an arbitrary rotated frame (there is no per-BC rotated-frame "
            f"projection in ElasticModel._penalty_residual). Got normal={normal!r} "
            f"(normalized {n!r}), not close to +/-x, +/-y or +/-z."
        )
    return idx


# -- compiled penalty measures --------------------------------------------

def _compiled_measure(mesh, dim, entities, kind, quadrature_degree):
    ents = np.asarray(entities, dtype=np.int32)
    mt = dmesh.meshtags(mesh, dim, ents, np.full(len(ents), _BC_TAG, dtype=np.int32))
    return ufl.Measure(kind, domain=mesh, subdomain_data=mt,
                       metadata={"quadrature_degree": quadrature_degree})(_BC_TAG)


@dataclass
class _PenaltyTerm:
    mask: tuple
    g: object                 # prescribed mixed-state Function on domain.W
    entities_ds: np.ndarray   # exterior facets (FE-local ids), locate_entities_boundary
    entities_dS: np.ndarray   # all facets (FE-local ids), locate_entities
    dss: object
    dSS: object


def _make_penalty_term(mesh, mask, g, entities_ds, entities_dS, quadrature_degree):
    fdim = mesh.topology.dim - 1
    return _PenaltyTerm(
        mask=tuple(int(v) for v in mask),
        g=g,
        entities_ds=np.asarray(entities_ds, dtype=np.int32),
        entities_dS=np.asarray(entities_dS, dtype=np.int32),
        dss=_compiled_measure(mesh, fdim, entities_ds, "ds", quadrature_degree),
        dSS=_compiled_measure(mesh, fdim, entities_dS, "dS", quadrature_degree),
    )


def _merge_penalty_terms(domain, terms_a, terms_b):
    """"Later wins": walk last-added -> first-added, subtracting each term's located
    entities from every earlier term before it is (re)compiled. A term left with no
    entities on either measure is dropped."""
    ordered = list(terms_a) + list(terms_b)
    claimed_ds, claimed_dS = set(), set()
    kept_rev = []
    for term in reversed(ordered):
        ds_ids = set(term.entities_ds.tolist())
        dS_ids = set(term.entities_dS.tolist())
        ds_keep = np.array(sorted(ds_ids - claimed_ds), dtype=np.int32)
        dS_keep = np.array(sorted(dS_ids - claimed_dS), dtype=np.int32)
        claimed_ds |= ds_ids
        claimed_dS |= dS_ids
        if ds_keep.size or dS_keep.size:
            kept_rev.append(_make_penalty_term(domain.mesh, term.mask, term.g, ds_keep, dS_keep, domain.quadrature_degree))
    kept_rev.reverse()
    return kept_rev


def _shared_measures(domain, terms, fdim):
    """Rebuild each term's ``(dss, dSS)`` sharing **one** ``MeshTags`` object per
    measure kind (a distinct tag per term), instead of each term's own independently
    tagged ``MeshTags`` (what ``_make_penalty_term`` gives every term, correct for a
    single term used alone). dolfinx requires every integral of a given type inside
    one compiled ``Form`` to share the same ``subdomain_data`` object -- summing two
    terms' independently-tagged measures into one residual
    (``ElasticModel._penalty_residual``'s multi-term sum) violates that and fails at
    ``dolfinx.fem.form()`` with an opaque assertion. Terms are disjoint per measure
    kind by construction (``_merge_penalty_terms``'s subtraction), so a plain
    concatenation is safe."""
    def _mt(ent_lists, tag_lists):
        if not ent_lists:
            return dmesh.meshtags(domain.mesh, fdim, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
        ents, tags = np.concatenate(ent_lists), np.concatenate(tag_lists)
        order = np.argsort(ents)
        return dmesh.meshtags(domain.mesh, fdim, ents[order], tags[order])

    ds_ents, ds_tags, dS_ents, dS_tags = [], [], [], []
    for i, t in enumerate(terms):
        tag = i + 1
        if t.entities_ds.size:
            ds_ents.append(t.entities_ds)
            ds_tags.append(np.full(t.entities_ds.size, tag, dtype=np.int32))
        if t.entities_dS.size:
            dS_ents.append(t.entities_dS)
            dS_tags.append(np.full(t.entities_dS.size, tag, dtype=np.int32))

    md = {"quadrature_degree": domain.quadrature_degree}
    ds_measure = ufl.Measure("ds", domain=domain.mesh, subdomain_data=_mt(ds_ents, ds_tags), metadata=md)
    dS_measure = ufl.Measure("dS", domain=domain.mesh, subdomain_data=_mt(dS_ents, dS_tags), metadata=md)
    return [(ds_measure(i + 1), dS_measure(i + 1)) for i in range(len(terms))]


def _value_evaluator(value):
    """Interpolate a public prescribed value into a mixed state Function.

    Public callables follow ``hm.from_function``: they receive ``(N, 3)``
    coordinates and return a scalar, ``(6,)``, or ``(N, 6)`` values.  Arrays use
    the same scalar/6-vector convention.  A CSDL Variable is deliberately not
    accepted here: making this coefficient differentiable needs a solve-input path,
    analogous to distributed loads, rather than a one-time interpolation.
    """
    if value is None:
        value = 0.0
    if value.__class__.__module__.split(".")[0] == "csdl_alpha":
        raise TypeError("prescribed BC values cannot yet be csdl.Variable; use a scalar, numpy array, or callable")
    # dolfinx.fem.Constant has a numpy ``value``; accepting it makes a spatially
    # constant FE coefficient as useful here as a Python/numpy constant.
    if not callable(value) and hasattr(value, "value"):
        value = value.value

    def values_at(x):
        n = x.shape[1]
        raw = value(x.T) if callable(value) else value
        if hasattr(raw, "value") and not isinstance(raw, np.ndarray):
            raw = raw.value
        a = np.asarray(raw, dtype=float)
        if a.ndim == 0 or a.size == 1:
            return np.full((6, n), float(a.reshape(-1)[0]))
        if a.shape == (6,):
            return np.broadcast_to(a[:, None], (6, n))
        if a.shape == (n, 6):
            return a.T
        # No (6, N) branch: it is indistinguishable from (N, 6) on a six-point
        # patch, and the public convention here is hm.from_function's (N, 6).
        raise ValueError(
            "BC value must be a scalar, shape (6,), or a callable returning shape "
            f"(N, 6) for its (N, 3) coordinates; got shape {a.shape}")

    return values_at


def _target_function(W, value):
    values_at = _value_evaluator(value)
    g = Function(W)
    g.sub(0).interpolate(lambda x: values_at(x)[:3])
    g.sub(1).interpolate(lambda x: values_at(x)[3:])
    g.x.scatter_forward()
    return g


def _penalty_bc(domain, where, mask, penalty_beta, value):
    fdim = domain.mesh.topology.dim - 1
    ents_ds = dmesh.locate_entities_boundary(domain.mesh, fdim, where)
    ents_dS = dmesh.locate_entities(domain.mesh, fdim, where)
    beta = _DEFAULT_PENALTY_BETA if penalty_beta is None else float(penalty_beta)
    term = _make_penalty_term(domain.mesh, mask, _target_function(domain.W, value),
                              ents_ds, ents_dS, domain.quadrature_degree)
    return BoundaryConditions(domain, penalty_terms=[term], strong=[], penalty_beta=beta)


# -- strong (Dirichlet) bcs, masked by named dof --------------------------
# Both helpers locate dofs per scalar sub-subspace (mirroring
# hermit.fenics.bcs._gauge_bcs) rather than the whole vector sub(0)/sub(1) the way
# build_bc's "clamp all" strong path does -- that is the only way to support a *subset*
# of the 6 dofs (build_bc's strong path has no such mask). For a full clamp (all 6)
# the two are set-equivalent: same physical dofs, just grouped differently.

def _component_subs(W):
    return (W.sub(0).sub(0), W.sub(0).sub(1), W.sub(0).sub(2),
           W.sub(1).sub(0), W.sub(1).sub(1), W.sub(1).sub(2))


def _strong_bcs_where(W, where, components, value):
    subs = _component_subs(W)
    values_at = _value_evaluator(value)
    out = []
    for c in components:
        sub = subs[c]
        Vs, _ = sub.collapse()
        dofs = locate_dofs_geometrical((sub, Vs), where)
        # A DirichletBC on a subspace needs its value Function on the *collapsed*
        # scalar subspace.  A mixed-space Function happens to work for zero values,
        # but reads unrelated mixed dofs for a nonzero target.
        target = Function(Vs)
        target.interpolate(lambda x, c=c: values_at(x)[c])
        out.append(dirichletbc(target, dofs, sub))
    return out


def _gauge_bcs(W, point, components, value):
    from scipy.spatial import cKDTree

    subs = _component_subs(W)
    values_at = _value_evaluator(value)
    out = []
    for c in components:
        sub = subs[c]
        Vs, _ = sub.collapse()
        coords = Vs.tabulate_dof_coordinates()
        target = coords[cKDTree(coords).query(point)[1]]
        locator = lambda x, target=target: np.all(np.isclose(x, target[:, None], atol=1e-9), axis=0)
        dofs = locate_dofs_geometrical((sub, Vs), locator)
        if dofs[0].size == 0:
            raise ValueError(  # pragma: no cover -- the nearest coordinate always self-matches
                f"gauge(): nearest-dof search for component {_DOF_NAMES[c]!r} found no "
                f"matching dof; unexpected."
            )
        # Same collapsed-subspace requirement as _strong_bcs_where: a mixed-space
        # Function only works by accident for a zero target.
        value_function = Function(Vs)
        value_function.interpolate(lambda x, c=c: values_at(x)[c])
        out.append(dirichletbc(value_function, dofs, sub))
    return out


# -- BoundaryConditions -----------------------------------------------------

class BoundaryConditions:
    """Located facets, dofs and measures, compiled once against a domain.

    Build one with :func:`clamp`, :func:`pin`, :func:`symmetry` or :func:`gauge`
    rather than calling this constructor directly. ``a + b`` merges two objects on
    the same domain, with later terms winning on overlapping entities.

    Parameters
    ----------
    domain : ShellDomain
    penalty_terms : sequence, optional
        Compiled penalty regions, each carrying a dof mask, a prescribed target and
        its facet measures.
    strong : sequence of dolfinx.fem.DirichletBC, optional
        Strong constraints.
    penalty_beta : float, optional
        Penalty scale for the whole object; ``None`` selects the default 1e15.

    Attributes
    ----------
    strong_dofs : ndarray
        Sorted unique dof indices held by the strong constraints.

    Notes
    -----
    Valid only for a solve that shares this ``domain``'s state space; see the module
    docstring.
    """

    def __init__(self, domain, *, penalty_terms=(), strong=(), penalty_beta=None):
        self.domain = domain
        self.penalty_terms = list(penalty_terms)
        self.strong = list(strong)
        self.penalty_beta = penalty_beta

    @property
    def strong_dofs(self):
        if not self.strong:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate([bc.dof_indices()[0] for bc in self.strong]))

    def __add__(self, other):
        if not isinstance(other, BoundaryConditions):
            return NotImplemented
        if other.domain is not self.domain:
            raise ValueError("BoundaryConditions.__add__ needs both operands on the same ShellDomain")
        if self.penalty_terms and other.penalty_terms and self.penalty_beta != other.penalty_beta:
            raise ValueError(
                f"penalty_beta mismatch on merge: {self.penalty_beta!r} vs "
                f"{other.penalty_beta!r} -- today's penalty residual takes one beta "
                f"for the whole compiled BoundaryConditions (see the module "
                f"docstring); pass matching penalty_beta= on both sides."
            )
        beta = self.penalty_beta if self.penalty_terms else other.penalty_beta
        merged_terms = _merge_penalty_terms(self.domain, self.penalty_terms, other.penalty_terms)
        return BoundaryConditions(self.domain, penalty_terms=merged_terms,
                                  strong=self.strong + other.strong, penalty_beta=beta)

    def to_bc_data(self):
        """Pack this object for the FEniCSx solve operation.

        Returns
        -------
        hermit.fenics.bcs.BCData
            Carrying a single masked penalty region when there is at most one, and
            a list of independently masked regions when several survive a merge.
        """
        from .fenics.bcs import BCData

        if not self.penalty_terms:
            return BCData(penalty=False, dss=None, dSS=None, bc_dof_mask=None,
                          strong=list(self.strong), penalty_entities=[])
        if len(self.penalty_terms) == 1:
            t = self.penalty_terms[0]
            return BCData(penalty=True, dss=t.dss, dSS=t.dSS, bc_dof_mask=t.mask, g=t.g,
                          strong=list(self.strong), penalty_entities=[t.entities_ds])
        fdim = self.domain.mesh.topology.dim - 1
        shared = _shared_measures(self.domain, self.penalty_terms, fdim)
        penalty_terms = [(dss, dSS, t.mask) for (dss, dSS), t in zip(shared, self.penalty_terms)]
        return BCData(penalty=True, dss=None, dSS=None, bc_dof_mask=None,
                      strong=list(self.strong), penalty_terms=penalty_terms,
                      penalty_targets=[t.g for t in self.penalty_terms],
                      penalty_entities=[t.entities_ds for t in self.penalty_terms])


# -- builders ---------------------------------------------------------------

def clamp(domain, *, where, method="penalty", penalty_beta=None, value=0.0):
    """Fix all six generalized dofs on a region.

    Parameters
    ----------
    domain : ShellDomain
    where : callable
        Coordinate predicate; receives a ``(3, N)`` array and returns an ``(N,)``
        boolean mask. :func:`near` and :func:`on_plane` build the common cases.
    method : {'penalty', 'strong'}, optional
        Enforcement path. Default ``'penalty'``.
    penalty_beta : float, optional
        Penalty scale, ``'penalty'`` only. Default 1e15.
    value : float, array_like or callable, optional
        Prescribed value: a scalar, a ``(6,)`` vector over
        ``(ux, uy, uz, rx, ry, rz)``, or a callable returning ``(N, 6)`` for its
        ``(N, 3)`` coordinates. Default 0. Not differentiable -- see the module
        docstring.

    Returns
    -------
    BoundaryConditions

    Raises
    ------
    ValueError
        If ``method`` is not recognised, or if ``penalty_beta`` is given with
        ``method='strong'``.

    Examples
    --------
    >>> bcs = hm.clamp(domain, where=hm.near("x", 0.0))

    See Also
    --------
    pin : constrain a subset of the six dofs.
    """
    return pin(domain, where=where, dofs=_DOF_NAMES, method=method,
               penalty_beta=penalty_beta, value=value)


def pin(domain, *, where, dofs, method="penalty", penalty_beta=None, value=0.0):
    """Fix a chosen subset of the six generalized dofs on a region.

    Parameters
    ----------
    domain : ShellDomain
    where : callable
        Coordinate predicate; see :func:`clamp`.
    dofs : sequence of str
        Any subset of ``('ux', 'uy', 'uz', 'rx', 'ry', 'rz')``.
    method : {'penalty', 'strong'}, optional
        Enforcement path. Default ``'penalty'``.
    penalty_beta : float, optional
        Penalty scale, ``'penalty'`` only.
    value : float, array_like or callable, optional
        Prescribed value; see :func:`clamp`. Default 0.

    Returns
    -------
    BoundaryConditions

    Raises
    ------
    ValueError
        If a dof name is unknown, if ``method`` is not recognised, or if
        ``penalty_beta`` is given with ``method='strong'``.

    Examples
    --------
    >>> diaphragm = hm.pin(domain, where=hm.near("x", 0.0), dofs=("uy", "uz"))
    """
    mask = _dofs_mask(dofs)
    if method == "penalty":
        return _penalty_bc(domain, where, mask, penalty_beta, value)
    if method == "strong":
        if penalty_beta is not None:
            raise ValueError("penalty_beta is only meaningful for method='penalty'")
        components = [i for i, m in enumerate(mask) if m]
        return BoundaryConditions(domain, strong=_strong_bcs_where(domain.W, where, components, value))
    raise ValueError(f"method must be 'penalty' or 'strong', got {method!r}")


def symmetry(domain, *, where, normal, penalty_beta=None, value=0.0):
    """Apply a symmetry-plane constraint (penalty enforcement).

    Constrains the displacement along ``normal`` and the two rotations about the
    in-plane axes, leaving the rotation about ``normal`` free.

    Parameters
    ----------
    domain : ShellDomain
    where : callable
        Coordinate predicate selecting the plane; see :func:`clamp`.
    normal : array_like
        A ``(3,)`` plane normal, which must be (close to) axis-aligned.
    penalty_beta : float, optional
        Penalty scale. Default 1e15.
    value : float, array_like or callable, optional
        Prescribed value; see :func:`clamp`. Default 0.

    Returns
    -------
    BoundaryConditions

    Raises
    ------
    ValueError
        If ``normal`` is zero or not axis-aligned. The penalty form carries a
        6-component mask in global x/y/z, with no rotated-frame projection.

    Examples
    --------
    >>> half = hm.symmetry(domain, where=hm.near("y", 0.0), normal=[0, 1, 0])
    """
    k = _axis_aligned_index(normal)
    disp_mask = [0, 0, 0]
    disp_mask[k] = 1
    rot_mask = [1, 1, 1]
    rot_mask[k] = 0
    return _penalty_bc(domain, where, tuple(disp_mask + rot_mask), penalty_beta, value)


def gauge(domain, *, at, dofs, value=0.0):
    """Pin selected dofs at a single point, to remove rigid-body modes.

    Always enforced strongly. Use it to fix the residual null space of an otherwise
    softly supported model.

    Parameters
    ----------
    domain : ShellDomain
    at : array_like
        A ``(3,)`` point; the nearest dof of each selected component is used.
    dofs : sequence of str
        Any subset of ``('ux', 'uy', 'uz', 'rx', 'ry', 'rz')``.
    value : float, array_like or callable, optional
        Prescribed value; see :func:`clamp`. Default 0.

    Returns
    -------
    BoundaryConditions

    Raises
    ------
    ValueError
        If a dof name is not one of the six.

    Examples
    --------
    >>> bcs = supports + hm.gauge(domain, at=[0.0, 0.0, 0.0], dofs=("ux", "uy"))
    """
    mask = _dofs_mask(dofs)
    components = [i for i, m in enumerate(mask) if m]
    point = np.asarray(at, dtype=float).reshape(3)
    return BoundaryConditions(domain, strong=_gauge_bcs(domain.W, point, components, value))
