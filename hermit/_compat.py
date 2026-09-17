"""Cross-version shims for the FEniCSx stack.

Hermit targets **DOLFINx 0.9 and 0.11** (and the NumPy 1.x / 2.x that pair with
them). Almost all version drift is absorbed here; the rest is in
``hermit._ufl_compat`` (the mesh-coordinate-derivative patch, which has to reach
into UFL internals that were reorganised between the two releases).

Known differences handled:

* ``FiniteElement.interpolation_points`` is a method in 0.9, a property in 0.11.
* ``dolfinx.fem.petsc.apply_lifting`` took ``scale=`` in 0.9, ``alpha=`` in 0.11
  (Hermit never passes it, but the wrapper keeps call sites uniform).
* ``Geometry.cmap`` (0.9) became ``Geometry.cmaps[0]`` (0.11, ``cmap`` deprecated).
* ``Geometry.dofmap`` (0.9) became ``Geometry.dofmaps[0]`` (0.11, ``dofmap`` deprecated).
"""

import numpy as np

try:
    from dolfinx import __version__ as _dolfinx_version
except Exception:  # pragma: no cover - dolfinx always present at runtime
    _dolfinx_version = "0.0.0"


def _parse(v: str) -> tuple:
    out = []
    for part in v.split(".")[:3]:
        num = "".join(c for c in part if c.isdigit())
        out.append(int(num) if num else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out)


DOLFINX_VERSION = _parse(_dolfinx_version)


def interpolation_points(V) -> np.ndarray:
    """Reference interpolation points of a function space's element.

    ``V.element.interpolation_points`` is a bound method in DOLFINx <= 0.9 and a
    plain array property from 0.10 on.
    """
    ip = V.element.interpolation_points
    return np.asarray(ip() if callable(ip) else ip)


def coordinate_element(mesh):
    """The mesh's coordinate element -- ``Geometry.cmaps[0]`` from DOLFINx 0.10 on,
    ``Geometry.cmap`` in 0.9 (where ``cmaps`` does not exist)."""
    g = mesh.geometry
    cmaps = getattr(g, "cmaps", None)
    return cmaps[0] if cmaps is not None else g.cmap


def geometry_dofmap(mesh):
    """The mesh's geometry dofmap -- ``Geometry.dofmaps[0]`` from DOLFINx 0.10 on,
    ``Geometry.dofmap`` in 0.9 (where ``dofmaps`` does not exist)."""
    g = mesh.geometry
    dofmaps = getattr(g, "dofmaps", None)
    return dofmaps[0] if dofmaps is not None else g.dofmap
