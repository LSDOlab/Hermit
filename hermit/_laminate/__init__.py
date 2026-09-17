"""Classical lamination theory + Tsai-Wu failure, vendored from LamAD.

This is a trimmed copy of the parts of `LamAD <https://github.com/LSDOlab/LamAD>`_
that Hermit needs: the ``Layup`` container, ``compute_clt`` (per-ply ABD), and the
Tsai-Wu failure index. LamAD is a private repo, so vendoring keeps Hermit
pip-installable without it. Both projects are LGPL-3.0-or-later.

The ``caddee_materials`` ``Material`` classes (``TransverseMaterial`` /
``IsotropicMaterial``) are still an external dependency -- that repo is public.
"""

from .clt import compute_clt
from .failure import (
    aggregate_failure,
    ply_local_stress_from_strain,
    ply_properties,
    tsai_wu_failure_index,
    tsai_wu_field,
    tsai_wu_stress_ratio,
)
from .layup import Layup

__all__ = [
    "Layup",
    "compute_clt",
    "ply_properties",
    "tsai_wu_field",
    "aggregate_failure",
    "tsai_wu_failure_index",
    "tsai_wu_stress_ratio",
    "ply_local_stress_from_strain",
]
