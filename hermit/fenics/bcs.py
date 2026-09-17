"""Boundary-condition data for the shell model.

:class:`BCData` is the FE-level bundle ``ShellSolveOp`` consumes: penalty ``ds``/``dS``
measures (fed to ``ElasticModel._penalty_residual``) plus optional strong Dirichlet
BCs. It is built by ``hermit.bcs.BoundaryConditions.to_bc_data()`` -- the builders
(``clamp`` / ``pin`` / ``symmetry`` / ``gauge``) live there, along with the measure
construction and the gauge-point pins.

Port of ``RMShellModel.set_up_bcs`` / ``_add_strong_bcs`` / ``_build_gauge_point_bcs``.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BCData:
    penalty: bool = True
    dss: object = None          # custom ds measure (or None) -- the single-term case
    dSS: object = None          # custom dS measure (or None) -- the single-term case
    bc_dof_mask: tuple | None = None
    # Target for the common single penalty term.  It is a Function on the mixed
    # state space, built by hermit.bcs; None retains the historical zero target.
    g: object = None
    strong: list = field(default_factory=list)   # list[dolfinx DirichletBC]
    # >1 differently-masked penalty region (hermit.bcs's multi-term merge): a list of
    # (dss, dSS, bc_dof_mask) triples, summed by ElasticModel._penalty_residual. Empty
    # for the common 0-or-1-term case, where dss/dSS/bc_dof_mask above are used
    # instead -- see ShellSolveOp, which picks whichever is non-empty.
    penalty_terms: list = field(default_factory=list)
    # Targets corresponding positionally to ``penalty_terms``. Kept separate so the
    # established public/FE triple representation remains backwards compatible.
    penalty_targets: list = field(default_factory=list)
    # Located facet ids parallel to the effective penalty terms.  This is retained
    # so an edge-load form can share one exterior-facet MeshTags object with a
    # penalty form (DOLFINx requires that within a compiled form).
    penalty_entities: list = field(default_factory=list)

    @property
    def strong_dofs(self):
        if not self.strong:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate([bc.dof_indices()[0] for bc in self.strong]))
