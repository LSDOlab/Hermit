"""Constitutive model for the R-M shell.

Hermit always drives the FEniCSx residual with the ABD stiffness matrices (A, B, D for
membrane/coupling/bending, plus the transverse-shear ``As``). The isotropic case builds
A/B/D/As in CSDL upstream (``hermit.csdl_helpers.assemble_abd_isotropic``); composite via
``hermit._laminate.compute_clt``. This module just holds them as a CLT tuple for the elastic model.

Port of ``MaterialModelComposite2`` from the femo dev_coupling branch.
"""


class ABDHolder:
    """Wrap the DOLFINx A, B, D (on VABD) and As (on VAs) functions as a CLT tuple."""

    def __init__(self, A, B, D, As):
        self.A, self.B, self.D, self.As = A, B, D, As
        self.CLT = (A, B, D, As)
