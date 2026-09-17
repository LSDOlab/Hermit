"""``Layup`` -- a laminate ply stack (materials, angles, ply heights).

Vendored from ``lamad.src.laminate`` (the ``Laminate``/``get_constants`` helper,
which Hermit does not use, is dropped).
"""

from dataclasses import dataclass
from typing import Union

import numpy as np
import csdl_alpha as csdl
from caddee_materials import Material


@dataclass
class Layup(csdl.VariableGroup):
    """A laminate layup.

    Parameters
    ----------
    materials : Material | list[Material]
        The ply materials. A single ``Material`` is broadcast to every ply.
    angles : csdl.Variable | np.ndarray | float
        Ply orientation angles in radians. A scalar is broadcast to every ply.
    heights : csdl.Variable | np.ndarray | float
        Ply thicknesses. A scalar is broadcast to every ply.
    num_plies : int, optional
        Number of plies. Inferred from the other arguments when omitted.
    """

    materials: Union[Material, list]
    angles: Union[csdl.Variable, np.ndarray, float]      # radians
    heights: Union[csdl.Variable, np.ndarray, float]
    num_plies: int = None

    def __post_init__(self):
        if isinstance(self.angles, (np.ndarray, float)):
            self.angles = csdl.Variable(value=self.angles)
        if isinstance(self.heights, (np.ndarray, float)):
            self.heights = csdl.Variable(value=self.heights)

        if self.num_plies is not None:
            pass
        elif isinstance(self.materials, list):
            self.num_plies = len(self.materials)
        elif self.angles.shape[0] > 1:
            self.num_plies = self.angles.shape[0]
        elif self.heights.shape[0] > 1:
            self.num_plies = self.heights.shape[0]
        else:
            raise ValueError("Number of plies not specified")

        if isinstance(self.materials, Material):
            self.materials = [self.materials] * self.num_plies
        if self.angles.shape[0] == 1:
            self.angles = csdl.blockmat([[self.angles.reshape((1, 1))]] * self.num_plies)
        if self.heights.shape[0] == 1:
            self.heights = csdl.blockmat([[self.heights.reshape((1, 1))]] * self.num_plies)

        if len(self.materials) != self.num_plies:
            raise ValueError("Number of materials does not match number of plies")
        if self.angles.shape[0] != self.num_plies:
            raise ValueError("Number of angles does not match number of plies")
        if self.heights.shape[0] != self.num_plies:
            raise ValueError("Number of heights does not match number of plies")

        self.h = csdl.sum(self.heights)
