from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..constants import GELSIGHT_MINI_GEL_THICKNESS_MM, GEL_PENETRATION_DEPTH_MM
from ..tactile_renderer import Device


class DepthEstimator(ABC):
    """Estimates the gel deformation that produced a tactile image.

    This is the inverse of a TactileRenderer: it maps real tactile images back to depth maps, which can then be fed
    to any TactileRenderer to re-render them in the domain of the simulated environments.
    """

    def __init__(self, device: Device, backend_name: str):
        self.__device = device
        self.__backend_name = backend_name

    @property
    def input_size(self) -> tuple[int, int]:
        """Size (width, height) the tactile images are scaled to before the depth is estimated."""
        return 256, 256

    @abstractmethod
    def estimate(
        self, images: np.ndarray, output_size: tuple[int, int]
    ) -> np.ndarray:
        """Estimate depth maps from tactile images.

        Parameters:
            images: (..., H, W, 3) tactile images, either uint8 or floating point in [0, 1].
            output_size: (width, height) of the returned depth maps.

        Returns a (..., output_size[1], output_size[0]) array of depths in meters, measured from the sensor as the
        TactileRenderers expect them: GEL_PENETRATION_DEPTH_MM for the deepest possible indentation and
        GELSIGHT_MINI_GEL_THICKNESS_MM where the gel is not in contact with anything.
        """

    def __call__(self, images: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
        return self.estimate(images, output_size)

    @property
    def device(self) -> Device:
        return self.__device

    @property
    def backend_name(self) -> str:
        return self.__backend_name


# The estimators predict the depth normalized the way DepthRenderer scales it, so it has to be scaled back to meters
DEPTH_SCALE_M = (GELSIGHT_MINI_GEL_THICKNESS_MM - GEL_PENETRATION_DEPTH_MM) / 1000
DEPTH_OFFSET_M = GEL_PENETRATION_DEPTH_MM / 1000
