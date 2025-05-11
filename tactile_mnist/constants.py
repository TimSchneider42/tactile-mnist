from __future__ import annotations

from pathlib import Path

import numpy as np

GELSIGHT_MINI_OUTER_SIZE = np.array([0.032, 0.028])
GELSIGHT_MINI_GEL_THICKNESS_MM = 4.25
GELSIGHT_MINI_IMAGE_SIZE_PX = np.array([320, 240])
GELSIGHT_MINI_SENSOR_SURFACE_SIZE = np.array([0.01888, 0.01416])
GRID_BORDER_THICKNESS = 0.005
CELL_SIZE = np.array([0.12, 0.12])
CELL_PADDING = (
    np.array([0.003, 0.003]) + GELSIGHT_MINI_OUTER_SIZE / 2 + GRID_BORDER_THICKNESS / 2
)
CACHE_BASE_DIR = Path.home() / ".cache" / "tactile-mnist"
GEL_PENETRATION_DEPTH_MM = GELSIGHT_MINI_GEL_THICKNESS_MM / 2
