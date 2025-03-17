from __future__ import annotations

from pathlib import Path

import numpy as np

GELSIGHT_DIMS = np.array([0.032, 0.028])
GRID_BORDER_THICKNESS = 0.005
CELL_SIZE = np.array([0.12, 0.12])
CELL_MARGIN = np.array([0.003, 0.003]) + GELSIGHT_DIMS / 2 + GRID_BORDER_THICKNESS / 2
GELSIGHT_GEL_THICKNESS_MM = 4.25
GELSIGHT_IMAGE_SIZE_PX = np.array([320, 240])
CACHE_BASE_DIR = Path.home() / ".cache" / "tactile-mnist"
