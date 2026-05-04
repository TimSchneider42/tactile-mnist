from __future__ import annotations

import numpy as np

COLOR_AGENT = (0, 85, 255)
COLOR_OBS_PRIMARY = (55, 255, 0)
COLOR_OBS_SECONDARY = (255, 55, 0)
COLOR_PRED = (200, 0, 200)
COLOR_GOOD = (0, 200, 0)
COLOR_BAD = (200, 0, 0)


def quality_color(quality: np.ndarray) -> tuple[int, int, int]:
    quality = np.clip(quality, 0, 1)[..., None]
    return tuple(
        (quality * np.array(COLOR_GOOD) + (1 - quality) * np.array(COLOR_BAD)).astype(
            np.int_
        )
    )
