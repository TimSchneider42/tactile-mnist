from __future__ import annotations

from abc import ABC

import numpy as np

from ap_gym.envs.dataset import Dataset


class FloorMapDataset(Dataset[np.ndarray, np.ndarray], ABC):
    def __init__(self, map_width: int, map_height: int):
        self.__map_width = map_width
        self.__map_height = map_height

    @property
    def map_width(self) -> int:
        return self.__map_width

    @property
    def map_height(self) -> int:
        return self.__map_height
