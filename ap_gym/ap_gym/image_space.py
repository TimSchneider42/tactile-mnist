from __future__ import annotations

from typing import Type, Any

import gymnasium as gym
import numpy as np


class ImageSpace(gym.spaces.Box):
    def __init__(
        self,
        width: int,
        height: int,
        channels: int,
        batch_shape: tuple[int, ...] = (),
        dtype: Type[np.floating[Any]] = np.float32,
        seed: np.random.Generator | int | None = None,
        low: float | np.ndarray = 0.0,
        high: float | np.ndarray = 1.0,
    ):
        super().__init__(
            low, high, (*batch_shape, height, width, channels), dtype, seed
        )

    @classmethod
    def from_box(cls, box: gym.spaces.Box):
        return cls(
            box.shape[-2],
            box.shape[-3],
            box.shape[-1],
            box.shape[:-3],
            box.dtype,
            box.np_random,
            box.low,
            box.high,
        )

    @property
    def height(self):
        return self.shape[-3]

    @property
    def width(self):
        return self.shape[-2]

    @property
    def channels(self):
        return self.shape[-1]

    @property
    def batch_shape(self):
        return self.shape[:-3]

    def __repr__(self) -> str:
        return (
            f"ImageSpace({self.low_repr}, {self.high_repr}, {self.shape}, {self.dtype})"
        )


@gym.vector.utils.batch_space.register(ImageSpace)
def _batch_space_image_space(space: ImageSpace, n: int = 1):
    return ImageSpace.from_box(gym.vector.utils.space_utils._batch_space_box(space, n))
