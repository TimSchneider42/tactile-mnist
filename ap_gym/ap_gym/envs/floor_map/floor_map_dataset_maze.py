from __future__ import annotations

from typing import Sequence, SupportsInt

import numpy as np

from .floor_map_dataset import FloorMapDataset


class FloorMapDatasetMaze(FloorMapDataset):
    def __init__(
        self,
        width: int = 21,
        height: int = 21,
        branching_prob: float = 1.0,
    ):
        if width % 2 == 0 or height % 2 == 0:
            raise ValueError("Width and height must be odd.")
        self.__width = width
        self.__height = height
        self.__branching_prob = branching_prob
        super().__init__(self.__width, self.__height)

    def get_data_point(self, idx: SupportsInt) -> np.ndarray:
        rng = np.random.default_rng(int(idx))

        # Create a maze full of walls (represented by 1)
        maze = np.ones((self.__height, self.__width), dtype=np.bool_)
        dimensions = np.array([self.__width, self.__height], dtype=np.int_)

        def carve(pos: np.ndarray):
            first = True  # Always carve at least one direction to ensure connectivity
            for direction in rng.permutation(directions):
                next_pos = pos + np.array(direction)
                if (
                    np.all(0 < next_pos)
                    and np.all(next_pos < dimensions - 1)
                    and maze[next_pos[1], next_pos[0]] == 1
                ):
                    # Always carve the first eligible branch; subsequent ones only if allowed by branching_prob
                    if first or rng.random() < self.__branching_prob:
                        intermediate_pos = pos + direction // 2
                        maze[intermediate_pos[1], intermediate_pos[0]] = (
                            False  # Carve passage between cells
                        )
                        maze[next_pos[1], next_pos[0]] = False  # Carve target cell
                        carve(next_pos)
                        first = False

        starting_pos = np.ones(2, dtype=np.int_)
        maze[tuple(starting_pos)] = 0
        directions = np.array([[2, 0], [-2, 0], [0, 2], [0, -2]])
        carve(starting_pos)

        return maze

    def get_data_point_batch(
        self, idx: Sequence[SupportsInt] | np.ndarray
    ) -> np.ndarray:
        return np.stack([self.get_data_point(i) for i in idx])

    def _get_length(self) -> int:
        return 2**32
