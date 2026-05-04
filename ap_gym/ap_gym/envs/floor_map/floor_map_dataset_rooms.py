from __future__ import annotations

from typing import Sequence, SupportsInt

import numpy as np

from .floor_map_dataset import FloorMapDataset


class FloorMapDatasetRooms(FloorMapDataset):
    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        max_rooms: bool = 10,
        door_width: int = 3,
    ):
        self.__width = width
        self.__height = height
        self.__max_rooms = max_rooms
        self.__door_width = door_width
        self.__min_room_size = self.__door_width + 2
        super().__init__(self.__width, self.__height)

    def get_data_point(self, idx: SupportsInt) -> np.ndarray:
        rng = np.random.default_rng(int(idx))

        map_int = np.zeros((self.__height, self.__width), dtype=np.int8)

        # Make walls
        map_int[0, :] = 1
        map_int[-1, :] = 1
        map_int[:, 0] = 1
        map_int[:, -1] = 1

        def distribute_integers(n: int, k: int, rng: np.random.Generator):
            r = np.arange(1, n)
            r = np.concatenate([np.zeros(max(0, k - n), dtype=np.int_), r])
            cuts = np.sort(rng.choice(r, k - 1, replace=False))
            return np.diff(np.concatenate(([0], cuts, [n])))

        def split_room(room: np.ndarray, max_rooms: int):
            max_rooms_local = min(
                max_rooms,
                (room.shape[0] - self.__min_room_size) // (self.__min_room_size + 1)
                + 1,
            )
            if max_rooms_local <= 1:
                return
            sub_rooms = rng.binomial(max_rooms_local - 2, 0.3) + 2
            sub_room_capacity = distribute_integers(max_rooms_local, sub_rooms, rng)
            room_sizes = (
                distribute_integers(
                    room.shape[0] - sub_rooms * (1 + self.__min_room_size) + 1,
                    sub_rooms,
                    rng,
                )
                + self.__min_room_size
            )
            room_sizes_with_walls = room_sizes + 1
            room_ends = np.cumsum(room_sizes_with_walls) - 1
            room_starts = np.concatenate(([0], room_ends[:-1] + 2))
            wall_positions = room_starts[1:] - 1

            door_positions = rng.integers(
                0, room.shape[1] - self.__door_width, size=sub_rooms - 1
            )
            door_range = np.arange(self.__door_width)

            room[wall_positions] = np.where(room[wall_positions] != -1, 1, -1)
            room[
                wall_positions[:, None, None] + door_range[None, :, None],
                door_positions[:, None, None] + door_range[None, None, :],
            ] = -1
            room[
                wall_positions[:, None, None] - door_range[None, :, None],
                door_positions[:, None, None] + door_range[None, None, :],
            ] = -1

            for s, e, c in zip(room_starts, room_ends, sub_room_capacity):
                split_room(room[s : e + 1].T, c)

        split_room(map_int[1:-1, 1:-1], self.__max_rooms)
        map_int[map_int == -1] = 0

        if rng.integers(0, 2) == 0:
            map_int = map_int.T

        return map_int.astype(np.bool_)

    def get_data_point_batch(
        self, idx: Sequence[SupportsInt] | np.ndarray
    ) -> np.ndarray:
        return np.stack([self.get_data_point(i) for i in idx])

    def _get_length(self) -> int:
        return 2**32
