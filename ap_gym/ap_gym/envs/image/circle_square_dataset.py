from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

import numpy as np

from .image_classification_dataset import ImageClassificationDataset


class BaseCircleSquareDataset(ImageClassificationDataset):
    def __init__(
        self,
        image_shape: tuple[int, int] = (28, 28),
        object_extents: int = 8,
    ):
        self._image_shape = image_shape
        self._object_extents = object_extents
        self._image_coords = np.stack(
            np.meshgrid(
                np.arange(self._image_shape[0]),
                np.arange(self._image_shape[1]),
                indexing="ij",
            ),
            axis=-1,
        )

    def _get_num_channels(self) -> int:
        return 1

    def _draw_object(self, img: np.ndarray, position: np.ndarray, label: int):
        if label == 0:
            # Rectangle
            img[
                (position[0] - self._object_extents / 2 <= self._image_coords[:, :, 0])
                & (
                    self._image_coords[:, :, 0]
                    <= position[0] + self._object_extents / 2
                )
                & (
                    position[1] - self._object_extents / 2
                    <= self._image_coords[:, :, 1]
                )
                & (
                    self._image_coords[:, :, 1]
                    <= position[1] + self._object_extents / 2
                )
            ] = 1.0
        else:
            # Circle
            img[
                np.linalg.norm(position - self._image_coords, axis=-1)
                <= self._object_extents / 2
            ] = 1.0

    def _pack(self, vals: Sequence[int]) -> int:
        multiplier = 1
        value_packed = 0
        for val, max_val in zip(vals, self._get_max_vals()):
            value_packed += val * multiplier
            multiplier *= max_val
        return value_packed

    def _unpack(self, value_packed: int) -> Sequence[int]:
        remainder = value_packed
        vals = []
        for max_val in self._get_max_vals():
            val = remainder % max_val
            vals.append(val)
            remainder = (remainder - val) // max_val
        return vals

    @abstractmethod
    def _get_max_vals(self) -> Sequence[int]: ...

    def _get_length(self) -> int:
        return int(np.prod(self._get_max_vals()))


class CircleSquareDataset(BaseCircleSquareDataset):
    def __init__(
        self,
        show_gradient: bool = True,
        image_shape: tuple[int, int] = (28, 28),
        object_extents: int = 8,
    ):
        super().__init__(image_shape=image_shape, object_extents=object_extents)
        self._show_gradient = show_gradient

    def _get_max_vals(self) -> Sequence[int]:
        return [2, self._image_shape[1], self._image_shape[0]]

    def _get_num_classes(self) -> int:
        return 2

    def _get_data_point(self, idx: int) -> tuple[np.ndarray, int]:
        position, label = self.get_object_position_and_label(idx)
        max_dist = np.sqrt(np.sum(np.array(self._image_shape) ** 2))

        if self._show_gradient:
            img = 1 - np.linalg.norm(position - self._image_coords, axis=-1) / max_dist
        else:
            img = np.zeros(self._image_shape)
        self._draw_object(img, position, label)
        return img[:, :, None], label

    def get_object_position_and_label(
        self, idx: int | np.ndarray
    ) -> tuple[np.ndarray, int]:
        label, pos_x, pos_y = self._unpack(idx)
        return np.stack([pos_y, pos_x], axis=-1), label


class DoubleCircleSquareDataset(BaseCircleSquareDataset):
    def __init__(
        self,
        show_gradient_a: bool = True,
        show_gradient_b: bool = True,
        image_shape: tuple[int, int] = (28, 28),
        object_extents: int = 8,
    ):
        super().__init__(image_shape=image_shape, object_extents=object_extents)
        self._show_gradient_a = show_gradient_a
        self._show_gradient_b = show_gradient_b
        coords = self._image_coords.reshape((-1, 2))

        coord_pairs = np.stack(
            np.broadcast_arrays(coords[:, None], coords[None, :]), axis=-2
        ).reshape((-1, 2, 2))

        coord_pair_valid = (
            (np.abs(coord_pairs[:, 0] - coord_pairs[:, 1]) >= object_extents + 1).any(
                axis=-1
            )
            # Ensure no symmetric duplicates
            & (coord_pairs[:, 0, 0] <= coord_pairs[:, 1, 0])
            & (
                (coord_pairs[:, 0, 0] < coord_pairs[:, 1, 0])
                | (coord_pairs[:, 0, 1] <= coord_pairs[:, 1, 1])
            )
        )

        self.__positions = coord_pairs[coord_pair_valid]

    def _get_num_classes(self) -> int:
        return 3

    def _get_data_point(self, idx: int) -> tuple[np.ndarray, int]:
        label_1, label_2, pos_idx = self._unpack(idx)
        pos_1, pos_2 = self.__positions[pos_idx]
        max_dist = np.sqrt(np.sum(np.array(self._image_shape) ** 2))

        coords = np.stack(
            np.meshgrid(
                np.arange(self._image_shape[0]),
                np.arange(self._image_shape[1]),
                indexing="ij",
            ),
            axis=-1,
        )
        img = (
            1
            - np.minimum(
                np.linalg.norm(pos_1 - coords, axis=-1) * self._show_gradient_a,
                np.linalg.norm(pos_2 - coords, axis=-1) * self._show_gradient_b,
            )
            / max_dist
        )
        for pos, label in [(pos_1, label_1), (pos_2, label_2)]:
            self._draw_object(img, pos, label)
        if label_1 == label_2:
            label = label_1
        else:
            label = 2
        return img[:, :, None], label

    def _get_max_vals(self) -> Sequence[int]:
        return [2, 2, len(self.__positions)]
