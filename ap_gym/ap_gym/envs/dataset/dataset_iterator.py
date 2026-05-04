from __future__ import annotations

from typing import Iterator, Generic, Any, Sequence

import numpy as np

from .dataset import Dataset, DataPointType, DataPointBatchType


class DatasetIterator(
    Iterator[tuple[DataPointType, np.ndarray]],
    Generic[DataPointType],
):
    def __init__(
        self,
        dataset: Dataset[DataPointType, Any],
        seed: int = 0,
        restrict_indices_to: np.ndarray | Sequence[int] | None = None,
    ):
        self.__dataset = dataset
        self.__rng = np.random.default_rng(seed)
        self.__restrict_indices_to = (
            None if restrict_indices_to is None else np.asarray(restrict_indices_to)
        )

    def __next__(self):
        if self.__restrict_indices_to is not None:
            idx = self.__rng.choice(self.__restrict_indices_to)
        else:
            idx = self.__rng.integers(0, len(self.__dataset))
        data = self.__dataset.get_data_point(idx)
        return data, idx


class DatasetBatchIterator(
    Iterator[tuple[DataPointBatchType, np.ndarray]],
    Generic[DataPointBatchType],
):
    def __init__(
        self,
        dataset: Dataset[Any, DataPointBatchType],
        batch_size: int = 1,
        seed: int = 0,
        restrict_indices_to: np.ndarray | Sequence[int] | None = None,
    ):
        self.__dataset = dataset
        self.__rng = np.random.default_rng(seed)
        self.__batch_size = batch_size
        self.__restrict_indices_to = (
            None if restrict_indices_to is None else np.asarray(restrict_indices_to)
        )

    def __next__(self):
        if self.__restrict_indices_to is not None:
            idx = self.__rng.choice(self.__restrict_indices_to, self.__batch_size)
        else:
            idx = self.__rng.integers(0, len(self.__dataset), self.__batch_size)
        data = self.__dataset.get_data_point_batch(idx)
        return data, idx
