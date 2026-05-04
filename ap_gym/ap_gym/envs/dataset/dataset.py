from __future__ import annotations

from abc import ABC, abstractmethod
from typing import SupportsInt, Sequence, overload, Generic, TypeVar

import numpy as np

DataPointType = TypeVar("DataPointType")
DataPointBatchType = TypeVar("DataPointBatchType")


class Dataset(ABC, Generic[DataPointType, DataPointBatchType]):
    def load(self):
        pass

    @abstractmethod
    def _get_length(self) -> int:
        pass

    @abstractmethod
    def get_data_point(self, idx: SupportsInt) -> DataPointType:
        pass

    @abstractmethod
    def get_data_point_batch(
        self, idx: Sequence[SupportsInt] | np.ndarray
    ) -> DataPointBatchType:
        pass

    @overload
    def __getitem__(self, item: SupportsInt) -> DataPointType: ...

    @overload
    def __getitem__(self, item: Sequence[SupportsInt]) -> DataPointBatchType: ...

    def __getitem__(self, item: int | Sequence[int] | np.ndarray):
        if isinstance(item, Sequence) or isinstance(item, np.ndarray):
            return self.get_data_point_batch(item)
        else:
            return self.get_data_point(item)

    def __len__(self):
        return self._get_length()
