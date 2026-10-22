from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import (
    Sequence,
    Generic,
    TypeVar,
    Literal,
    overload,
)

import numpy as np

DataPointType = TypeVar("DataPointType")
SelfType = TypeVar("SelfType", bound="Dataset")


class Dataset(Sequence[DataPointType], Generic[DataPointType, SelfType], ABC):
    def __init__(
        self,
        cache_size: int | Literal["full"] = 0,
    ):
        self.__cache_size = cache_size
        self.__init_item_cache()

    def __init_item_cache(self):
        self.__cached_get_item_func = lru_cache(
            None if self.__cache_size == "full" else self.__cache_size
        )(self._get_item)

    def __getstate__(self) -> dict:
        # The item cache cannot be pickled (e.g. to send the dataset to a worker process) and is rebuilt empty instead
        state = self.__dict__.copy()
        del state["_Dataset__cached_get_item_func"]
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        self.__init_item_cache()

    @abstractmethod
    def _get_item(self, index: int) -> DataPointType:
        pass

    @abstractmethod
    def _select(self, indices: np.ndarray) -> SelfType:
        pass

    @abstractmethod
    def _get_length(self) -> int:
        pass

    @overload
    def __getitem__(self, index: int) -> DataPointType: ...

    @overload
    def __getitem__(
        self, index: slice | list[int] | list[bool] | np.ndarray
    ) -> SelfType: ...

    def __getitem__(self, index: int | slice | list[int] | list[bool]):
        if isinstance(index, slice):
            return self[np.arange(*index.indices(len(self)))]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            indices = np.asarray(index)
            if indices.dtype == bool:
                indices = np.where(indices)[0]
            return self._select(indices)
        else:
            index = int(index)
            if -len(self) <= index < len(self):
                if index < 0:
                    index += len(self)
                return self.__cached_get_item_func(index)
            else:
                raise IndexError(
                    f"Index {index} is out of bounds for data set of size {len(self)}."
                )

    def __len__(self) -> int:
        return self._get_length()

    @property
    def cache_size(self) -> int | Literal["full"]:
        return self.__cache_size
