from __future__ import annotations

import ctypes
from abc import ABC
from functools import lru_cache
from itertools import chain
from typing import (
    Sequence,
    Iterable,
    Generic,
    TypeVar,
    Literal,
    Any,
    overload,
    Callable,
)

import numpy as np

FullDataPointType = TypeVar("FullDataPointType")


class PartialDataPoint(Generic[FullDataPointType], ABC):
    def load_full(self, *args, **kwargs) -> FullDataPointType:
        pass


PartialDataPointType = TypeVar("PartialDataPointType", bound=PartialDataPoint)
SubDatasetType = TypeVar("SubDatasetType", bound="Dataset")


class Dataset(
    Sequence[FullDataPointType],
    Generic[PartialDataPointType, FullDataPointType, SubDatasetType],
):
    def __init__(
        self,
        partial_data_points: Iterable[PartialDataPointType],
        load_full_kwargs: dict[str, Any] | Iterable[dict[str, Any]] | None = None,
        cache_size: int | Literal["full"] = 0,
        _cached_get_item_func: Callable[[int, int], FullDataPointType] | None = None,
    ):
        partial_data_points = tuple(partial_data_points)
        if load_full_kwargs is None:
            load_full_kwargs = {}
        if isinstance(load_full_kwargs, dict):
            load_full_kwargs = (load_full_kwargs,) * len(partial_data_points)
        assert len(load_full_kwargs) == len(partial_data_points)
        self.__load_full_kwargs = load_full_kwargs
        self.__partial_data_points = partial_data_points
        if cache_size == "full":
            cache_size = None
        if _cached_get_item_func is None:
            self.__cached_get_item_func = lru_cache(cache_size)(self._get_item_inner)
        else:
            self.__cached_get_item_func = _cached_get_item_func

    @classmethod
    def _instantiate(
        cls,
        partial_data_points: tuple[PartialDataPointType, ...],
        load_full_kwargs: tuple[dict[str, Any], ...],
        cached_get_item_func: Callable[[int, int], FullDataPointType] | None = None,
    ) -> SubDatasetType:
        return cls(
            partial_data_points,
            load_full_kwargs,
            _cached_get_item_func=cached_get_item_func,
        )

    @staticmethod
    def _get_item_inner(
        partial_data_point_id: int, load_full_kwargs_id: int
    ) -> FullDataPointType:
        # Cachable variant of _get_item
        partial_data_point: PartialDataPoint = ctypes.cast(
            partial_data_point_id, ctypes.py_object
        ).value
        load_full_kwargs: dict[str, Any] = ctypes.cast(
            load_full_kwargs_id, ctypes.py_object
        ).value
        return partial_data_point.load_full(**load_full_kwargs)

    def _get_item(self, index: int) -> FullDataPointType:
        return self.__cached_get_item_func(
            id(self.__partial_data_points[index]), id(self.__load_full_kwargs[index])
        )

    def concatenate(self, *datasets: Dataset) -> SubDatasetType:
        return self._instantiate(
            tuple(dp for ds in chain((self,), datasets) for dp in ds.partial),
            tuple(dp for ds in chain((self,), datasets) for dp in ds.load_full_kwargs),
        )

    @property
    def partial(self) -> tuple[PartialDataPointType, ...]:
        return self.__partial_data_points

    @property
    def load_full_kwargs(self) -> tuple[dict[str, Any], ...]:
        return self.__load_full_kwargs

    @overload
    def __getitem__(self, index: int) -> FullDataPointType:
        ...

    @overload
    def __getitem__(self, index: slice | Sequence[int] | np.ndarray) -> SubDatasetType:
        ...

    def __getitem__(self, index: int | slice | Sequence[int]):
        if isinstance(index, slice):
            return self._instantiate(
                self.partial[index],
                self.__load_full_kwargs[index],
                cached_get_item_func=self.__cached_get_item_func,
            )
        elif isinstance(index, Sequence) or isinstance(index, np.ndarray):
            return self._instantiate(
                tuple(self.partial[i] for i in index),
                tuple(self.__load_full_kwargs[i] for i in index),
                cached_get_item_func=self.__cached_get_item_func,
            )
        else:
            if -len(self) <= index < len(self):
                return self._get_item(index)
            else:
                raise IndexError(
                    f"Index {index} is out of bounds for data set of size {len(self)}."
                )

    def __add__(
        self, other: Dataset[PartialDataPointType, FullDataPointType]
    ) -> Dataset[PartialDataPointType, FullDataPointType]:
        return self.concatenate(other)

    def __len__(self) -> int:
        return len(self.__partial_data_points)
