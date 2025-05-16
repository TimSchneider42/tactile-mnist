from __future__ import annotations

import inspect
from abc import abstractmethod, ABC
from functools import lru_cache, partial
from typing import (
    TypeVar,
    Generic,
    Literal,
    Iterable,
    Callable,
    Any,
)

import datasets
import numpy as np

from .dataset import Dataset

try:
    import torch
    import torchvision.io.image
except ImportError:
    torch = torchvision = None


class HuggingfaceDatapoint:
    def __init__(
        self,
        get_columns_fn: Callable[[str], datasets.Dataset],
        column_names: Iterable[str],
        index: int,
    ):
        annotations = {}
        conversion_fns = {}
        for base in reversed(type(self).__mro__):
            annotations.update(getattr(base, "__annotations__", {}))
            conversion_fns.update(getattr(base, "__dict__", {}))
        self.__fields = {k: v for k, v in annotations.items()}
        self.__column_names = tuple(column_names)
        self.__annotations = annotations
        self.__conversion_fns = conversion_fns
        self.__fetch_value_cached = lru_cache(maxsize=None)(self.__fetch_value)
        self.__get_columns_fn = get_columns_fn
        self.__index = index

    @staticmethod
    def __unflatten_dict(d: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for k, v in d.items():
            keys = k.split(".")
            current = result
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                assert isinstance(current, dict)
                current = current[key]
            current[keys[-1]] = v
        return result

    def __fetch_value(self, item: str):
        if item not in self.__annotations:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'"
            )
        item_type = self.__annotations[item]
        if inspect.isclass(item_type) and issubclass(item_type, HuggingfaceDatapoint):
            value = item_type(
                get_columns_fn=lambda col: self.__get_columns_fn(f"{item}.{col}"),
                index=self.__index,
            )
        else:
            if item in self.__column_names:
                value = self.__get_columns_fn(item)[self.__index][item]
            else:
                value = self.__unflatten_dict(
                    {
                        k[len(item) + 1 :]: self.__get_columns_fn(k)[self.__index][k]
                        for k in self.__column_names
                        if k.startswith(f"{item}.")
                    }
                )
        if item in self.__conversion_fns:
            if "datapoint" in inspect.signature(self.__conversion_fns[item]).parameters:
                value = self.__conversion_fns[item](value, datapoint=self)
            else:
                value = self.__conversion_fns[item](value)
        return value

    def __getattribute__(self, item: str):
        if item == f"_HuggingfaceDatapoint__fields" or item not in self.__fields:
            return super().__getattribute__(item)
        return self.__fetch_value_cached(item)

    def __dir__(self):
        return [*super().__dir__(), *self.__fields.keys()]


DataPointType = TypeVar("DataPointType", bound=HuggingfaceDatapoint)
SelfType = TypeVar("SelfType", bound="HuggingfaceDataset")


class HuggingfaceDataset(
    Dataset[DataPointType, "HuggingfaceDataset[DataPointType, SelfType]"],
    Generic[DataPointType, SelfType],
    ABC,
):
    def __init__(
        self,
        huggingface_dataset: datasets.Dataset,
        cache_size: int | Literal["full"] = 0,
    ):
        super().__init__(cache_size=cache_size)
        self.__huggingface_dataset = huggingface_dataset
        self.__get_columns_fn_cached = lru_cache(maxsize=None)(
            self.__huggingface_dataset.select_columns
        )

    def __iter__(self) -> Iterable[DataPointType]:
        if isinstance(self.__huggingface_dataset, datasets.Dataset):
            # It is more efficient to iterate by index if we are not streaming
            yield from super().__iter__()
        else:
            # Streaming mode
            for dp in self.__huggingface_dataset:
                yield self._get_data_point_type()(
                    lambda col, _dp=dp: [{col: _dp[col]}],
                    self.__huggingface_dataset.column_names,
                    0,
                )

    @abstractmethod
    def _get_data_point_type(self) -> type[DataPointType]:
        pass

    def _get_item(self, index: int) -> DataPointType:
        return self._get_data_point_type()(
            self.__get_columns_fn_cached, self.__huggingface_dataset.column_names, index
        )

    def _select(self, indices: np.ndarray) -> SelfType:
        return type(self)(
            self.__huggingface_dataset.select(indices), cache_size=self.cache_size
        )

    def _get_length(self) -> int:
        return len(self.__huggingface_dataset)

    def filter_labels(self, labels: int | Iterable[int]) -> DataPointType:
        if not isinstance(labels, Iterable):
            labels = [labels]
        labels = np.asarray(list(set(labels)))
        actual_labels = np.asarray(self.__huggingface_dataset["label"])
        return self[np.any(labels[None] == actual_labels[:, None], axis=1)]

    @property
    def by_labels(self) -> tuple[SelfType, ...]:
        labels = np.asarray(self.__huggingface_dataset["label"])
        return tuple(self[labels == label] for label in range(len(self.label_names)))

    @property
    def label_names(self) -> tuple[str, ...]:
        return tuple(self.__huggingface_dataset.features["label"].names)

    @property
    def huggingface_dataset(self) -> datasets.Dataset:
        return self.__huggingface_dataset
