from __future__ import annotations

from typing import SupportsInt, Iterable

import PIL.Image
import aiohttp
import numpy as np
from datasets import load_dataset, Dataset, ClassLabel

from .image_classification_dataset import ImageClassificationDataset


class HuggingfaceImageClassificationDataset(ImageClassificationDataset):
    def __init__(
        self,
        dataset_name: str,
        channels: int = 3,
        split: str = "train",
        image_feature_name: str = "image",
        label_feature_name: str = "label",
        filter_labels: Iterable[str] | None = None,
    ):
        self.__dataset_name = dataset_name
        self.__split = split
        self.__train_split = self.__data = None
        self.__image_feature_name = image_feature_name
        self.__label_feature_name = label_feature_name
        self.__channels = channels
        self.__filter_labels = None if filter_labels is None else list(filter_labels)

    def __filter_dataset(self, dataset: Dataset) -> Dataset:
        label_names = self.__train_split.features[self.__label_feature_name].names
        label_idx = [label_names.index(l) for l in self.__filter_labels]
        mapping = {v: i for i, v in enumerate(label_idx)}
        dataset_filtered = dataset.select(
            np.where(
                (
                    np.array(dataset[self.__label_feature_name])[:, None] == label_idx
                ).any(axis=-1)
            )[0]
        ).map(
            lambda d: {
                **d,
                self.__label_feature_name: mapping[d[self.__label_feature_name]],
            }
        )
        new_features = dataset_filtered.features.copy()
        new_features[self.__label_feature_name] = ClassLabel(names=self.__filter_labels)
        return dataset_filtered.cast(new_features)

    def load(self):
        dataset = load_dataset(
            self.__dataset_name,
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=60 * 60 * 6)}
            },
        )
        self.__data = dataset[self.__split]
        self.__train_split = dataset["train"]
        if self.__filter_labels is not None:
            self.__data = self.__filter_dataset(self.__data)
            self.__train_split = self.__filter_dataset(self.__train_split)

    def _get_num_classes(self) -> int:
        return self.__train_split.features[self.__label_feature_name].num_classes

    def _get_num_channels(self) -> int:
        return self.__channels

    def _get_length(self) -> int:
        return len(self.__data)

    def _get_data_point(
        self, idx: int
    ) -> tuple[np.ndarray | PIL.Image.Image, SupportsInt]:
        data_point = self.__data[idx]
        return (
            data_point[self.__image_feature_name],
            data_point[self.__label_feature_name],
        )
