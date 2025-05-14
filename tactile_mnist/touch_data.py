from __future__ import annotations

import copy
import datetime
from functools import partial
from typing import (
    TypeVar,
    Sequence,
    Generic,
    Callable,
    Any,
)

import PIL.Image
import numpy as np
from transformation import Transformation

from .huggingface_dataset import HuggingfaceDataset, HuggingfaceDatapoint

try:
    import torchvision
except ImportError:
    torchvision = None

InType = TypeVar("InType")
OutType = TypeVar("OutType")


class LazySequence(Sequence[OutType], Generic[InType, OutType]):
    def __init__(self, data: Sequence[InType], transform: Callable[[InType], OutType]):
        self.__data = tuple(data)
        self.__transform = transform
        self.__transformed_data = [None] * len(self.__data)

    def __getitem__(self, index: int) -> OutType:
        if self.__transformed_data[index] is None:
            self.__transformed_data[index] = self.__transform(self.__data[index])
        return self.__transformed_data[index]

    def __len__(self) -> int:
        return len(self.__data)


def mk_transformations_ragged(
    dict_repr: dict[str, list[list[list]]], datapoint: TouchSeq
) -> list[list[Transformation]]:
    return [
        [Transformation.from_pos_quat(p, q) for p, q, _ in zip(pl, ql, range(l))]
        for pl, ql, l in zip(
            dict_repr["position"],
            dict_repr["quaternion"],
            datapoint.video_length_frames,
        )
    ]


def transform_ragged(
    data: list[list[Any]], datapoint: TouchSeq
) -> list[list[Transformation]]:
    return [d[:l] for d, l in zip(data, datapoint.video_length_frames)]


def process_info(info: dict[str, Any]) -> dict[str, Any]:
    info = copy.copy(info)
    for k in list(info.keys()):
        if isinstance(info[k], dict) and info[k].keys() == {"position", "quaternion"}:
            info[k] = Transformation.from_pos_quat(**info[k])
    return info


class TouchData(HuggingfaceDatapoint):
    id: str
    label: int
    object_id: int
    info: dict[str, Any] = process_info
    pos_in_cell: np.ndarray = np.asarray


class TouchSeq(TouchData):
    video_length_frames: list[int]
    gel_pose_cell_frame_seq: list[list[Transformation]] = mk_transformations_ragged
    time_stamp_rel_seq: list[list[datetime.timedelta]] = transform_ragged
    touch_start_time_rel: list[datetime.timedelta]
    touch_end_time_rel: list[datetime.timedelta]
    sensor_video: list[torchvision.io.VideoReader]


class TouchSingle(TouchData):
    gel_pose_cell_frame: Transformation = lambda d: Transformation.from_pos_quat(**d)
    sensor_image: LazySequence[PIL.Image.Image, np.ndarray] = partial(
        LazySequence, transform=np.array
    )


SelfType = TypeVar("SelfType", bound="TouchDataset")


class TouchSeqDataset(HuggingfaceDataset[TouchSeq, "TouchSeqDataset"]):
    def _get_data_point_type(self) -> type[TouchSeq]:
        return TouchSeq


class TouchSingleDataset(HuggingfaceDataset[TouchSingle, "TouchSingleDataset"]):
    def _get_data_point_type(self) -> type[TouchSingle]:
        return TouchSingle
