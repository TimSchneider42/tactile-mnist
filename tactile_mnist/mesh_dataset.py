from __future__ import annotations

from abc import ABC
from typing import TypeVar, Generic

from trimesh import Trimesh

from .huggingface_dataset import HuggingfaceDataset, HuggingfaceDatapoint


class MeshDataPoint(HuggingfaceDatapoint):
    id: int | str
    label: int
    mesh: Trimesh


DatapointType = TypeVar("DatapointType", bound=MeshDataPoint)
SelfType = TypeVar("SelfType", bound="MeshDataset")


class MeshDataset(
    HuggingfaceDataset[MeshDataPoint, SelfType], Generic[DatapointType, SelfType], ABC
):
    pass
