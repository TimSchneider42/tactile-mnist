from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Protocol

from trimesh import Trimesh

from .dataset import Dataset
from .huggingface_dataset import HuggingfaceDataset, HuggingfaceDatapoint


class MeshDataPoint(Protocol):
    """Interface of the data points of a MeshDataset: an id, a class label, and a (lazily loaded) mesh."""

    id: int | str
    label: int
    mesh: Trimesh


DatapointType = TypeVar("DatapointType", bound=MeshDataPoint)
SelfType = TypeVar("SelfType", bound="MeshDataset")


class MeshDataset(
    Dataset[DatapointType, SelfType], Generic[DatapointType, SelfType], ABC
):
    """Base class of datasets of meshes with class labels, regardless of where the meshes come from."""

    @property
    @abstractmethod
    def label_names(self) -> tuple[str, ...]:
        pass


class HuggingfaceMeshDataPoint(HuggingfaceDatapoint):
    id: int | str
    label: int
    mesh: Trimesh


HuggingfaceDatapointType = TypeVar(
    "HuggingfaceDatapointType", bound=HuggingfaceMeshDataPoint
)
HuggingfaceSelfType = TypeVar("HuggingfaceSelfType", bound="HuggingfaceMeshDataset")


class HuggingfaceMeshDataset(
    HuggingfaceDataset[HuggingfaceDatapointType, HuggingfaceSelfType],
    MeshDataset[HuggingfaceDatapointType, HuggingfaceSelfType],
    Generic[HuggingfaceDatapointType, HuggingfaceSelfType],
    ABC,
):
    """Mesh dataset backed by a Huggingface dataset with (at least) an id, a label, and a mesh column."""

    pass
