from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from trimesh import Trimesh

from .dataset import Dataset
from .huggingface_dataset import HuggingfaceDataset, HuggingfaceDatapoint


class MeshDataPoint:
    """
    Base class of the data points of a MeshDataset: an id, a class label, and a (lazily loaded) mesh. Subclasses
    provide the three attributes in whatever way suits them (dataclass fields, Huggingface columns, properties).
    """

    id: int | str
    label: int
    mesh: Trimesh

    @property
    def volume(self) -> float:
        """Volume of the mesh (in m^3). Subclasses that know the volume without the mesh may override this."""
        return self.mesh.volume


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

    @property
    def cache_key(self) -> str:
        """
        Identifies the content of the dataset (its data points, in order) in the caches of values derived from it
        (see util.cached_arrays). Composed of the prefix of the dataset type and a key unique among the datasets of
        that type, so that datasets of different types never collide.
        """
        return f"{self._cache_key_prefix()}_{self._cache_key_body()}"

    @classmethod
    @abstractmethod
    def _cache_key_prefix(cls) -> str:
        """Prefix of the cache keys of this type of dataset."""
        pass

    @abstractmethod
    def _cache_key_body(self) -> str:
        """Key of the content of this dataset, unique among the datasets of its type."""
        pass


class HuggingfaceMeshDataPoint(HuggingfaceDatapoint, MeshDataPoint):
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

    def _cache_key_body(self) -> str:
        return self.huggingface_dataset._fingerprint
