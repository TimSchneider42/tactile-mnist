from __future__ import annotations

import trimesh
from trimesh import Trimesh

from .huggingface_dataset import HuggingfaceDataset, HuggingfaceDatapoint, DataPointType


class MeshDataPoint(HuggingfaceDatapoint):
    id: int
    label: int
    mesh: Trimesh = lambda d: trimesh.Trimesh(vertices=d["vertices"], faces=d["faces"])


class MeshDataset(HuggingfaceDataset[MeshDataPoint, "MeshDataset"]):
    def _get_data_point_type(self) -> type[DataPointType]:
        return MeshDataPoint
