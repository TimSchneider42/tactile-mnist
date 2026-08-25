from __future__ import annotations

import trimesh
from trimesh import Trimesh

from .huggingface_dataset import DataPointType
from .mesh_dataset import HuggingfaceMeshDataPoint, HuggingfaceMeshDataset


class SimpleMeshDataPoint(HuggingfaceMeshDataPoint):
    mesh: Trimesh = lambda d: trimesh.Trimesh(**d)


class SimpleMeshDataset(
    HuggingfaceMeshDataset[SimpleMeshDataPoint, "SimpleMeshDataset"]
):
    def _get_data_point_type(self) -> type[DataPointType]:
        return SimpleMeshDataPoint
