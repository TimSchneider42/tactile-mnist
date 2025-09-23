from __future__ import annotations

import trimesh
from trimesh import Trimesh

from tactile_mnist import MeshDataset
from .huggingface_dataset import DataPointType
from .mesh_dataset import MeshDataPoint


class SimpleMeshDataPoint(MeshDataPoint):
    mesh: Trimesh = lambda d: trimesh.Trimesh(**d)


class SimpleMeshDataset(MeshDataset[SimpleMeshDataPoint, "SimpleMeshDataset"]):
    def _get_data_point_type(self) -> type[DataPointType]:
        return SimpleMeshDataPoint
