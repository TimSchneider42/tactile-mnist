from __future__ import annotations

import trimesh
from trimesh import Trimesh

from tactile_mnist import MeshDataset
from .huggingface_dataset import DataPointType
from .mesh_dataset import MeshDataPoint


class SimpleMeshDataPoint(MeshDataPoint):
    mesh: Trimesh = lambda d: trimesh.Trimesh(vertices=d["vertices"], faces=d["faces"])


class SimpleMeshDataset(MeshDataset[SimpleMeshDataPoint, "SimpleMeshDataset"]):
    def _get_data_point_type(self) -> type[DataPointType]:
        return SimpleMeshDataPoint
