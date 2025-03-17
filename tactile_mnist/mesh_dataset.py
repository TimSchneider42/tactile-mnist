from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Literal

import trimesh
from trimesh import Trimesh

from .dataset import Dataset, PartialDataPoint


@dataclass(frozen=True)
class MeshMetadata(PartialDataPoint["MeshDataPoint"]):
    id: int
    label: int
    _mesh_path: Path

    @classmethod
    def load(cls, path: Path) -> List["MeshMetadata"]:
        mesh_path = path / "meshes"
        with (path / "labels.csv").open("r") as f:
            reader = csv.DictReader(f)
            return [
                MeshMetadata(
                    int(e["id"]), int(e["label"]), mesh_path / f"{e['id']}.stl"
                )
                for e in reader
            ]

    def load_full(self) -> MeshDataPoint:
        mesh = trimesh.load(self._mesh_path)
        return MeshDataPoint(self, mesh)


@dataclass(frozen=True)
class MeshDataPoint:
    metadata: MeshMetadata
    mesh: Trimesh


class MeshDataset(
    Dataset[MeshMetadata, MeshDataPoint, "MeshDataset[MeshMetadata, MeshDataPoint]"]
):
    def by_labels(self) -> dict[int, MeshDataset]:
        return {
            l: self[[i for i, e in enumerate(self.partial) if e.label == l]]
            for l in self.labels
        }

    def filter_labels(self, labels: int | Iterable[int]) -> MeshDataset:
        if not isinstance(labels, Iterable):
            labels = [labels]
        labels = set(labels)
        return self[[i for i, e in enumerate(self.partial) if e.label in labels]]

    @property
    def labels(self) -> Set[int]:
        return set(e.label for e in self.partial)

    @property
    def metadata(self) -> tuple[MeshMetadata, ...]:
        return self.partial

    @staticmethod
    def load(path: Path, cache_size: int | Literal["full"] = 0) -> MeshDataset:
        return MeshDataset(MeshMetadata.load(path), cache_size=cache_size)
