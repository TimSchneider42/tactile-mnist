"""Procedurally generated PrimitiveShapes mesh dataset.

Every object is one of four geometric primitives, a cylinder, a cuboid, an extruded triangle (a triangular prism), or a
half cylinder, with randomly drawn dimensions, so objects of the same class differ in size and proportions. Like the Starstruck
dataset, the objects are derived on the fly from their index with a stable pseudo random number generator instead of
being loaded from a stored dataset. The train and test splits are drawn from separate seed streams.

The meshes are centered on the origin in their canonical orientation, in which they also rest on the platform:
cylinders and half cylinders stand on their base and prisms lie on their triangular face (all axes along z). The
rotation about the vertical axis is left to the environment.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

import numpy as np
import shapely.geometry
import trimesh
from trimesh import Trimesh

from .mesh_dataset import MeshDataPoint, MeshDataset

LABEL_NAMES = ("cylinder", "cuboid", "triangular prism", "half cylinder")

# Object geometry (in meters). Every free parameter is drawn independently and uniformly from its range: the radius and
# height of the (half) cylinders, the three side lengths of the cuboids, and the three side lengths and the extrusion
# height of the triangular prisms. The footprints are large relative to the 120 mm cell, so the objects are easy to find
# and the challenge lies in telling the shapes apart; the smallest footprints of all classes are similar in area
# (about 19 cm^2), and the largest ones still fit into the cell with the default object placement margin for any rotation
# (rotated extent <= 92 mm of the 100 mm available). The heights stay within the range of the ABC objects.
CYLINDER_RADIUS_RANGE = (0.025, 0.0325)
CUBOID_SIDE_RANGE = (0.045, 0.065)
TRIANGLE_SIDE_RANGE = (0.065, 0.080)
# Half cylinders are only half as deep as they are wide, hence the larger radius
HALF_CYLINDER_RADIUS_RANGE = (0.035, 0.040)
HEIGHT_RANGE = (0.020, 0.040)
# Smallest interior angle of the triangles; side length triples violating it are rejected and redrawn, which avoids
# slivers that are hard to tell apart from thin cuboids
MIN_TRIANGLE_ANGLE = np.deg2rad(25)
CYLINDER_SECTIONS = 64

SPLITS = ("train", "test")
# 100,000 objects per class in every split
DEFAULT_SPLIT_SIZES = {"train": 400_000, "test": 400_000}
DEFAULT_SEED = 0


def triangle_angles(sides: np.ndarray) -> np.ndarray:
    """Interior angles opposite the given side lengths (law of cosines); NaN if the sides form no triangle."""
    a, b, c = sides
    with np.errstate(invalid="ignore"):
        return np.arccos(
            np.array(
                [
                    (b**2 + c**2 - a**2) / (2 * b * c),
                    (a**2 + c**2 - b**2) / (2 * a * c),
                    (a**2 + b**2 - c**2) / (2 * a * b),
                ]
            )
        )


def sample_triangle(rng: np.random.Generator) -> np.ndarray:
    """
    Triangle (3x2 vertices, counter-clockwise, centroid at the origin) with side lengths drawn independently from
    TRIANGLE_SIDE_RANGE, redrawn until all angles are >= MIN_TRIANGLE_ANGLE.
    """
    while True:
        sides = rng.uniform(*TRIANGLE_SIDE_RANGE, size=3)
        angles = triangle_angles(sides)
        if np.all(angles >= MIN_TRIANGLE_ANGLE):
            break
    return triangle_from_sides(sides)


def triangle_from_sides(sides: np.ndarray) -> np.ndarray:
    """Triangle vertices (3x2, counter-clockwise, centroid at the origin) with the given side lengths."""
    a, b, c = sides
    alpha = triangle_angles(sides)[0]
    # Side c along the x-axis, the vertex opposite c at distance b from the origin under the angle alpha
    vertices = np.array([[0.0, 0.0], [c, 0.0], [b * np.cos(alpha), b * np.sin(alpha)]])
    return vertices - vertices.mean(axis=0)


def make_primitive_mesh(
    label: int, dimensions: np.ndarray, triangle: np.ndarray | None
) -> Trimesh:
    """Mesh of a primitive in its canonical orientation, centered on the origin."""
    if label == 0:
        mesh = trimesh.creation.cylinder(
            radius=dimensions[0] / 2, height=dimensions[2], sections=CYLINDER_SECTIONS
        )
    elif label == 1:
        mesh = trimesh.creation.box(extents=dimensions)
    elif label == 2:
        mesh = trimesh.creation.extrude_polygon(
            shapely.geometry.Polygon(triangle), dimensions[2]
        )
        mesh.apply_translation([0, 0, -dimensions[2] / 2])
    elif label == 3:
        angles = np.linspace(0, np.pi, CYLINDER_SECTIONS // 2 + 1)
        semicircle = (
            dimensions[0] / 2 * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        )
        mesh = trimesh.creation.extrude_polygon(
            shapely.geometry.Polygon(semicircle), dimensions[2]
        )
        # Center the bounding box on the origin (the semicircle spans y in [0, radius])
        mesh.apply_translation([0, -dimensions[1] / 2, -dimensions[2] / 2])
    else:
        raise ValueError(f"Unknown label {label}.")
    return mesh


@dataclass
class PrimitiveShapesMeshDataPoint(MeshDataPoint):
    id: int
    label: int
    # Extents along x, y, z in the canonical orientation (for cylinders, x = y = diameter, z = height; for prisms,
    # x and y are the extents of the triangle and z is the extrusion length; for half cylinders, x = diameter,
    # y = radius, z = height)
    dimensions: np.ndarray
    # Triangle vertices of a prism (3x2, in meters), None for the other classes
    triangle: np.ndarray | None = None

    @property
    def class_name(self) -> str:
        return LABEL_NAMES[self.label]

    @property
    def volume(self) -> float:
        """Exact volume of the primitive (the cylinder mesh is a polygonal approximation and slightly smaller)."""
        return _primitive_volume(self.label, self.dimensions, self.triangle)

    @cached_property
    def mesh(self) -> Trimesh:
        return make_primitive_mesh(self.label, self.dimensions, self.triangle)


def _primitive_volume(
    label: int, dimensions: np.ndarray, triangle: np.ndarray | None
) -> float:
    if label == 0:
        return float(np.pi * (dimensions[0] / 2) ** 2 * dimensions[2])
    if label == 1:
        return float(np.prod(dimensions))
    if label == 3:
        return float(np.pi * (dimensions[0] / 2) ** 2 * dimensions[2] / 2)
    (x0, y0), (x1, y1), (x2, y2) = triangle
    area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    return float(area * dimensions[2])


def _sample_datapoint(
    index: int, seed: int, split: str = "train"
) -> PrimitiveShapesMeshDataPoint:
    rng = np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(SPLITS.index(split), index))
    )
    label = index % len(LABEL_NAMES)
    triangle = None
    if label == 0:
        diameter = 2 * rng.uniform(*CYLINDER_RADIUS_RANGE)
        dimensions = np.array([diameter, diameter, rng.uniform(*HEIGHT_RANGE)])
    elif label == 1:
        dimensions = np.concatenate(
            [rng.uniform(*CUBOID_SIDE_RANGE, size=2), [rng.uniform(*HEIGHT_RANGE)]]
        )
    elif label == 2:
        triangle = sample_triangle(rng)
        length = float(rng.uniform(*HEIGHT_RANGE))
        dimensions = np.concatenate([np.ptp(triangle, axis=0), [length]])
    else:
        radius = rng.uniform(*HALF_CYLINDER_RADIUS_RANGE)
        dimensions = np.array([2 * radius, radius, rng.uniform(*HEIGHT_RANGE)])
    return PrimitiveShapesMeshDataPoint(
        id=index, label=label, dimensions=dimensions, triangle=triangle
    )


class PrimitiveShapesMeshDataset(
    MeshDataset[PrimitiveShapesMeshDataPoint, "PrimitiveShapesMeshDataset"]
):
    """
    PrimitiveShapes mesh dataset whose objects are generated on the fly.

    Datapoint i of a split has label i % 4, so every split is balanced; its dimensions come from a generator seeded
    with SeedSequence(seed, spawn_key=(split index, i)). The meshes are only generated when a datapoint's mesh field is accessed and
    are cached on the datapoint afterwards.
    """

    def __init__(
        self,
        split: str = "train",
        size: int | None = None,
        seed: int = DEFAULT_SEED,
        cache_size: int = 0,
        _indices: np.ndarray | None = None,
    ):
        super().__init__(cache_size=cache_size)
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}, expected one of {SPLITS}.")
        self.__split = split
        self.__size = DEFAULT_SPLIT_SIZES[split] if size is None else size
        self.__seed = seed
        self.__indices = _indices

    def _get_item(self, index: int) -> PrimitiveShapesMeshDataPoint:
        if self.__indices is not None:
            index = int(self.__indices[index])
        return _sample_datapoint(index, self.__seed, self.__split)

    def _select(self, indices: np.ndarray) -> PrimitiveShapesMeshDataset:
        indices = np.asarray(indices)
        if self.__indices is not None:
            indices = self.__indices[indices]
        return PrimitiveShapesMeshDataset(
            self.__split,
            self.__size,
            self.__seed,
            cache_size=self.cache_size,
            _indices=indices,
        )

    def _get_length(self) -> int:
        return self.__size if self.__indices is None else len(self.__indices)

    @property
    def label_names(self) -> tuple[str, ...]:
        return LABEL_NAMES

    @property
    def labels(self) -> np.ndarray:
        indices = np.arange(self.__size) if self.__indices is None else self.__indices
        return indices % len(LABEL_NAMES)

    def filter_labels(self, labels: int | Iterable[int]) -> PrimitiveShapesMeshDataset:
        if not isinstance(labels, Iterable):
            labels = [labels]
        return self[np.isin(self.labels, list(labels))]

    @property
    def by_labels(self) -> tuple[PrimitiveShapesMeshDataset, ...]:
        labels = self.labels
        return tuple(self[labels == label] for label in range(len(LABEL_NAMES)))

    @property
    def split(self) -> str:
        return self.__split

    @property
    def seed(self) -> int:
        return self.__seed

    @classmethod
    def _cache_key_prefix(cls) -> str:
        return "primitive-shapes"

    def _cache_key_body(self) -> str:
        key = f"{self.__split}_seed{self.__seed}_size{self.__size}"
        if self.__indices is not None:
            indices_hash = hashlib.sha256(
                np.asarray(self.__indices, dtype=np.int64).tobytes()
            ).hexdigest()[:16]
            key += f"_select{indices_hash}"
        return key
