"""Procedurally generated Starstruck mesh dataset.

Every scene of the Starstruck dataset consists of one to three stars and up to five distractors (boxes and cylinders)
scattered over the cell. Instead of shipping a fixed set of pre-computed scenes, every scene is derived on the fly
from its index with a stable pseudo random number generator, so the dataset can be large enough that an agent cannot
memorize the object arrangements, at no storage or start-up cost. The scene layout mirrors the process that produced
the static version of the Starstruck dataset (mesh_datasets/starstruck/generate_mesh_dataset.py of
tactile-mnist-generation). The only difference is how the per-scene metadata is drawn: that script shuffled
a balanced list of star counts and drew the distractor counts and a base seed from one generator up front, whereas
here the star count cycles through the labels with the index and the distractor count and layout seed of scene i come
from a generator seeded with SeedSequence(seed, spawn_key=(split, i)), so no scene depends on any other.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Iterable

import numpy as np
import trimesh
from transformation import Transformation
from trimesh import Trimesh

from .constants import CELL_SIZE, CELL_PADDING
from .mesh_dataset import MeshDataset

# Object geometry of the static version of the Starstruck dataset (in meters)
STAR_RADIUS_OUTER = 0.01
STAR_RADIUS_INNER = 0.005
STAR_JAG_COUNT = 5
OBJECT_HEIGHT = 0.005

MIN_STARS = 1
MAX_STARS = 3
MIN_DISTRACTORS = 0
MAX_DISTRACTORS = 5

LABEL_NAMES = tuple(str(n) for n in range(MIN_STARS, MAX_STARS + 1))

SPLITS = ("train", "test")
DEFAULT_SPLIT_SIZES = {"train": 300_000, "test": 300_000}
DEFAULT_SEED = 0

# Objects are placed such that a circle of the outer star radius around their center lies inside the padded cell
VALID_CELL_SIZE = CELL_SIZE - CELL_PADDING - 2 * STAR_RADIUS_OUTER
MIN_CENTER_DISTANCE = 2 * STAR_RADIUS_OUTER


def make_star_mesh(
    radius_outer: float = STAR_RADIUS_OUTER,
    radius_inner: float = STAR_RADIUS_INNER,
    jag_count: int = STAR_JAG_COUNT,
    height: float = OBJECT_HEIGHT,
) -> Trimesh:
    """Extrude a star polygon with jag_count jags from z = 0 to z = height, centered on the XY origin."""
    angles_outer = np.linspace(0, 2 * np.pi, jag_count, endpoint=False)
    angles_inner = angles_outer + np.pi / jag_count
    points_outer = radius_outer * np.stack(
        [np.cos(angles_outer), np.sin(angles_outer)], axis=-1
    )
    points_inner = radius_inner * np.stack(
        [np.cos(angles_inner), np.sin(angles_inner)], axis=-1
    )
    # Counter-clockwise outline alternating between outer and inner points
    outline = np.stack([points_outer, points_inner], axis=1).reshape(-1, 2)
    n = len(outline)

    # Vertices: outline at the bottom (0..n-1) and top (n..2n-1), followed by the bottom and top center
    bottom = np.concatenate([outline, np.zeros((n, 1))], axis=-1)
    top = np.concatenate([outline, np.full((n, 1), height)], axis=-1)
    vertices = np.concatenate([bottom, top, [[0, 0, 0], [0, 0, height]]], axis=0)
    center_bottom, center_top = 2 * n, 2 * n + 1

    # The star is star-shaped w.r.t. its center, so fans from the center triangulate both caps
    i = np.arange(n)
    j = (i + 1) % n
    faces = np.concatenate(
        [
            np.stack([np.full(n, center_top), n + i, n + j], axis=-1),
            np.stack([np.full(n, center_bottom), j, i], axis=-1),
            np.stack([i, j, n + j], axis=-1),
            np.stack([i, n + j, n + i], axis=-1),
        ],
        axis=0,
    )
    return Trimesh(vertices=vertices, faces=faces, process=False)


@lru_cache(maxsize=1)
def _base_objects() -> tuple[Trimesh, tuple[Trimesh, ...]]:
    star = make_star_mesh()
    box_side_length = np.sqrt(2 * STAR_RADIUS_OUTER**2)
    box = trimesh.creation.box(
        extents=[box_side_length, box_side_length, OBJECT_HEIGHT]
    )
    cylinder = trimesh.creation.cylinder(
        radius=STAR_RADIUS_OUTER, height=OBJECT_HEIGHT, sections=32
    )
    objects = [star, box, cylinder]
    for obj in objects:
        obj.apply_translation([0, 0, -np.min(obj.vertices[:, 2])])
    return star, (box, cylinder)


def generate_starstruck_scene(
    seed: int, star_count: int, distractor_count: int
) -> Trimesh:
    """
    Lay out a Starstruck scene deterministically from its seed.

    The stars are placed first, followed by the distractors, each at a uniformly random position within the cell and
    with a uniformly random rotation about the z-axis. Positions closer than twice the outer star radius to an
    already placed object are rejected; if an object cannot be placed within 100 attempts, the whole scene is laid
    out again.
    """
    star, distractors = _base_objects()
    rng = np.random.default_rng(seed)
    distractor_indices = rng.choice(len(distractors), distractor_count, replace=True)
    objects_to_place = [star] * star_count + [
        distractors[i] for i in distractor_indices
    ]
    scene = []
    positions = []
    while len(scene) < len(objects_to_place):
        scene = []
        positions = []
        for obj in objects_to_place:
            attempts = 0
            transform = None
            pos_arr = np.array(positions).reshape((-1, 2))
            while transform is None and attempts < 100:
                angle = rng.random() * np.pi * 2
                pos = (2 * rng.random(2) - 1) * VALID_CELL_SIZE / 2
                if np.all(np.linalg.norm(pos_arr - pos, axis=1) > MIN_CENTER_DISTANCE):
                    transform = Transformation.from_pos_euler(
                        position=[pos[0], pos[1], 0], euler_angles=[0, 0, angle]
                    )
                else:
                    attempts += 1
            if transform is None:
                break
            obj = obj.copy()
            obj.apply_transform(transform.matrix)
            scene.append(obj)
            positions.append(transform.translation[:2])
    return trimesh.util.concatenate(scene)


@dataclass
class StarstruckMeshDataPoint:
    id: int
    label: int
    seed: int
    distractor_count: int

    @property
    def star_count(self) -> int:
        return MIN_STARS + self.label

    @cached_property
    def mesh(self) -> Trimesh:
        return generate_starstruck_scene(
            self.seed, self.star_count, self.distractor_count
        )


class StarstruckMeshDataset(
    MeshDataset[StarstruckMeshDataPoint, "StarstruckMeshDataset"]
):
    """
    Starstruck mesh dataset whose scenes are generated on the fly.

    Datapoint i of a split has label i % 3 (i.e. 1 + i % 3 stars), so every split is balanced; its distractor count
    and scene seed come from a generator seeded with SeedSequence(seed, spawn_key=(split index, i)). The meshes are
    only generated when a datapoint's mesh field is accessed and are cached on the datapoint afterwards.
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

    def _get_item(self, index: int) -> StarstruckMeshDataPoint:
        if self.__indices is not None:
            index = int(self.__indices[index])
        meta_rng = np.random.default_rng(
            np.random.SeedSequence(
                self.__seed, spawn_key=(SPLITS.index(self.__split), index)
            )
        )
        return StarstruckMeshDataPoint(
            id=index,
            label=index % len(LABEL_NAMES),
            distractor_count=int(
                meta_rng.integers(MIN_DISTRACTORS, MAX_DISTRACTORS + 1)
            ),
            seed=int(meta_rng.integers(0, 2**63)),
        )

    def _select(self, indices: np.ndarray) -> StarstruckMeshDataset:
        indices = np.asarray(indices)
        if self.__indices is not None:
            indices = self.__indices[indices]
        return StarstruckMeshDataset(
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

    def filter_labels(self, labels: int | Iterable[int]) -> StarstruckMeshDataset:
        if not isinstance(labels, Iterable):
            labels = [labels]
        return self[np.isin(self.labels, list(labels))]

    @property
    def by_labels(self) -> tuple[StarstruckMeshDataset, ...]:
        labels = self.labels
        return tuple(self[labels == label] for label in range(len(LABEL_NAMES)))

    @property
    def split(self) -> str:
        return self.__split

    @property
    def seed(self) -> int:
        return self.__seed
