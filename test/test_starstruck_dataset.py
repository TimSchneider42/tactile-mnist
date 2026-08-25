"""Tests for the procedurally generated Starstruck dataset."""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pytest
import trimesh

import tactile_mnist.starstruck_dataset as starstruck_dataset
from tactile_mnist import (
    MeshDataset,
    PrefetchedDataset,
    StarstruckMeshDataset,
    TactileClassificationVectorEnv,
    TactilePerceptionConfig,
)
from tactile_mnist.starstruck_dataset import (
    DEFAULT_SPLIT_SIZES,
    LABEL_NAMES,
    MAX_DISTRACTORS,
    MAX_STARS,
    MIN_CENTER_DISTANCE,
    MIN_DISTRACTORS,
    MIN_STARS,
    OBJECT_HEIGHT,
    STAR_RADIUS_INNER,
    STAR_RADIUS_OUTER,
    VALID_CELL_SIZE,
    make_star_mesh,
)

NUM_CHECKED_SCENES = 50


@pytest.fixture
def generation_counter(monkeypatch) -> dict[str, int]:
    counter = {"calls": 0}
    original = starstruck_dataset.generate_starstruck_scene

    def counted(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(starstruck_dataset, "generate_starstruck_scene", counted)
    return counter


def test_splits_have_default_sizes_and_labels():
    for split, size in DEFAULT_SPLIT_SIZES.items():
        dataset = StarstruckMeshDataset(split)
        assert isinstance(dataset, MeshDataset)
        assert len(dataset) == size
        assert dataset.label_names == LABEL_NAMES == ("1", "2", "3")
    assert len(StarstruckMeshDataset("train", size=10)) == 10
    with pytest.raises(ValueError):
        StarstruckMeshDataset("holdout")


def test_metadata_is_derived_from_index():
    dataset = StarstruckMeshDataset("train")
    datapoints = [dataset[i] for i in range(3000)]
    assert [dp.id for dp in datapoints] == list(range(3000))
    assert [dp.label for dp in datapoints] == [
        i % len(LABEL_NAMES) for i in range(3000)
    ]
    assert all(dp.star_count == MIN_STARS + dp.label for dp in datapoints)
    assert all(MIN_STARS <= dp.star_count <= MAX_STARS for dp in datapoints)
    distractor_counts = np.array([dp.distractor_count for dp in datapoints])
    assert distractor_counts.min() == MIN_DISTRACTORS
    assert distractor_counts.max() == MAX_DISTRACTORS
    assert len({dp.seed for dp in datapoints}) == len(datapoints)
    assert dataset[-1].id == len(dataset) - 1
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_scenes_are_deterministic_and_distinct():
    reference = StarstruckMeshDataset("train")[7]
    same = StarstruckMeshDataset("train")[7]
    assert same.seed == reference.seed
    assert np.array_equal(same.mesh.vertices, reference.mesh.vertices)
    for other in [
        StarstruckMeshDataset("train")[8],
        StarstruckMeshDataset("test")[7],
        StarstruckMeshDataset("train", seed=1)[7],
    ]:
        assert other.seed != reference.seed
        assert not np.array_equal(other.mesh.vertices, reference.mesh.vertices)


def test_star_mesh_matches_extruded_polygon():
    shapely = pytest.importorskip("shapely")
    angles_outer = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    angles_inner = angles_outer + np.pi / 5
    points = np.stack(
        [
            np.stack(
                [
                    STAR_RADIUS_OUTER * np.cos(angles_outer),
                    STAR_RADIUS_OUTER * np.sin(angles_outer),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    STAR_RADIUS_INNER * np.cos(angles_inner),
                    STAR_RADIUS_INNER * np.sin(angles_inner),
                ],
                axis=-1,
            ),
        ],
        axis=1,
    ).reshape(-1, 2)
    reference = trimesh.creation.extrude_polygon(
        shapely.geometry.Polygon(points), OBJECT_HEIGHT
    )
    star = make_star_mesh()
    assert star.is_watertight and star.is_winding_consistent
    assert np.isclose(star.volume, reference.volume)
    assert np.isclose(star.area, reference.area)
    assert np.allclose(star.bounds, reference.bounds)


def test_scene_geometry():
    star, distractors = starstruck_dataset._base_objects()
    volumes = np.array([obj.volume for obj in (star, *distractors)])
    assert not np.any(
        np.isclose(volumes[:, None], volumes[None])[~np.eye(3, dtype=bool)]
    )
    dataset = StarstruckMeshDataset("train")
    for dp in [dataset[i] for i in range(NUM_CHECKED_SCENES)]:
        mesh = dp.mesh
        assert mesh.is_watertight
        assert np.isclose(mesh.bounds[0, 2], 0) and np.isclose(
            mesh.bounds[1, 2], OBJECT_HEIGHT
        )
        bodies = mesh.split(only_watertight=True)
        assert len(bodies) == dp.star_count + dp.distractor_count
        assert (
            sum(np.isclose(body.volume, star.volume) for body in bodies)
            == dp.star_count
        )
        # All objects are symmetric about their placement position, so their centers of mass recover it
        centers = np.array([body.center_mass[:2] for body in bodies])
        assert np.all(np.abs(centers) <= VALID_CELL_SIZE / 2 + 1e-9)
        distances = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
        assert np.all(distances[~np.eye(len(bodies), dtype=bool)] > MIN_CENTER_DISTANCE)


def test_subsets():
    dataset = StarstruckMeshDataset("train")
    subset = dataset[[3, 5, 7]]
    assert isinstance(subset, StarstruckMeshDataset) and len(subset) == 3
    assert subset[1].seed == dataset[5].seed and subset[1].id == 5
    assert subset[[2]][0].id == 7
    assert [dp.id for dp in dataset[10:13]] == [10, 11, 12]
    by_labels = dataset.by_labels
    assert [len(ds) for ds in by_labels] == [len(dataset) // len(LABEL_NAMES)] * len(
        LABEL_NAMES
    )
    assert all(
        ds[i].label == label for label, ds in enumerate(by_labels) for i in range(3)
    )
    assert all(dp.label in (0, 2) for dp in dataset.filter_labels([0, 2])[:10])
    assert dataset.filter_labels(1)[4].star_count == 2


def test_mesh_is_generated_once_per_datapoint(generation_counter):
    dataset = StarstruckMeshDataset("train")
    dp = dataset[0]
    assert generation_counter["calls"] == 0
    assert dp.mesh is dp.mesh
    assert generation_counter["calls"] == 1
    with PrefetchedDataset(dataset, load_fn=lambda dp: dp.mesh) as prefetched:
        prefetched.prefetch([1, 2])
        for i in (1, 2):
            dp = prefetched[i]
            assert generation_counter["calls"] == i + 1
            dp.mesh
            assert generation_counter["calls"] == i + 1


def test_env_generates_every_scene_once(generation_counter):
    num_envs = 2
    step_limit = 4
    env = TactileClassificationVectorEnv(
        TactilePerceptionConfig(
            StarstruckMeshDataset("train"),
            step_limit=step_limit,
            sensor_output_size=(32, 32),
            allow_sensor_rotation=False,
            sensor_type="depth",
            sensor_backend="numpy",
        ),
        num_envs,
    )
    try:
        env.reset(seed=0)
        env.action_space.seed(0)
        # The current scene of every environment plus the prefetched scene of its next episode
        expected = 2 * num_envs
        assert generation_counter["calls"] == expected
        for _ in range(step_limit):
            env.step(env.action_space.sample())
        assert generation_counter["calls"] == expected
        ids = [dp.id for dp in env.current_data_points]
        # Autoreset: the prefetched scenes become current and the following ones are prefetched
        env.step(env.action_space.sample())
        assert [dp.id for dp in env.current_data_points] != ids
        assert generation_counter["calls"] == expected + num_envs
    finally:
        env.close()
