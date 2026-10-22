"""Tests for the procedurally generated PrimitiveShapes dataset."""

import collections

import gymnasium as gym
import numpy as np
import pytest

import tactile_mnist  # noqa: F401 (registers the environments)
from tactile_mnist import MeshDataset, PrimitiveShapesMeshDataset
from tactile_mnist.primitive_shapes_dataset import (
    DEFAULT_SIZE,
    LABEL_NAMES,
    MIN_TRIANGLE_ANGLE,
    triangle_angles,
)

NUM_SAMPLES = 400


def _shape_from_mesh(mesh) -> int:
    """The class of a primitive judged from its mesh alone (vertex count, footprint proportions and fill)."""
    extents = mesh.extents
    footprint = mesh.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
    fill = footprint.to_2D()[0].area / (extents[0] * extents[1])
    if len(mesh.vertices) == 8:
        return 1
    if len(mesh.vertices) == 6:
        return 2
    if fill > 0.75 and extents[0] == pytest.approx(extents[1]):
        return 0
    if fill > 0.75 and extents[0] == pytest.approx(2 * extents[1]):
        return 3
    return -1


@pytest.fixture(scope="module")
def samples():
    dataset = PrimitiveShapesMeshDataset()
    indices = np.random.default_rng(0).choice(len(dataset), NUM_SAMPLES, replace=False)
    return [dataset[int(i)] for i in indices]


def test_default_size_and_labels():
    dataset = PrimitiveShapesMeshDataset()
    assert isinstance(dataset, MeshDataset)
    assert len(dataset) == DEFAULT_SIZE
    assert dataset.label_names == LABEL_NAMES
    assert [dataset[i].label for i in range(8)] == [0, 1, 2, 3] * 2
    np.testing.assert_array_equal(dataset.labels[:8], [0, 1, 2, 3] * 2)
    assert all(
        {sub[j].label for j in range(10)} == {label}
        for label, sub in enumerate(dataset.by_labels)
    )


def test_meshes_match_their_labels(samples):
    confusion = collections.Counter(
        (dp.label, _shape_from_mesh(dp.mesh)) for dp in samples
    )
    assert all(label == shape for label, shape in confusion), confusion
    assert {label for label, _ in confusion} == set(range(len(LABEL_NAMES)))


def test_meshes_are_closed_and_centered(samples):
    for dp in samples:
        assert dp.mesh.is_watertight and dp.mesh.is_winding_consistent
        center = dp.mesh.bounds.mean(axis=0)
        if dp.triangle is not None:
            # Prisms are centered on the centroid of their triangle instead of their bounding box
            center[:2] = dp.triangle.mean(axis=0)
        np.testing.assert_allclose(center, 0, atol=1e-9)
        np.testing.assert_allclose(dp.mesh.extents, dp.dimensions, rtol=1e-6)


def test_volume_matches_the_mesh_without_building_it(samples):
    for dp in samples:
        # The cylinder meshes are polygonal approximations and slightly smaller than the exact volume
        assert dp.mesh.volume == pytest.approx(dp.volume, rel=2e-3)
    dp = PrimitiveShapesMeshDataset(size=1)[0]
    dp.volume
    assert "mesh" not in vars(dp)


def test_triangles_have_no_slivers(samples):
    for dp in samples:
        if dp.triangle is not None:
            sides = np.linalg.norm(
                dp.triangle - np.roll(dp.triangle, -1, axis=0), axis=1
            )
            assert np.all(triangle_angles(sides) >= MIN_TRIANGLE_ANGLE - 1e-9)


def test_objects_are_deterministic_and_distinct():
    reference = PrimitiveShapesMeshDataset()[7]
    np.testing.assert_array_equal(
        PrimitiveShapesMeshDataset()[7].dimensions, reference.dimensions
    )
    for other in [
        PrimitiveShapesMeshDataset()[11],
        PrimitiveShapesMeshDataset(seed=1)[7],
    ]:
        assert not np.array_equal(other.dimensions, reference.dimensions)


def test_cache_keys_identify_the_content():
    dataset = PrimitiveShapesMeshDataset(size=16)
    assert dataset.cache_key.startswith("primitive-shapes_")
    assert dataset.cache_key == PrimitiveShapesMeshDataset(size=16).cache_key
    assert (
        len(
            {
                dataset.cache_key,
                PrimitiveShapesMeshDataset(size=16, seed=1).cache_key,
                PrimitiveShapesMeshDataset(size=17).cache_key,
                dataset[:4].cache_key,
            }
        )
        == 4
    )


def test_environments_are_registered_without_test_split():
    ids = {i for i in gym.registry if i.startswith("PrimitiveShapes")}
    for task in ["PrimitiveShapes", "PrimitiveShapesVolume"]:
        for variant in ["", "-DR", "-Depth", "-Depth-DR"]:
            assert f"{task}{variant}-v0" in ids
    assert not any("-test-" in i or "-train-" in i for i in ids)
