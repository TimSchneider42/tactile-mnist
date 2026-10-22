"""Tests for the sensor height limit derived from the object heights (tactile_perception_vector_env.get_sensor_z_max)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import tactile_mnist.util as util
from tactile_mnist import StarstruckMeshDataset
from tactile_mnist.constants import (
    GELSIGHT_MINI_GEL_THICKNESS_MM,
    GELSIGHT_MINI_SENSOR_SURFACE_SIZE,
)
from tactile_mnist.starstruck_dataset import OBJECT_HEIGHT
from tactile_mnist.tactile_perception_vector_env import (
    get_max_object_height,
    get_sensor_z_max,
)

GEL_THICKNESS = GELSIGHT_MINI_GEL_THICKNESS_MM / 1000


def _sensor_height_above_contact(rotation: Rotation) -> float:
    """
    Height of the sensor position above the lowest point of its gel surface (brute force over the gel corners). The
    untilted sensor points down, its gel surface lies GEL_THICKNESS below the sensor position.
    """
    half = GELSIGHT_MINI_SENSOR_SURFACE_SIZE / 2
    corners = np.array(
        [
            [x, y, -GEL_THICKNESS]
            for x in (-half[0], half[0])
            for y in (-half[1], half[1])
        ]
    )
    return float(-np.min(rotation.apply(corners)[:, 2]))


def test_untilted_sensor_adds_the_gel_thickness():
    assert get_sensor_z_max(0.01) == pytest.approx(0.01 + GEL_THICKNESS)


@pytest.mark.parametrize("max_tilt_angle", [np.pi / 8, np.pi / 4, np.pi / 2])
def test_bound_covers_every_tilt_and_is_tight(max_tilt_angle):
    rng = np.random.default_rng(0)
    heights = []
    for _ in range(2000):
        axis_angle = rng.uniform(0, 2 * np.pi)
        tilt = rng.uniform(0, max_tilt_angle)
        axis = np.array([np.cos(axis_angle), np.sin(axis_angle), 0.0])
        heights.append(_sensor_height_above_contact(Rotation.from_rotvec(tilt * axis)))
    bound = get_sensor_z_max(0.0, max_tilt_angle)
    assert max(heights) <= bound + 1e-12
    # The worst case (tilting towards a corner by the largest angle) comes close to the bound
    assert max(heights) == pytest.approx(bound, rel=2e-2)


def test_max_object_height_of_flat_objects(tmp_path, monkeypatch):
    monkeypatch.setattr(util, "CACHE_BASE_DIR", tmp_path)
    dataset = StarstruckMeshDataset(size=12)
    assert get_max_object_height([dataset, dataset]) == pytest.approx(OBJECT_HEIGHT)
    # Smallest dimension up keeps flat scenes flat and is cached separately
    assert get_max_object_height(
        [dataset], smallest_dimension_up=True
    ) == pytest.approx(OBJECT_HEIGHT)
    assert len(list((tmp_path / "object_height").glob("*.npz"))) == 2
