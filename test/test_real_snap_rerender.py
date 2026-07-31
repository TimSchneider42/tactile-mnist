"""Tests for the re-rendering variants of TactileRealSnapVectorEnv.

If TactileRealSnapConfig.sensor_type is set, the recorded tactile images are not returned as they are. Instead, a
depth map is estimated from every image with a DepthEstimator and rendered with the corresponding TactileRenderer,
which maps the recordings into the domain of the simulated environments.
"""

import numpy as np
import pytest
from datasets import load_dataset

from tactile_mnist import (
    GELSIGHT_MINI_GEL_THICKNESS_MM,
    GEL_PENETRATION_DEPTH_MM,
    TactileRealSnapClassificationVectorEnv,
    TactileRealSnapConfig,
    TouchSingleDataset,
)
from tactile_mnist.depth_estimator import mk_depth_estimator
from tactile_mnist.depth_estimator.factory import DEPTH_ESTIMATOR_FACTORIES
from tactile_mnist.tactile_renderer.factory import ModuleNotLoaded

NUM_ENVS = 2
OUTPUT_SIZE = (64, 64)

AVAILABLE_BACKENDS = [
    backend
    for backend, factory in DEPTH_ESTIMATOR_FACTORIES["cycle_gan"].items()
    if not isinstance(factory, ModuleNotLoaded)
]


@pytest.fixture(scope="module")
def dataset() -> TouchSingleDataset:
    return TouchSingleDataset(
        load_dataset(
            "TimSchneider42/tactile-mnist-touch-real-single-t256-64x64", split="test"
        )
    )


@pytest.fixture(scope="module")
def estimator():
    return mk_depth_estimator()


def _mk_env(dataset: TouchSingleDataset, **config) -> TactileRealSnapClassificationVectorEnv:
    return TactileRealSnapClassificationVectorEnv(
        TactileRealSnapConfig(
            dataset,
            mesh_dataset=None,
            enable_rendering=False,
            sensor_output_size=OUTPUT_SIZE,
            **config,
        ),
        NUM_ENVS,
    )


@pytest.mark.parametrize(
    "sensor_type,channels", [("cycle_gan", 3), ("taxim", 3), ("depth", 1)]
)
def test_rerendered_observations_match_the_observation_space(
    dataset, sensor_type, channels
):
    env = _mk_env(dataset, sensor_type=sensor_type)
    assert env.single_observation_space["sensor_img"].shape == (*OUTPUT_SIZE, channels)
    obs, _ = env.reset(seed=0)
    env.action_space.seed(0)
    for _ in range(3):
        assert obs["sensor_img"].shape == (NUM_ENVS, *OUTPUT_SIZE, channels)
        assert env.observation_space["sensor_img"].contains(obs["sensor_img"])
        obs, _, _, _, _ = env.step(env.action_space.sample())
    env.close()


def test_rerendering_changes_the_observations(dataset):
    """The re-rendered images must differ from the recordings, but still depend on the touch."""
    recorded = _mk_env(dataset)
    rerendered = _mk_env(dataset, sensor_type="cycle_gan")
    assert recorded.sensor is None and recorded.depth_estimator is None
    assert rerendered.sensor is not None and rerendered.depth_estimator is not None

    obs_recorded, _ = recorded.reset(seed=0)
    obs_rerendered, _ = rerendered.reset(seed=0)
    # Both environments select the same touches, so the observations correspond to the same recorded images
    assert not np.allclose(obs_recorded["sensor_img"], obs_rerendered["sensor_img"])

    rerendered.action_space.seed(0)
    obs_next, _, _, _, _ = rerendered.step(rerendered.action_space.sample())
    assert not np.allclose(obs_rerendered["sensor_img"], obs_next["sensor_img"])
    recorded.close()
    rerendered.close()


def test_estimated_depth_stays_within_the_gel(estimator, dataset):
    images = np.stack([np.asarray(dataset[0].sensor_image[i]) for i in range(4)])
    depth = estimator.estimate(images, (128, 128))
    assert depth.shape == (4, 128, 128)
    assert depth.dtype == np.float32
    assert np.all(depth >= GEL_PENETRATION_DEPTH_MM / 1000 - 1e-9)
    assert np.all(depth <= GELSIGHT_MINI_GEL_THICKNESS_MM / 1000 + 1e-9)
    # The touches are not identical, so neither are their depth maps
    assert np.std(depth, axis=0).max() > 0


def test_estimator_accepts_uint8_and_float_images(estimator, dataset):
    images = np.stack([np.asarray(dataset[0].sensor_image[i]) for i in range(2)])
    depth_uint8 = estimator.estimate(images, (64, 64))
    depth_float = estimator.estimate(images.astype(np.float32) / 255, (64, 64))
    assert depth_uint8 == pytest.approx(depth_float, abs=1e-6)


@pytest.mark.skipif(
    len(AVAILABLE_BACKENDS) < 2, reason="requires both the torch and the jax backend"
)
def test_depth_estimator_backends_agree(dataset):
    images = np.stack([np.asarray(dataset[0].sensor_image[i]) for i in range(2)])
    depth = [
        mk_depth_estimator(backend=backend).estimate(images, (256, 256))
        for backend in AVAILABLE_BACKENDS
    ]
    # 1e-6 m is far below the resolution of the sensor and comes down to float32 rounding
    assert depth[0] == pytest.approx(depth[1], abs=1e-6)
