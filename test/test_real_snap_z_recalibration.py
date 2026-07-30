"""Tests for the sensor z recalibration of TactileRealSnapVectorEnv.

The recorded gel z-positions of the real touch datasets are offset by an unknown per-round constant. The environment
re-zeroes them on the deepest touch of each round, which corresponds to touching the platform, and aligns them with
the convention of the simulated environments (touching the platform yields a sensor z of GEL_PENETRATION_DEPTH_MM).
"""

import numpy as np
import pytest
from datasets import load_dataset

from tactile_mnist import (
    GEL_PENETRATION_DEPTH_MM,
    TactileRealSnapClassificationVectorEnv,
    TactileRealSnapConfig,
    TouchSingleDataset,
)

NUM_ENVS = 4
NUM_STEPS = 24


@pytest.fixture(scope="module")
def dataset() -> TouchSingleDataset:
    return TouchSingleDataset(
        load_dataset(
            "TimSchneider42/tactile-mnist-touch-real-single-t256-64x64", split="train"
        )
    )


def _rollout(
    dataset: TouchSingleDataset, recalibrate: bool, num_steps: int = NUM_STEPS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = TactileRealSnapClassificationVectorEnv(
        TactileRealSnapConfig(
            dataset,
            mesh_dataset=None,
            enable_rendering=False,
            recalibrate_sensor_z=recalibrate,
        ),
        NUM_ENVS,
    )
    obs, info = env.reset(seed=0)
    env.action_space.seed(0)
    observed_z = [obs["sensor_pos"][:, 2]]
    sensor_z = [info["sensor_pose"].translation[:, 2]]
    for _ in range(num_steps):
        obs, _, _, _, info = env.step(env.action_space.sample())
        observed_z.append(obs["sensor_pos"][:, 2])
        sensor_z.append(info["sensor_pose"].translation[:, 2])
    offsets = env.current_sensor_z_offsets
    env.close()
    return np.concatenate(observed_z), np.concatenate(sensor_z), offsets


def test_recalibration_zeroes_z_on_the_deepest_touch_of_the_round(dataset):
    env = TactileRealSnapClassificationVectorEnv(
        TactileRealSnapConfig(dataset, mesh_dataset=None, enable_rendering=False),
        NUM_ENVS,
    )
    env.reset(seed=0)
    for offset, dp in zip(env.current_sensor_z_offsets, env.current_data_points):
        recorded_min = np.min(dp.gel_pose_cell_frame.translation[:, 2])
        assert recorded_min + offset == pytest.approx(GEL_PENETRATION_DEPTH_MM / 1000)
    # The rounds were recorded with different gel states, so their offsets differ
    assert np.std(env.current_sensor_z_offsets) > 0
    env.close()


def test_recalibrated_z_never_goes_below_the_simulated_platform_height(dataset):
    """The simulated environments report GEL_PENETRATION_DEPTH_MM when the sensor touches the platform."""
    _, sensor_z, _ = _rollout(dataset, recalibrate=True)
    platform_z = GEL_PENETRATION_DEPTH_MM / 1000
    assert np.min(sensor_z) >= platform_z - 1e-9
    # Touching the platform does happen, so the lower end of the range is actually reached
    assert np.min(sensor_z) < platform_z + 1e-3


def test_recalibration_only_shifts_z(dataset):
    """With the same actions, the recalibration selects the same touches and only shifts their z-positions."""
    # Fewer steps than the step limit, so that the environments stay in their initial rounds
    num_steps = 10
    _, sensor_z_raw, _ = _rollout(dataset, recalibrate=False, num_steps=num_steps)
    _, sensor_z_recalibrated, offsets = _rollout(
        dataset, recalibrate=True, num_steps=num_steps
    )
    # The offset is constant within a round
    shift = (sensor_z_recalibrated - sensor_z_raw).reshape(-1, NUM_ENVS)
    np.testing.assert_allclose(shift, np.broadcast_to(offsets, shift.shape), atol=1e-9)


def test_recalibration_can_be_disabled(dataset):
    env = TactileRealSnapClassificationVectorEnv(
        TactileRealSnapConfig(
            dataset,
            mesh_dataset=None,
            enable_rendering=False,
            recalibrate_sensor_z=False,
        ),
        NUM_ENVS,
    )
    _, info = env.reset(seed=0)
    np.testing.assert_array_equal(env.current_sensor_z_offsets, np.zeros(NUM_ENVS))
    for i, dp in enumerate(env.current_data_points):
        touch_idx = env.current_touch_indices[i]
        np.testing.assert_allclose(
            info["sensor_pose"].translation[i],
            dp.gel_pose_cell_frame[int(touch_idx)].translation,
        )
    env.close()
