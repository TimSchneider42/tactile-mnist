"""Tests for the RNG streams of TactilePerceptionVectorEnv.

The sensor noise draws from generators spawned off np_random on the fly, which does not advance np_random itself.
A seed thus yields the same object sequence and poses in every variant of an environment, regardless of the sensor
noise (the -DR variants) and the tactile renderer.
"""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import datasets
import numpy as np
import pytest
import trimesh

from tactile_mnist import (
    SensorNoiseConfig,
    SimpleMeshDataset,
    TactileClassificationVectorEnv,
    TactilePerceptionConfig,
)

NUM_ENVS = 2
NUM_OBJECTS = 16
STEP_LIMIT = 2
NUM_STEPS = 8


@pytest.fixture(scope="module")
def dataset() -> SimpleMeshDataset:
    meshes = [
        trimesh.creation.box(extents=(0.02, 0.03, 0.005 + 0.001 * i))
        for i in range(NUM_OBJECTS)
    ]
    return SimpleMeshDataset(
        datasets.Dataset.from_dict(
            {
                "id": list(range(NUM_OBJECTS)),
                "label": [i % 2 for i in range(NUM_OBJECTS)],
                "mesh.vertices": [m.vertices.tolist() for m in meshes],
                "mesh.faces": [m.faces.tolist() for m in meshes],
            },
            features=datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "label": datasets.ClassLabel(names=["even", "odd"]),
                    "mesh.vertices": datasets.Sequence(
                        datasets.Sequence(datasets.Value("float64"), length=3)
                    ),
                    "mesh.faces": datasets.Sequence(
                        datasets.Sequence(datasets.Value("int64"), length=3)
                    ),
                }
            ),
        )
    )


def _mk_env(dataset: SimpleMeshDataset, **config) -> TactileClassificationVectorEnv:
    return TactileClassificationVectorEnv(
        TactilePerceptionConfig(
            dataset,
            step_limit=STEP_LIMIT,
            sensor_output_size=(32, 32),
            allow_sensor_rotation=False,
            **{"sensor_type": "depth", "sensor_backend": "numpy", **config},
        ),
        NUM_ENVS,
    )


def _rollout(
    env: TactileClassificationVectorEnv, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll the environment for NUM_STEPS steps (several episodes) with actions sampled from the seeded action space
    and record the object ids, the object poses, and the observed tactile images."""
    ids = []
    poses = []
    images = []
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    while True:
        ids.append([dp.id for dp in env.current_data_points])
        poses.append(env.current_object_poses_platform_frame.matrix.copy())
        images.append(obs["sensor_img"].copy())
        if len(ids) > NUM_STEPS:
            return np.array(ids), np.array(poses), np.array(images)
        obs, _, _, _, _ = env.step(env.action_space.sample())


def _rollout_images_and_positions(
    env: TactileClassificationVectorEnv, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    obs, _ = env.reset(seed=seed)
    all_obs = [obs]
    env.action_space.seed(seed)
    for _ in range(NUM_STEPS):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        all_obs.append(obs)
    return (
        np.array([o["sensor_img"] for o in all_obs]),
        np.array([o["sensor_pos"] for o in all_obs]),
    )


def test_penetration_depth_randomization_only_affects_the_observed_height(dataset):
    """The imprint always presses the gel in by the nominal penetration depth; the randomization only perturbs the
    sensor height the agent observes."""
    plain = _mk_env(dataset)
    randomized = _mk_env(dataset, penetration_depth_reduction_std=0.00065)
    plain_images, plain_pos = _rollout_images_and_positions(plain, seed=7)
    rand_images, rand_pos = _rollout_images_and_positions(randomized, seed=7)
    plain.close()
    randomized.close()
    np.testing.assert_array_equal(plain_images, rand_images)
    np.testing.assert_array_equal(plain_pos[..., :2], rand_pos[..., :2])
    assert np.abs(plain_pos[..., 2] - rand_pos[..., 2]).max() > 1e-4


def test_sensor_noise_does_not_change_the_object_sequence(dataset):
    """The -DR variants only add sensor noise, so they see the same objects in the same poses as the plain ones."""
    plain = _mk_env(dataset)
    dr = _mk_env(dataset, sensor_noise=SensorNoiseConfig())
    plain_ids, plain_poses, plain_images = _rollout(plain, seed=3)
    dr_ids, dr_poses, dr_images = _rollout(dr, seed=3)
    plain.close()
    dr.close()
    np.testing.assert_array_equal(plain_ids, dr_ids)
    np.testing.assert_allclose(plain_poses, dr_poses)
    # The noise itself must still be there
    assert np.abs(plain_images - dr_images).max() > 1e-4
    # Multiple episodes were compared, and not all of them saw the same object
    assert len(np.unique(plain_ids)) > 1


def test_renderer_does_not_change_the_object_sequence(dataset):
    """The per-frame noise draws depend on the channel count of the renderer (1 for depth, 3 for cycle_gan), which
    must not leak into the object selection."""
    pytest.importorskip("torch")
    depth = _mk_env(dataset, sensor_noise=SensorNoiseConfig())
    cycle_gan = _mk_env(
        dataset,
        sensor_noise=SensorNoiseConfig(),
        sensor_type="cycle_gan",
        sensor_backend="torch",
    )
    depth_ids, depth_poses, _ = _rollout(depth, seed=5)
    cycle_gan_ids, cycle_gan_poses, _ = _rollout(cycle_gan, seed=5)
    depth.close()
    cycle_gan.close()
    np.testing.assert_array_equal(depth_ids, cycle_gan_ids)
    np.testing.assert_allclose(depth_poses, cycle_gan_poses)


def test_rollouts_are_reproducible_for_a_given_seed(dataset):
    env = _mk_env(dataset, sensor_noise=SensorNoiseConfig())
    first = _rollout(env, seed=11)
    second = _rollout(env, seed=11)
    env.close()
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_rollouts_are_reproducible_when_np_random_is_re_seeded_directly(dataset):
    """Re-seeding np_random by assigning the property (instead of calling reset(seed=...)) must make rollouts
    reproducible as well, as the environment keeps no RNG state besides np_random itself."""
    env = _mk_env(dataset)

    def rollout_ids() -> np.ndarray:
        env.np_random = np.random.default_rng(13)
        env.reset()
        ids = [[dp.id for dp in env.current_data_points]]
        env.action_space.seed(13)
        for _ in range(NUM_STEPS):
            env.step(env.action_space.sample())
            ids.append([dp.id for dp in env.current_data_points])
        return np.array(ids)

    first = rollout_ids()
    second = rollout_ids()
    env.close()
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) > 1
