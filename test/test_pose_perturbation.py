"""Tests for the object pose perturbation of TactilePerceptionVectorEnv.

The perturbation moves the object before each touch (except the first touch of an episode), so the touch observes the
pose that the next prediction is scored against: the targets of a step are computed at its start, from the pose of the
previous touch.
"""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import datasets
import numpy as np
import pytest
import trimesh

from tactile_mnist import (
    SimpleMeshDataset,
    TactileClassificationVectorEnv,
    TactilePerceptionConfig,
)

NUM_ENVS = 2
NUM_OBJECTS = 8
STEP_LIMIT = 4
NUM_STEPS = 12


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


def _rollout(dataset: SimpleMeshDataset):
    """Roll a perturbed environment and record the object pose at every touch, the pose at the start of every step
    (which the targets of that step are computed from), and which environments started a new episode at each touch.
    """
    env = TactileClassificationVectorEnv(
        TactilePerceptionConfig(
            dataset,
            step_limit=STEP_LIMIT,
            sensor_output_size=(32, 32),
            allow_sensor_rotation=False,
            sensor_type="depth",
            sensor_backend="numpy",
            perturb_object_pose=True,
            translation_perturbation_scale=0.002,
            rotation_perturbation_scale=0.05,
        ),
        NUM_ENVS,
    )
    touch_poses = []
    original_touch = env.touch

    def recording_touch(sensor_target_poses):
        touch_poses.append(env.current_object_poses_platform_frame.matrix.copy())
        return original_touch(sensor_target_poses)

    env.touch = recording_touch
    # Poses right after each partial reset (the initial poses of new episodes), recorded per call
    reset_partial_poses = []
    original_reset_partial = env._TactilePerceptionVectorEnv__reset_partial

    def recording_reset_partial(mask, *args, **kwargs):
        result = original_reset_partial(mask, *args, **kwargs)
        reset_partial_poses.append(
            env.current_object_poses_platform_frame.matrix.copy()
        )
        return result

    env._TactilePerceptionVectorEnv__reset_partial = recording_reset_partial
    env.reset(seed=0)
    reset_poses = env.current_object_poses_platform_frame.matrix.copy()
    env.action_space.seed(0)
    step_start_poses, new_episode = [], [np.ones(NUM_ENVS, dtype=np.bool_)]
    done = np.zeros(NUM_ENVS, dtype=np.bool_)
    for _ in range(NUM_STEPS):
        step_start_poses.append(env.current_object_poses_platform_frame.matrix.copy())
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        # With next-step autoreset, the environments done in the previous step start a new episode in this one
        new_episode.append(done)
        done = terminated | truncated
    env.close()
    # reset() makes one partial reset call and every step one more, at its start
    assert len(reset_partial_poses) == NUM_STEPS + 1
    return (
        reset_poses,
        np.array(touch_poses),
        np.array(step_start_poses),
        np.array(new_episode),
        np.array(reset_partial_poses),
    )


def test_reset_touch_observes_the_initial_pose(dataset):
    reset_poses, touch_poses, _, _, _ = _rollout(dataset)
    np.testing.assert_allclose(touch_poses[0], reset_poses)


def test_targets_are_computed_from_the_pose_of_the_previous_touch(dataset):
    _, touch_poses, step_start_poses, _, _ = _rollout(dataset)
    # The object does not move between a touch and the start of the next step
    np.testing.assert_allclose(step_start_poses, touch_poses[:-1])


def test_each_touch_after_the_first_observes_a_new_pose(dataset):
    _, touch_poses, _, new_episode, _ = _rollout(dataset)
    moved = np.any(~np.isclose(touch_poses[1:], touch_poses[:-1]), axis=(-1, -2))
    continuing = ~new_episode[1:]
    assert np.any(continuing)
    assert np.all(moved[continuing])


def test_first_touch_after_an_autoreset_is_not_perturbed(dataset):
    _, touch_poses, _, new_episode, reset_partial_poses = _rollout(dataset)
    autoreset = new_episode[1:]
    assert np.any(autoreset)
    # The touch of an environment reset at the start of a step observes exactly the initial pose the reset assigned
    for t, env_idx in zip(*np.nonzero(autoreset)):
        np.testing.assert_allclose(
            touch_poses[t + 1, env_idx], reset_partial_poses[t + 1, env_idx]
        )


def test_each_environment_draws_its_own_perturbation(dataset):
    _, touch_poses, _, new_episode, _ = _rollout(dataset)
    # Relative pose change between two consecutive touches of a continuing episode, per environment
    changes = []
    for t in range(1, len(touch_poses)):
        if not np.any(new_episode[t]):
            changes.append(
                [
                    np.linalg.inv(touch_poses[t - 1, i]) @ touch_poses[t, i]
                    for i in range(NUM_ENVS)
                ]
            )
    assert changes
    changes = np.array(changes)
    # The environments must not all move their objects by the same transformation
    assert not np.allclose(changes[:, 0], changes[:, 1])
