"""Tests for TactileShapeReconstructionVectorEnv.

The COD-VAE decoder must never be JIT-compiled after __init__ when the JAX backend is used, as compiling inside a
host callback deadlocks (see TactilePerceptionVectorEnv.__init__). Since JAX recompiles for every new batch size,
the environment pads/expands all decode batches to num_envs (with query batches padded to a fixed chunk size) and
warms both decode functions up in __init__. The encoder is only used during __init__, by the prediction target
statistics computation (which, when the statistics are not cached on disk yet, also compiles decode variants for
its own batch size); nothing may compile after __init__.
"""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import datasets
import numpy as np
import pytest
import trimesh
from cod_vae import pack_cube_transform, points_to_cube_transform
from scipy.spatial.transform import Rotation

from tactile_mnist import (
    SimpleMeshDataset,
    TactilePerceptionConfig,
    TactileShapeReconstructionVectorEnv,
)

NUM_ENVS = 2
NUM_OBJECTS = 4
STEP_LIMIT = 2
NUM_STEPS = 6

JIT_FNS = ("_jit_decode_planes", "_jit_decode_logits", "_jit_decode_logits_full")


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


def _jit_cache_sizes(vae) -> dict[str, int]:
    return {name: getattr(vae, name)._cache_size() for name in JIT_FNS}


def test_jax_vae_does_not_compile_after_init(dataset):
    env = TactileShapeReconstructionVectorEnv(
        TactilePerceptionConfig(
            dataset,
            step_limit=STEP_LIMIT,
            sensor_output_size=(32, 32),
            allow_sensor_rotation=False,
            sensor_type="depth",
            sensor_backend="numpy",
            smallest_dimension_up=True,
        ),
        NUM_ENVS,
        backend="jax",
        renderer_show_shadow_objects=True,
        shadow_object_resolution=64,
        loss_fn_kwargs=dict(
            num_vol_queries=256,
            vol_pool_size=2048,
            vol_database_size=16384,
            num_near_points=256,
        ),
    )
    try:
        assert env.vae.backend == "jax"
        # The decode functions may have been compiled for the statistics pass' batch
        # size on top of the warmup's num_envs (only when the statistics were not
        # cached; same for the encoder, which only the statistics pass uses).
        after_init = _jit_cache_sizes(env.vae)
        assert all(size >= 1 for size in after_init.values()), after_init
        encoder_cache_after_init = env.vae._jit_encode._cache_size()

        # Roll through several episodes (multiple resets) and check nothing recompiles.
        env.reset(seed=0)
        env.action_space.seed(0)
        for _ in range(NUM_STEPS):
            _, _, _, _, info = env.step(env.action_space.sample())
            assert _jit_cache_sizes(env.vae) == after_init
            assert env.vae._jit_encode._cache_size() == encoder_cache_after_init
            assert np.all(np.isfinite(info["prediction"]["loss"]))

        # The targets identify the ground-truth geometry.
        targets = env._get_prediction_targets()
        assert set(targets) == {"mesh_index", "position", "quaternion", "box"}
        assert targets["mesh_index"].shape == (NUM_ENVS,)
        assert np.all((0 <= targets["mesh_index"]) & (targets["mesh_index"] < NUM_OBJECTS))
        assert targets["position"].shape == (NUM_ENVS, 3)
        assert targets["quaternion"].shape == (NUM_ENVS, 4)
        assert targets["box"].shape == (NUM_ENVS, 4)
        np.testing.assert_allclose(
            np.linalg.norm(targets["quaternion"], axis=-1), 1.0, atol=1e-5
        )
        # The target quaternion maps the raw dataset mesh into the platform frame:
        # the smallest-dimension-up pre-processing rotation must be composed into it.
        poses = env.current_object_poses_platform_frame
        for i in range(NUM_ENVS):
            mesh = dataset[int(targets["mesh_index"][i])].mesh
            expected = (
                Rotation.from_quat(poses.quaternion[i])
                * TactileShapeReconstructionVectorEnv._smallest_dimension_up_rotation(
                    mesh
                )
            ).as_quat()
            assert (
                min(
                    np.linalg.norm(targets["quaternion"][i] - expected),
                    np.linalg.norm(targets["quaternion"][i] + expected),
                )
                < 1e-5
            )
            # The target box is the posed object's normalized bounding box, computed
            # with the exact functions COD-VAE's encoding uses.
            posed_vertices = (
                Rotation.from_quat(targets["quaternion"][i]).apply(mesh.vertices)
                + targets["position"][i]
            )
            expected_box = pack_cube_transform(
                points_to_cube_transform(posed_vertices, 0.9),
                frame_half_size=env.frame_half_size,
                object_scale=0.9,
            )
            np.testing.assert_allclose(targets["box"][i], expected_box, atol=1e-5)

        # The prediction space is bounded by the cached per-dataset target
        # statistics, which must contain the observed targets.
        space = env.single_prediction_space
        assert np.all(np.isfinite(space.low)) and np.all(np.isfinite(space.high))
        assert np.all(space.low < space.high)
        stats = env.prediction_target_stats
        assert set(stats) == {"mean", "std", "min", "max"}
        assert all(
            value.shape == (env.vae.full_latent_size,) for value in stats.values()
        )
        assert np.all(space.low < stats["min"]) and np.all(stats["max"] < space.high)
        assert np.all(targets["box"] > space.low[-4:])
        assert np.all(targets["box"] < space.high[-4:])
        # The loss is normalized by the blind guessing expected value, so predicting
        # the target statistics' mean must score around 1 on average.
        rng = np.random.default_rng(0)
        blind_loss = env.loss_fn.numpy(
            np.repeat(stats["mean"][None], NUM_ENVS, axis=0),
            targets,
            (NUM_ENVS,),
            rng=rng,
        )
        assert np.all(blind_loss > 0.1) and np.all(blind_loss < 3.0)
    finally:
        env.close()
