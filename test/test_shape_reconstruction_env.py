"""Tests for TactileShapeReconstructionVectorEnv.

The COD-VAE encoder/decoder must never be JIT-compiled after __init__ when the JAX backend is used, as compiling
inside a host callback deadlocks (see TactilePerceptionVectorEnv.__init__). Since JAX recompiles for every new batch
size, the environment pads/expands all encode and decode batches to num_envs and warms both paths up in __init__.
"""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import datasets
import numpy as np
import pytest
import trimesh

from tactile_mnist import (
    SimpleMeshDataset,
    TactilePerceptionConfig,
    TactileShapeReconstructionVectorEnv,
)

NUM_ENVS = 2
NUM_OBJECTS = 4
STEP_LIMIT = 2
NUM_STEPS = 6

JIT_FNS = ("_jit_encode", "_jit_decode_planes", "_jit_decode_logits")


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
        ),
        NUM_ENVS,
        backend="jax",
        renderer_show_shadow_objects=True,
        shadow_object_resolution=64,
    )
    try:
        assert env.vae.backend == "jax"
        after_init = _jit_cache_sizes(env.vae)
        assert all(size == 1 for size in after_init.values()), after_init

        # Roll through several episodes (multiple resets) and check nothing recompiles.
        env.reset(seed=0)
        env.action_space.seed(0)
        for _ in range(NUM_STEPS):
            env.step(env.action_space.sample())
            assert _jit_cache_sizes(env.vae) == after_init

        # Force the encode padding path: sync the latent cache to the current poses, then evict one entry so
        # only a single latent is missing. The padded batch must neither recompile nor change the result.
        env._get_prediction_targets()
        latent_cache = env._TactileShapeReconstructionVectorEnv__latent_cache
        assert len(latent_cache) == NUM_ENVS
        evicted_key = next(iter(latent_cache))
        evicted_latent, _ = latent_cache.pop(evicted_key)
        env._get_prediction_targets()
        assert _jit_cache_sizes(env.vae) == after_init
        latent_cache = env._TactileShapeReconstructionVectorEnv__latent_cache
        np.testing.assert_allclose(
            latent_cache[evicted_key][0], evicted_latent, rtol=1e-5, atol=1e-6
        )
    finally:
        env.close()
