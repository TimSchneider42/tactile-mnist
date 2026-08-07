"""Tests for CODVAEReconstructionLossFn."""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import datasets
import numpy as np
import pytest
import trimesh
from cod_vae import CODVAE, pack_cube_transform, points_to_cube_transform
from transformation import Transformation

from tactile_mnist import CODVAEReconstructionLossFn, SimpleMeshDataset

MODEL = "TimSchneider42/cod-vae-4x32"
FRAME_HALF_SIZE = 0.06
OBJECT_SCALE = 0.9
NUM_OBJECTS = 2

# Small pools for speed; the test meshes are watertight primitives.
LOSS_KWARGS = dict(
    frame_half_size=FRAME_HALF_SIZE,
    object_scale=OBJECT_SCALE,
    num_vol_queries=512,
    vol_pool_size=4096,
    vol_database_size=65536,
    num_near_points=512,
)


@pytest.fixture(scope="module")
def dataset() -> SimpleMeshDataset:
    meshes = [
        trimesh.creation.box(extents=(0.03, 0.05, 0.02)),
        trimesh.creation.cylinder(radius=0.015, height=0.04),
    ]
    return SimpleMeshDataset(
        datasets.Dataset.from_dict(
            {
                "id": list(range(NUM_OBJECTS)),
                "label": [0] * NUM_OBJECTS,
                "mesh.vertices": [m.vertices.tolist() for m in meshes],
                "mesh.faces": [m.faces.tolist() for m in meshes],
            },
            features=datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "label": datasets.ClassLabel(names=["obj"]),
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


@pytest.fixture(scope="module", params=["torch", "jax"])
def vae(request):
    pytest.importorskip(request.param)
    return CODVAE.from_pretrained(MODEL, backend=request.param)


@pytest.fixture(scope="module")
def loss_fn(vae, dataset):
    return CODVAEReconstructionLossFn(vae, dataset=dataset, **LOSS_KWARGS)


@pytest.fixture(scope="module")
def poses() -> list[Transformation]:
    return [
        Transformation.from_pos_quat([0.01, -0.02, 0.011], [0, 0, 0, 1]),
        Transformation.from_pos_euler([-0.02, 0.03, 0.02], [0, 0, 0.7]),
    ]


def _normalized_box(posed_mesh: trimesh.Trimesh) -> np.ndarray:
    """The ground-truth "box" target entry, computed exactly like COD-VAE's encoding."""
    return pack_cube_transform(
        points_to_cube_transform(posed_mesh.bounds, OBJECT_SCALE),
        frame_half_size=FRAME_HALF_SIZE,
        object_scale=OBJECT_SCALE,
    )


@pytest.fixture(scope="module")
def targets(dataset, poses) -> dict[str, np.ndarray]:
    boxes = []
    for i, pose in enumerate(poses):
        mesh = dataset[i].mesh.copy()
        mesh.apply_transform(pose.matrix)
        boxes.append(_normalized_box(mesh))
    return {
        "mesh_index": np.arange(NUM_OBJECTS, dtype=np.int64),
        "position": np.stack([p.translation for p in poses]).astype(np.float32),
        "quaternion": np.stack([p.quaternion for p in poses]).astype(np.float32),
        "box": np.stack(boxes).astype(np.float32),
    }


@pytest.fixture(scope="module")
def good_prediction(vae, dataset, poses) -> np.ndarray:
    """Encode the posed ground-truth meshes into full-latent predictions."""
    meshes = []
    for i, pose in enumerate(poses):
        mesh = dataset[i].mesh.copy()
        mesh.apply_transform(pose.matrix)
        meshes.append(mesh)
    return vae.encode_mesh_full(
        meshes, object_scale=OBJECT_SCALE, seed=0, frame_half_size=FRAME_HALF_SIZE
    )


def test_requires_rng(loss_fn, good_prediction, targets):
    with pytest.raises(ValueError, match="rng"):
        loss_fn.numpy(good_prediction, targets, (NUM_OBJECTS,))


def test_numpy_geometry(loss_fn, good_prediction, targets):
    rng = np.random.default_rng(0)
    good = loss_fn.numpy(good_prediction, targets, (NUM_OBJECTS,), rng=rng)
    assert good.shape == (NUM_OBJECTS,)
    assert np.all(np.isfinite(good))
    # The encoding of the true geometry must beat blind guessing by a clear margin.
    # (It does not reach zero: the VAE reconstruction is imperfect, and near-surface
    # labels close to the decision boundary remain ambiguous.)
    assert np.all(good < 0.6 * loss_fn.blind_guessing_expected_value)

    # A garbage latent must score much worse than the encoding of the true geometry.
    bad_latent = good_prediction.copy()
    bad_latent[:, :-4] = np.random.default_rng(1).standard_normal(
        bad_latent[:, :-4].shape
    )
    bad_latent_loss = loss_fn.numpy(bad_latent, targets, (NUM_OBJECTS,), rng=rng)
    assert np.all(bad_latent_loss > 2 * good)

    # A shifted bounding box center must also score much worse: the ground-truth
    # points then decode at the wrong cube locations.
    bad_center = good_prediction.copy()
    bad_center[:, -4] += 0.5
    bad_center_loss = loss_fn.numpy(bad_center, targets, (NUM_OBJECTS,), rng=rng)
    assert np.all(bad_center_loss > 2 * good)

    # And so must a wrong bounding box size.
    bad_size = good_prediction.copy()
    bad_size[:, -1] *= 2.0
    bad_size_loss = loss_fn.numpy(bad_size, targets, (NUM_OBJECTS,), rng=rng)
    assert np.all(bad_size_loss > 2 * good)


def test_numpy_deterministic_given_rng(loss_fn, good_prediction, targets):
    a = loss_fn.numpy(
        good_prediction, targets, (NUM_OBJECTS,), rng=np.random.default_rng(2)
    )
    b = loss_fn.numpy(
        good_prediction, targets, (NUM_OBJECTS,), rng=np.random.default_rng(2)
    )
    np.testing.assert_array_equal(a, b)
    c = loss_fn.numpy(
        good_prediction, targets, (NUM_OBJECTS,), rng=np.random.default_rng(3)
    )
    assert not np.array_equal(a, c)


def test_numpy_unbatched(loss_fn, good_prediction, targets):
    loss = loss_fn.numpy(
        good_prediction[0],
        {key: value[0] for key, value in targets.items()},
        (),
        rng=np.random.default_rng(4),
    )
    assert loss.shape == ()


def test_normalized_wrapper_forwards_rng(loss_fn, good_prediction, targets):
    loss = loss_fn.normalized.numpy(
        good_prediction, targets, (NUM_OBJECTS,), rng=np.random.default_rng(5)
    )
    assert loss.shape == (NUM_OBJECTS,)
    assert np.all(np.isfinite(loss))


def test_backend_variant(vae, loss_fn, good_prediction, targets):
    numpy_reference = loss_fn.numpy(
        good_prediction, targets, (NUM_OBJECTS,), rng=np.random.default_rng(6)
    )
    if vae.backend == "jax":
        import jax

        jitted = jax.jit(
            lambda p, t, key: loss_fn.jax(p, t, (NUM_OBJECTS,), rng=key)
        )
        loss = np.asarray(jitted(good_prediction, targets, jax.random.PRNGKey(0)))
        with pytest.raises(ValueError, match="rng"):
            loss_fn.jax(good_prediction, targets, (NUM_OBJECTS,))
    else:
        import torch

        generator = torch.Generator().manual_seed(0)
        prediction = torch.from_numpy(good_prediction).requires_grad_(True)
        loss_tensor = loss_fn.torch(
            prediction,
            {key: torch.from_numpy(value) for key, value in targets.items()},
            (NUM_OBJECTS,),
            rng=generator,
        )
        loss_tensor.sum().backward()
        assert prediction.grad is not None
        grad = prediction.grad.numpy()
        assert np.all(np.isfinite(grad))
        # The prediction's box equals the target's here, so the box gradient is
        # exactly zero: the decoder's transform gradient is stopped and the MSE term
        # is at its minimum (see test_box_gradient_is_exactly_the_mse_gradient).
        np.testing.assert_array_equal(grad[:, -4:], 0.0)
        loss = loss_tensor.detach().cpu().numpy()
        with pytest.raises(ValueError, match="rng"):
            loss_fn.torch(
                prediction,
                {key: torch.from_numpy(value) for key, value in targets.items()},
                (NUM_OBJECTS,),
            )
    assert loss.shape == (NUM_OBJECTS,)
    assert np.all(np.isfinite(loss))
    # Different rng streams sample different query points, so the losses only need to
    # agree in magnitude.
    np.testing.assert_allclose(loss.mean(), numpy_reference.mean(), atol=0.25)


def test_box_mse_term(vae, dataset, loss_fn, good_prediction, targets):
    # The encoder derives its box entries from the same computation as the target's
    # "box" (points_to_cube_transform + pack_cube_transform), so they must agree.
    np.testing.assert_allclose(good_prediction[:, -4:], targets["box"], atol=1e-6)

    # With identical rng streams the occupancy terms are identical, so the loss
    # difference between box_coeff=1 (the default) and box_coeff=0 is exactly the
    # mean squared bounding box error.
    unboxed = CODVAEReconstructionLossFn(
        vae, dataset=dataset, box_coeff=0.0, **LOSS_KWARGS
    )
    offset = np.array([0.1, -0.2, 0.3, 0.15], dtype=np.float32)
    off = good_prediction.copy()
    off[:, -4:] += offset
    with_box = loss_fn.numpy(off, targets, (NUM_OBJECTS,), rng=np.random.default_rng(12))
    without_box = unboxed.numpy(
        off, targets, (NUM_OBJECTS,), rng=np.random.default_rng(12)
    )
    expected = np.mean((off[:, -4:] - targets["box"]) ** 2, axis=-1)
    np.testing.assert_allclose(with_box - without_box, expected, atol=1e-5)
    assert unboxed.blind_guessing_expected_value < loss_fn.blind_guessing_expected_value


def test_occupancy_only(loss_fn, good_prediction, targets):
    off = good_prediction.copy()
    off[:, -4:] += np.array([0.1, -0.2, 0.3, 0.15], dtype=np.float32)
    # With identical rng streams the occupancy terms are identical, so dropping the
    # box MSE term via occupancy_only removes exactly the mean squared box error.
    full = loss_fn.numpy(off, targets, (NUM_OBJECTS,), rng=np.random.default_rng(13))
    occupancy = loss_fn.numpy(
        off, targets, (NUM_OBJECTS,), rng=np.random.default_rng(13), occupancy_only=True
    )
    expected = np.mean((off[:, -4:] - targets["box"]) ** 2, axis=-1)
    np.testing.assert_allclose(full - occupancy, expected, atol=1e-5)


def test_empirical_blind_guessing_stats(vae, dataset):
    loss_fn = CODVAEReconstructionLossFn(vae, dataset=dataset, **LOSS_KWARGS)
    heuristic = loss_fn.blind_guessing_expected_value
    box_target_std = np.array([0.5, 0.5, 0.25, 0.25])
    loss_fn.set_blind_guessing_stats(0.42, box_target_std)
    # box_coeff is 1.0 by default.
    expected = 0.42 + np.mean(box_target_std**2)
    assert np.isclose(loss_fn.blind_guessing_expected_value, expected)
    # The normalized loss maps the empirical blind guessing expected value to 1.
    assert np.isclose(loss_fn.normalized.blind_guessing_expected_value, 1.0)
    loss_fn.set_blind_guessing_stats(None, None)
    assert np.isclose(loss_fn.blind_guessing_expected_value, heuristic)


def test_box_gradient_is_exactly_the_mse_gradient(vae, loss_fn, good_prediction, targets):
    """
    The bounding box entries are supervised exclusively by the MSE term: the decoder
    is evaluated with COD-VAE's stop_transform_gradient, so even for a far-off tiny
    predicted box (where the decoder's own box gradient would vanish entirely), the
    box gradient equals the analytic MSE gradient and pulls the box toward the
    ground truth.
    """
    far = good_prediction.copy()
    far[:, -4:-1] = 1.5
    far[:, -1] = 0.05
    if vae.backend == "jax":
        import jax

        grad = np.asarray(
            jax.grad(
                lambda p: loss_fn.jax(
                    p, targets, (NUM_OBJECTS,), rng=jax.random.PRNGKey(2)
                ).sum()
            )(far)
        )
    else:
        import torch

        prediction = torch.from_numpy(far).requires_grad_(True)
        loss_fn.torch(
            prediction,
            {key: torch.from_numpy(value) for key, value in targets.items()},
            (NUM_OBJECTS,),
            rng=torch.Generator().manual_seed(2),
        ).sum().backward()
        grad = prediction.grad.numpy()
    box_grad = grad[:, -4:]
    assert np.all(np.isfinite(box_grad))
    # box_coeff * d/d(box) mean((box - gt)^2) = 2 * (box - gt) / 4 for box_coeff 1.
    box_error = far[:, -4:] - targets["box"]
    np.testing.assert_allclose(box_grad, box_error / 2, atol=1e-6)


@pytest.fixture(scope="module")
def loss_fn_fallback(vae, dataset):
    """Loss with device residency disabled, forcing the fallback query assembly."""
    return CODVAEReconstructionLossFn(
        vae, dataset=dataset, max_pool_vram_fraction=0.0, **LOSS_KWARGS
    )


def test_precision_inherited_from_vae(vae, dataset, loss_fn, good_prediction, targets):
    """
    The loss takes its compute dtype from the VAE rather than from an own flag, so a
    float16 model yields a float16 decode whose loss still closely matches float32.
    """
    half_vae = CODVAE.from_pretrained(MODEL, backend=vae.backend, dtype="float16")
    half = CODVAEReconstructionLossFn(half_vae, dataset=dataset, **LOSS_KWARGS)
    half_loss = half.numpy(
        good_prediction, targets, (NUM_OBJECTS,), rng=np.random.default_rng(11)
    )
    full_loss = loss_fn.numpy(
        good_prediction, targets, (NUM_OBJECTS,), rng=np.random.default_rng(11)
    )
    assert np.all(np.isfinite(half_loss))
    # The loss itself is always float32, whatever the model's dtype.
    assert half_loss.dtype == np.float32
    np.testing.assert_allclose(half_loss, full_loss, atol=0.05)


def test_pre_rotation_composed_into_target(vae, loss_fn, dataset, poses, targets):
    """
    Environment-side mesh pre-processing rotations (e.g. smallest_dimension_up) are
    composed into the target quaternion by the target provider (see
    TactileShapeReconstructionVectorEnv._get_prediction_targets); the raw-mesh pools
    must score the correspondingly rotated geometry.
    """
    from scipy.spatial.transform import Rotation

    from tactile_mnist.tactile_perception_vector_env import TactilePerceptionVectorEnv

    meshes = []
    composed_quaternions = []
    for i, pose in enumerate(poses):
        pre_rotation = TactilePerceptionVectorEnv._smallest_dimension_up_rotation(
            dataset[i].mesh
        )
        mesh = dataset[i].mesh.copy()
        mesh.vertices = pre_rotation.apply(mesh.vertices)
        mesh.apply_transform(pose.matrix)
        meshes.append(mesh)
        composed_quaternions.append(
            (Rotation.from_quat(pose.quaternion) * pre_rotation).as_quat()
        )
    composed_targets = {
        **targets,
        "quaternion": np.stack(composed_quaternions).astype(np.float32),
        "box": np.stack([_normalized_box(m) for m in meshes]).astype(np.float32),
    }
    good = vae.encode_mesh_full(
        meshes, object_scale=OBJECT_SCALE, seed=0, frame_half_size=FRAME_HALF_SIZE
    )
    loss_good = loss_fn.numpy(
        good, composed_targets, (NUM_OBJECTS,), rng=np.random.default_rng(9)
    )
    assert np.all(loss_good < 0.6 * loss_fn.blind_guessing_expected_value)

    # The cylinder's re-orientation is non-trivial (its axis is tipped over), so
    # forgetting the composition must score clearly worse.
    loss_uncomposed = loss_fn.numpy(
        good, targets, (NUM_OBJECTS,), rng=np.random.default_rng(9)
    )
    assert loss_uncomposed[1] > 2 * loss_good[1]

    # The device-resident variants must handle the composed targets identically.
    if vae.backend == "jax":
        import jax

        loss_backend = np.asarray(
            jax.jit(lambda p, t, key: loss_fn.jax(p, t, (NUM_OBJECTS,), rng=key))(
                good, composed_targets, jax.random.PRNGKey(3)
            )
        )
    else:
        import torch

        loss_backend = (
            loss_fn.torch(
                torch.from_numpy(good),
                {
                    key: torch.from_numpy(value)
                    for key, value in composed_targets.items()
                },
                (NUM_OBJECTS,),
                rng=torch.Generator().manual_seed(3),
            )
            .detach()
            .cpu()
            .numpy()
        )
    assert np.all(loss_backend < 0.6 * loss_fn.blind_guessing_expected_value)


def test_near_query_subsampling(vae, dataset, loss_fn, good_prediction, targets):
    with pytest.raises(ValueError, match="num_near_queries"):
        CODVAEReconstructionLossFn(
            vae, dataset=dataset, num_near_queries=513, **LOSS_KWARGS
        )
    subsampled = CODVAEReconstructionLossFn(
        vae, dataset=dataset, num_near_queries=128, **LOSS_KWARGS
    )
    loss = subsampled.numpy(
        good_prediction, targets, (NUM_OBJECTS,), rng=np.random.default_rng(10)
    )
    assert loss.shape == (NUM_OBJECTS,)
    assert np.all(np.isfinite(loss))
    again = subsampled.numpy(
        good_prediction, targets, (NUM_OBJECTS,), rng=np.random.default_rng(10)
    )
    np.testing.assert_array_equal(loss, again)
    # The subsampled loss is an unbiased estimate of the full-pool loss, so their
    # means only need to agree in magnitude.
    reference = loss_fn.numpy(
        good_prediction, targets, (NUM_OBJECTS,), rng=np.random.default_rng(10)
    )
    np.testing.assert_allclose(loss.mean(), reference.mean(), atol=0.25)


def test_backend_variant_fallback(
    vae, loss_fn, loss_fn_fallback, good_prediction, targets
):
    numpy_reference = loss_fn.numpy(
        good_prediction, targets, (NUM_OBJECTS,), rng=np.random.default_rng(8)
    )
    if vae.backend == "jax":
        import jax

        jitted = jax.jit(
            lambda p, t, key: loss_fn_fallback.jax(p, t, (NUM_OBJECTS,), rng=key)
        )
        loss = np.asarray(jitted(good_prediction, targets, jax.random.PRNGKey(1)))
    else:
        import torch

        generator = torch.Generator().manual_seed(1)
        prediction = torch.from_numpy(good_prediction).requires_grad_(True)
        loss_tensor = loss_fn_fallback.torch(
            prediction,
            {key: torch.from_numpy(value) for key, value in targets.items()},
            (NUM_OBJECTS,),
            rng=generator,
        )
        loss_tensor.sum().backward()
        grad = prediction.grad.numpy()
        assert np.all(np.isfinite(grad))
        np.testing.assert_array_equal(grad[:, -4:], 0.0)
        loss = loss_tensor.detach().cpu().numpy()
    assert loss.shape == (NUM_OBJECTS,)
    assert np.all(np.isfinite(loss))
    np.testing.assert_allclose(loss.mean(), numpy_reference.mean(), atol=0.25)
