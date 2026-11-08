"""Tests for the depth map variations of the -DR variants (tactile_mnist.sensor_noise)."""

import numpy as np
import pytest

from tactile_mnist import SensorNoiseConfig, SensorNoiseModel
from tactile_mnist.constants import GELSIGHT_MINI_GEL_THICKNESS_MM

NUM_ENVS = 8
NO_CONTACT = GELSIGHT_MINI_GEL_THICKNESS_MM / 1000
CONTACT = dict(
    contact_shift_mean=0.0006,
    contact_shift_std=0.00035,
    contact_blur_prob=0.3,
    contact_blur_sigma_max=0.0003,
)


def _stroke(num_envs: int = NUM_ENVS, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Depth maps of a 1 mm deep, sloped vertical stroke in the middle of the sensor."""
    width, height = size
    x = np.linspace(-1, 1, width)
    indentation = np.clip(0.001 * (1 - np.abs(x) / 0.15), 0, None)
    depth = np.broadcast_to(NO_CONTACT - indentation, (height, width))
    return np.repeat(depth[None].astype(np.float32), num_envs, axis=0)


def test_contact_variation_is_disabled_by_default():
    model = SensorNoiseModel(SensorNoiseConfig(), NUM_ENVS)
    depth = _stroke()
    assert model.apply(depth, np.random.default_rng(0)) is depth


def test_contact_variation_shrinks_contacts_on_average():
    model = SensorNoiseModel(SensorNoiseConfig(**CONTACT), NUM_ENVS)
    depth = _stroke()
    varied = model.apply_contact_variation(depth, np.random.default_rng(0))
    model.destroy()
    assert varied.shape == depth.shape and varied.dtype == depth.dtype
    area = (depth < NO_CONTACT).sum(axis=(1, 2))
    varied_area = (varied < NO_CONTACT).sum(axis=(1, 2))
    # The mean boundary shift erodes the contact region, and every touch draws its own shift
    assert varied_area.mean() < area.mean()
    assert len(np.unique(varied_area)) > 1
    # The variation never indents the gel deeper than the original contact
    assert varied.min() >= depth.min() - 1e-9


def test_contact_variation_leaves_touches_without_contact_and_flat_presses_unchanged():
    model = SensorNoiseModel(SensorNoiseConfig(**CONTACT), 2)
    depth = np.stack(
        [
            np.full((32, 32), NO_CONTACT + 0.001, dtype=np.float32),
            np.full((32, 32), NO_CONTACT - 0.002, dtype=np.float32),
        ]
    )
    varied = model.apply_contact_variation(depth, np.random.default_rng(0))
    model.destroy()
    np.testing.assert_array_equal(varied, depth)


def test_contact_variation_is_reproducible_for_a_given_seed():
    model = SensorNoiseModel(SensorNoiseConfig(**CONTACT), NUM_ENVS)
    first = model.apply_contact_variation(_stroke(), np.random.default_rng(42))
    second = model.apply_contact_variation(_stroke(), np.random.default_rng(42))
    model.destroy()
    np.testing.assert_array_equal(first, second)


REAL_TOUCH_DATASET = "TimSchneider42/tactile-mnist-touch-real-single-t256-320x240"


@pytest.fixture(scope="module")
def base_depth_model() -> SensorNoiseModel:
    model = SensorNoiseModel(
        SensorNoiseConfig(real_base_depth_dataset=REAL_TOUCH_DATASET), NUM_ENVS
    )
    model.init(NUM_ENVS, (256, 256))
    yield model
    model.destroy()


def test_base_depth_overlay_is_an_elementwise_minimum(base_depth_model):
    rng = np.random.default_rng(0)
    # A no-contact depth map picks up the artifacts of the depth estimator, most prominently the phantom
    # indentation in the bottom-right corner
    no_contact = np.full((NUM_ENVS, 256, 256), NO_CONTACT, dtype=np.float32)
    composed = base_depth_model.apply_base_depth(no_contact, rng)
    assert composed.shape == no_contact.shape
    assert np.all(composed <= no_contact + 1e-9)
    assert composed[:, 192:, 192:].mean() < 0.0035
    assert composed[:, :64, :64].mean() > 0.0040
    # Simulated contact deeper than the artifacts shows through unchanged
    deep_contact = np.full((NUM_ENVS, 256, 256), 0.002125, dtype=np.float32)
    np.testing.assert_allclose(
        base_depth_model.apply_base_depth(deep_contact, rng), deep_contact
    )


def test_base_depth_varies_between_touches_and_environments(base_depth_model):
    rng = np.random.default_rng(0)
    no_contact = np.full((NUM_ENVS, 256, 256), NO_CONTACT, dtype=np.float32)
    first = base_depth_model.apply_base_depth(no_contact, rng)
    second = base_depth_model.apply_base_depth(no_contact, rng)
    # Every touch draws a fresh empty frame, like the real sensor sees a fresh frame in every step
    assert np.abs(first - second).max() > 1e-5
    # and the environments draw their frames independently, so they cannot all be identical
    assert any(
        np.abs(first[i] - first[j]).max() > 1e-5
        for i in range(NUM_ENVS)
        for j in range(i + 1, NUM_ENVS)
    )


def test_base_depth_is_scaled_down_on_touches_with_contact():
    model = SensorNoiseModel(
        SensorNoiseConfig(
            real_base_depth_dataset=REAL_TOUCH_DATASET,
            base_depth_contact_scale_range=(0.5, 0.5),
        ),
        NUM_ENVS,
    )
    model.init(NUM_ENVS, (256, 256))
    try:
        no_contact = np.full((NUM_ENVS, 256, 256), NO_CONTACT, dtype=np.float32)
        contact = no_contact.copy()
        # A small, deep contact far from the bottom-right corner
        contact[:, 10:20, 10:20] = NO_CONTACT - 0.002
        # The same seed draws the same overlays for both batches
        plain = model.apply_base_depth(no_contact, np.random.default_rng(3))
        scaled = model.apply_base_depth(contact, np.random.default_rng(3))
        np.testing.assert_allclose(
            NO_CONTACT - scaled[:, 128:, 128:],
            0.5 * (NO_CONTACT - plain[:, 128:, 128:]),
            atol=1e-7,
        )
    finally:
        model.destroy()


def test_base_depth_matches_the_depth_map_size():
    model = SensorNoiseModel(
        SensorNoiseConfig(real_base_depth_dataset=REAL_TOUCH_DATASET), NUM_ENVS
    )
    model.init(NUM_ENVS, (32, 24))
    try:
        no_contact = np.full((NUM_ENVS, 24, 32), NO_CONTACT, dtype=np.float32)
        composed = model.apply_base_depth(no_contact, np.random.default_rng(0))
        assert composed.shape == no_contact.shape
        assert composed[:, 18:, 24:].mean() < 0.0035
    finally:
        model.destroy()


def test_base_depth_requires_init():
    model = SensorNoiseModel(
        SensorNoiseConfig(real_base_depth_dataset=REAL_TOUCH_DATASET), NUM_ENVS
    )
    with pytest.raises(RuntimeError):
        model.apply_base_depth(
            np.full((NUM_ENVS, 256, 256), NO_CONTACT, dtype=np.float32),
            np.random.default_rng(0),
        )


def test_base_depth_is_disabled_by_default():
    model = SensorNoiseModel(SensorNoiseConfig(), NUM_ENVS)
    # init and destroy are no-ops without a dataset, and the depth maps pass through unchanged
    model.init(NUM_ENVS, (256, 256))
    no_contact = np.full((NUM_ENVS, 256, 256), NO_CONTACT, dtype=np.float32)
    np.testing.assert_array_equal(
        model.apply_base_depth(no_contact, np.random.default_rng(0)), no_contact
    )
    model.destroy()
