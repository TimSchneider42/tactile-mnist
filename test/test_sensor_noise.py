"""Tests for the sensor noise model (tactile_mnist.sensor_noise)."""

import dataclasses

import numpy as np
import pytest

from tactile_mnist import SensorNoiseConfig, SensorNoiseModel

IMAGE_SIZE = (64, 48)  # width, height
CHANNELS = 3
NUM_ENVS = 8
BACKGROUND = 0.344


def _mk_model(num_envs: int = NUM_ENVS, **config_overrides) -> SensorNoiseModel:
    config = dataclasses.replace(SensorNoiseConfig(), **config_overrides)
    return SensorNoiseModel(config, num_envs, IMAGE_SIZE, CHANNELS)


def _background(num_envs: int = NUM_ENVS) -> np.ndarray:
    return np.full(
        (num_envs, IMAGE_SIZE[1], IMAGE_SIZE[0], CHANNELS), BACKGROUND, dtype=np.float32
    )


def _lag_autocorrelation(images: np.ndarray, lag: int) -> float:
    gray = images.mean(-1).reshape(-1, IMAGE_SIZE[1], IMAGE_SIZE[0])
    gray = gray - gray.mean(axis=(1, 2), keepdims=True)
    return float(np.mean(gray[:, :, :-lag] * gray[:, :, lag:]) / np.mean(gray**2))


def test_frames_without_contact_are_not_identical():
    """The whole point of the noise model: no two images are ever exactly the same."""
    model = _mk_model()
    np_random = np.random.default_rng(0)
    model.reset(np.ones(NUM_ENVS, dtype=np.bool_), np_random)
    images = np.stack([model.apply(_background(), np_random) for _ in range(4)])
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            assert np.abs(images[i] - images[j]).mean() > 1e-4


def test_frame_noise_matches_configured_magnitude_and_correlation():
    model = _mk_model(
        episode_pattern_std=0.0, episode_gain_std=0.0, episode_offset_std=0.0
    )
    np_random = np.random.default_rng(0)
    model.reset(np.ones(NUM_ENVS, dtype=np.bool_), np_random)
    residual = (
        np.stack([model.apply(_background(), np_random) for _ in range(16)])
        - BACKGROUND
    )
    assert residual.std() == pytest.approx(
        SensorNoiseConfig().frame_noise_std, rel=0.05
    )
    # The noise is spatially correlated (calibrated on the real sensor) and correlated across the channels
    assert _lag_autocorrelation(residual, 1) == pytest.approx(0.47, abs=0.06)
    assert _lag_autocorrelation(residual, 2) == pytest.approx(0.25, abs=0.06)
    channels = residual.reshape(-1, CHANNELS)
    assert np.corrcoef(channels.T)[0, 1] == pytest.approx(
        SensorNoiseConfig().frame_noise_channel_correlation, abs=0.05
    )


def test_episode_parameters_are_constant_within_and_differ_between_episodes():
    model = _mk_model(frame_noise_std=0.0)
    np_random = np.random.default_rng(0)
    model.reset(np.ones(NUM_ENVS, dtype=np.bool_), np_random)
    first = model.apply(_background(), np_random)
    second = model.apply(_background(), np_random)
    # Without frame noise, the sensor is deterministic within an episode
    np.testing.assert_allclose(first, second)
    # but different environments see different gel and illumination states
    assert np.abs(first[0] - first[1]).mean() > 1e-4

    # Resetting only some environments leaves the others untouched
    mask = np.zeros(NUM_ENVS, dtype=np.bool_)
    mask[: NUM_ENVS // 2] = True
    model.reset(mask, np_random)
    third = model.apply(_background(), np_random)
    assert np.abs(third[0] - first[0]).mean() > 1e-4
    np.testing.assert_allclose(third[NUM_ENVS // 2 :], first[NUM_ENVS // 2 :])


def test_episode_statistics_match_configuration():
    num_envs = 256
    model = _mk_model(num_envs, frame_noise_std=0.0)
    np_random = np.random.default_rng(0)
    model.reset(np.ones(num_envs, dtype=np.bool_), np_random)
    images = model.apply(_background(num_envs), np_random)
    config = SensorNoiseConfig()
    assert images.mean(axis=(1, 2, 3)).std() == pytest.approx(
        config.episode_offset_std, rel=0.15
    )
    pattern = images - images.mean(axis=(1, 2, 3), keepdims=True)
    assert pattern.std() == pytest.approx(config.episode_pattern_std, rel=0.15)


def test_gain_scales_contrast_without_shifting_the_mean():
    model = _mk_model(
        num_envs=256,
        frame_noise_std=0.0,
        episode_pattern_std=0.0,
        episode_offset_std=0.0,
    )
    np_random = np.random.default_rng(0)
    model.reset(np.ones(256, dtype=np.bool_), np_random)
    # Values well inside [0, 1], so that the result is not affected by clipping
    images = (
        np_random.uniform(size=(256, IMAGE_SIZE[1], IMAGE_SIZE[0], CHANNELS)) * 0.4
        + 0.3
    )
    noisy = model.apply(images, np_random)
    np.testing.assert_allclose(
        noisy.mean(axis=(1, 2, 3)), images.mean(axis=(1, 2, 3)), atol=1e-6
    )
    contrast_ratio = noisy.std(axis=(1, 2, 3)) / images.std(axis=(1, 2, 3))
    assert contrast_ratio.std() == pytest.approx(
        SensorNoiseConfig().episode_gain_std, rel=0.15
    )


def test_output_is_clipped_to_the_valid_image_range():
    model = _mk_model()
    np_random = np.random.default_rng(0)
    model.reset(np.ones(NUM_ENVS, dtype=np.bool_), np_random)
    for value in (0.0, 1.0):
        images = np.full(
            (NUM_ENVS, IMAGE_SIZE[1], IMAGE_SIZE[0], CHANNELS), value, dtype=np.float32
        )
        noisy = model.apply(images, np_random)
        assert noisy.min() >= 0.0 and noisy.max() <= 1.0


def test_channels_first_images_are_supported():
    """The PyTorch renderers return channels-first images, the others channels-last."""
    model = _mk_model()
    np_random = np.random.default_rng(0)
    model.reset(np.ones(NUM_ENVS, dtype=np.bool_), np_random)
    channels_first = np.moveaxis(_background(), -1, -3)
    noisy = model.apply(channels_first, np_random)
    assert noisy.shape == channels_first.shape
    assert noisy.std() == pytest.approx(
        model.apply(_background(), np_random).std(), rel=0.3
    )

    with pytest.raises(ValueError):
        model.apply(np.zeros((NUM_ENVS, 8, 8, CHANNELS), dtype=np.float32), np_random)


def test_torch_images_are_supported():
    torch = pytest.importorskip("torch")
    model = _mk_model()
    np_random = np.random.default_rng(0)
    model.reset(np.ones(NUM_ENVS, dtype=np.bool_), np_random)
    images = torch.full((NUM_ENVS, CHANNELS, IMAGE_SIZE[1], IMAGE_SIZE[0]), BACKGROUND)
    noisy = model.apply(images, np_random)
    assert isinstance(noisy, torch.Tensor)
    assert noisy.shape == images.shape and noisy.dtype == images.dtype
    assert float((noisy - images).abs().mean()) > 1e-4


def test_penetration_depths_are_never_deeper_than_nominal():
    config = SensorNoiseConfig()
    model = _mk_model()
    np_random = np.random.default_rng(0)
    nominal = np.full(4096, 0.002125)
    depths = model.sample_penetration_depths(np_random, nominal, 0.00425)
    # The nominal depth is the deepest press, all other touches are less deep (i.e. further away from the surface)
    assert np.all(depths >= nominal)
    assert np.all(depths <= 0.00425)
    reduction = depths - nominal
    assert np.median(reduction) == pytest.approx(
        0.6745 * config.penetration_depth_reduction_std, rel=0.1
    )


def test_noise_is_reproducible_for_a_given_seed():
    def rollout() -> np.ndarray:
        model = _mk_model()
        np_random = np.random.default_rng(42)
        model.reset(np.ones(NUM_ENVS, dtype=np.bool_), np_random)
        return np.stack([model.apply(_background(), np_random) for _ in range(3)])

    np.testing.assert_array_equal(rollout(), rollout())
