from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class SensorNoiseConfig:
    """
    Model of the appearance variations that a real GelSight Mini sensor shows on top of the pure contact geometry.

    Simulated tactile images are deterministic functions of the contact geometry, so images without contact are all
    exactly identical. Real images never are: every frame carries sensor noise, and every episode (round) sees a
    slightly different gel and illumination state. A policy trained on noise-free images can therefore learn
    "image == background => no contact", a cue that does not exist on the real sensor.

    The model applied to each image is

        img_out = mean(img) + (img - mean(img)) * gain + offset + pattern + noise

    where `gain`, `offset`, and `pattern` are drawn once per episode and `noise` is drawn per frame.

    The default values are calibrated on the
    [tactile-mnist-touch-real-single-t256-64x64](https://huggingface.co/datasets/TimSchneider42/tactile-mnist-touch-real-single-t256-64x64)
    dataset: for each round, the background image (per-round pixel-wise median) was fitted against the global
    background image with a contrast gain and an offset, and the per-frame residuals of the frames without contact
    were used to characterize the noise. Both `pattern` and `noise` are sampled as Gaussian random fields with a
    `1 / f^spectrum_exponent` power spectrum (`f` in cycles per image), which reproduces the measured spatial
    autocorrelation of the real residuals reasonably well.
    """

    # Per-frame sensor noise (measured: std 0.0042 per channel, inter-channel correlation 0.24-0.39, spatial
    # autocorrelation 0.47/0.25/0.20 at a lag of 1/2/3 pixels, which an exponent of 1.4 reproduces)
    frame_noise_std: float = 0.0042
    frame_noise_spectrum_exponent: float = 1.4
    frame_noise_channel_correlation: float = 0.3
    # Per-episode static background pattern (measured: rms 0.0093, spatial autocorrelation 0.98 at a lag of 1 pixel
    # and 0.93 at a lag of 8 pixels, i.e. a very smooth field, which requires an exponent of at least 4)
    episode_pattern_std: float = 0.0093
    episode_pattern_spectrum_exponent: float = 4.0
    episode_pattern_channel_correlation: float = 0.3
    # Per-episode contrast gain and brightness offset (measured: gain std 0.047, offset std 0.0094)
    episode_gain_std: float = 0.047
    episode_offset_std: float = 0.0094
    # Per-touch variation of the depth the sensor is pressed into the surface. The simulation always presses the gel
    # in by exactly GEL_PENETRATION_DEPTH_MM, so all touches that miss the object end up at exactly the same height,
    # whereas the real robot presses down until a force threshold is reached and thus penetrates by an amount that
    # depends on the contact area and the state of the gel. GEL_PENETRATION_DEPTH_MM is treated as the deepest press
    # (that is what the recalibration of the real environment aligns it with, see
    # TactileRealSnapConfig.recalibrate_sensor_z), and the penetration of each touch is reduced by the absolute value
    # of a normally distributed variable with this scale. The default is calibrated such that the touches reaching
    # the platform spread like they do in the real dataset (median 0.46mm and 90th percentile 1.0mm less deep than
    # the deepest touch of the round).
    penetration_depth_reduction_std: float = 0.00065


def _as_type_of(array: np.ndarray, reference: Any) -> Any:
    """Convert a numpy array to the array type (and device) of `reference`."""
    if isinstance(reference, np.ndarray):
        return array.astype(reference.dtype)
    module_name = type(reference).__module__.split(".")[0]
    if module_name == "torch":
        import torch

        return torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
    import jax.numpy as jnp

    return jnp.asarray(array, dtype=reference.dtype)


def _mean_per_image(images: Any) -> Any:
    axes = tuple(range(1, len(images.shape)))
    try:
        return images.mean(axis=axes, keepdims=True)
    except TypeError:
        # torch.Tensor.mean does not accept the numpy argument names
        return images.mean(dim=axes, keepdim=True)


class SensorNoiseModel:
    """
    Samples and applies the per-episode and per-frame sensor variations described by [SensorNoiseConfig].

    The noise is always sampled in numpy (using the environment's random number generator) and converted to the array
    type of the images it is applied to, so it works with the numpy, PyTorch, and JAX sensor backends alike.
    """

    def __init__(
        self,
        config: SensorNoiseConfig,
        num_envs: int,
        image_size: tuple[int, int],
        channels: int,
    ):
        self.__config = config
        self.__num_envs = num_envs
        width, height = image_size
        self.__image_shape = (height, width, channels)
        self.__frame_noise_scale = self.__mk_spectrum_scale(
            height, width, config.frame_noise_spectrum_exponent
        )
        self.__pattern_scale = self.__mk_spectrum_scale(
            height, width, config.episode_pattern_spectrum_exponent
        )
        self.__gain = np.ones(num_envs)
        self.__offset = np.zeros(num_envs)
        self.__pattern = np.zeros((num_envs, *self.__image_shape))

    @staticmethod
    def __mk_spectrum_scale(
        height: int, width: int, spectrum_exponent: float
    ) -> np.ndarray:
        """
        Compute the per-frequency amplitude scaling of a Gaussian random field with a 1 / f^spectrum_exponent power
        spectrum, normalized such that filtering unit-variance white noise with it yields unit variance again.
        """
        freq_y = np.fft.fftfreq(height) * height
        freq_x = np.fft.fftfreq(width) * width
        freq = np.hypot(freq_y[:, None], freq_x[None, :])
        scale = np.zeros_like(freq)
        # The DC component is dropped, as the mean of the field is modeled by the offset term
        scale[freq > 0] = freq[freq > 0] ** (-spectrum_exponent / 2)
        return scale / np.sqrt(np.mean(scale**2))

    def __sample_field(
        self,
        np_random: np.random.Generator,
        count: int,
        spectrum_scale: np.ndarray,
        channel_correlation: float,
    ) -> np.ndarray:
        height, width, channels = self.__image_shape

        def filtered_white_noise(shape: tuple[int, ...]) -> np.ndarray:
            white = np_random.normal(size=shape)
            spectrum = np.fft.fft2(white, axes=(1, 2), norm="ortho")
            return np.real(
                np.fft.ifft2(
                    spectrum * spectrum_scale[None, :, :, None],
                    axes=(1, 2),
                    norm="ortho",
                )
            )

        # A component shared between the channels plus an independent one per channel, mixed such that the result has
        # unit variance and the requested inter-channel correlation
        shared = filtered_white_noise((count, height, width, 1))
        independent = filtered_white_noise((count, height, width, channels))
        return (
            np.sqrt(channel_correlation) * shared
            + np.sqrt(1 - channel_correlation) * independent
        )

    def reset(self, mask: Sequence[bool], np_random: np.random.Generator) -> None:
        """Resample the per-episode parameters of all environments selected by `mask`."""
        mask = np.asarray(mask, dtype=np.bool_)
        count = int(np.sum(mask))
        if count == 0:
            return
        self.__gain[mask] = 1 + np_random.normal(
            scale=self.__config.episode_gain_std, size=count
        )
        self.__offset[mask] = np_random.normal(
            scale=self.__config.episode_offset_std, size=count
        )
        self.__pattern[mask] = (
            self.__sample_field(
                np_random,
                count,
                self.__pattern_scale,
                self.__config.episode_pattern_channel_correlation,
            )
            * self.__config.episode_pattern_std
        )

    def sample_penetration_depths(
        self,
        np_random: np.random.Generator,
        nominal_depth: np.ndarray,
        max_depth: float,
    ) -> np.ndarray:
        """
        Sample the depth by which the sensor is pressed into the surface for a batch of touches.

        Note that `nominal_depth` is the distance between the sensor frame and the closest point of the touched
        surface, so it *decreases* as the sensor is pressed in further. Accordingly, the sampled depths are always
        larger than the nominal ones, that is, the nominal depth is the deepest press.

        :param np_random:       Random number generator used to sample the penetration depths.
        :param nominal_depth:   Penetration depths the simulation would use without noise.
        :param max_depth:       Maximum depth (the thickness of the gel, i.e. no contact at all).
        :return:                Sampled penetration depths of the same shape as `nominal_depth`.
        """
        return np.minimum(
            nominal_depth
            + np.abs(
                np_random.normal(
                    scale=self.__config.penetration_depth_reduction_std,
                    size=nominal_depth.shape,
                )
            ),
            max_depth,
        )

    def apply(self, images: Any, np_random: np.random.Generator) -> Any:
        """
        Apply the sampled per-episode parameters and fresh per-frame noise to a batch of tactile images.

        :param images:      Batch of tactile images, either channels-last (N, H, W, C) or channels-first (N, C, H, W),
                            with values in [0, 1].
        :param np_random:   Random number generator used to sample the per-frame noise.
        :return:            Batch of augmented tactile images of the same type and shape as `images`.
        """
        height, width, channels = self.__image_shape
        if tuple(images.shape) == (self.__num_envs, height, width, channels):
            channels_last = True
        elif tuple(images.shape) == (self.__num_envs, channels, height, width):
            channels_last = False
        else:
            raise ValueError(
                f"Expected images of shape {(self.__num_envs, height, width, channels)} or "
                f"{(self.__num_envs, channels, height, width)}, but got {tuple(images.shape)}."
            )
        noise = (
            self.__sample_field(
                np_random,
                self.__num_envs,
                self.__frame_noise_scale,
                self.__config.frame_noise_channel_correlation,
            )
            * self.__config.frame_noise_std
        )
        additive = self.__pattern + noise + self.__offset[:, None, None, None]
        gain = self.__gain[:, None, None, None]
        if not channels_last:
            additive = np.moveaxis(additive, -1, -3)
        image_mean = _mean_per_image(images)
        return (
            (images - image_mean) * _as_type_of(gain, images)
            + image_mean
            + _as_type_of(additive, images)
        ).clip(0, 1)

    @property
    def config(self) -> SensorNoiseConfig:
        return self.__config
