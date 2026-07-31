from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from datasets import load_dataset

from .depth_estimator import mk_depth_estimator

# Touches whose recorded gel z stops within this distance of the round's minimum pressed down to the platform and
# thus did not touch the object. The deepest touch of a round is always a platform press (see
# TactileRealSnapConfig.recalibrate_sensor_z), and the per-touch spread of the platform presses (90th percentile
# 1mm, see penetration_depth_reduction_std) stays below the height of the objects.
EMPTY_TOUCH_Z_TOLERANCE_M = 0.001
# Number of base depth map batches the feeder thread keeps ready for apply_base_depth
BASE_DEPTH_PREFETCH_BATCHES = 2


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
    dataset. The per-episode terms are obtained by fitting the background image of each round (its pixel-wise median)
    against the global background image with a contrast gain and an offset, and taking the gain, the offset, and the
    remaining difference as `gain`, `offset`, and `pattern`. The per-frame term is obtained by subtracting the
    background image of a round from its individual frames, which removes everything the per-episode terms already
    model. Only the frames that press down on the platform (the deepest touches of a round) enter both fits, as any
    frame with contact would contribute the imprint rather than the sensor variation.

    Both `pattern` and `noise` are sampled as Gaussian random fields with a `1 / f^spectrum_exponent` power spectrum
    (`f` in cycles per image).

    Besides these image-space terms, the model can also emulate the artifacts of the depth estimator that the
    re-rendering real-snap environments use (see `real_base_depth_dataset`), which operates on the depth maps before
    they are rendered rather than on the rendered images.
    """

    # Per-frame sensor noise. The amplitude is measured on the residuals of the frames that press down on the
    # platform (std 0.0038 per channel, inter-channel correlation 0.24-0.39), and the exponent is chosen such that
    # the Laplacian energy of the residuals matches the measured 0.0064 (it comes out at 0.0066). Note that the
    # exponent does not reproduce the measured short-lag autocorrelation of 0.34/0.10/0.06 at a lag of 1/2/3 pixels
    # at the same time (it yields 0.47/0.25/0.19): the real noise is not exactly a power-law field. Matching the
    # Laplacian energy is what keeps the simulated images from being separable from the real ones, so it wins.
    frame_noise_std: float = 0.0038
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
    # The re-rendering real-snap environments (see TactileRealSnapConfig.sensor_type) obtain their depth maps from
    # the CycleGAN depth estimator, which does not produce a clean no-contact estimate on real images: it writes a
    # phantom indentation into the bottom-right corner of every estimate (~8% of the frame, stable across recording
    # sessions) and a spatially structured noise floor elsewhere. Neither artifact varies enough with the actual
    # contact to be removed by the cycle-consistency training, so real-snap observations carry them in every frame
    # while purely simulated depth maps do not. If real_base_depth_dataset names a touch dataset (e.g.
    # "TimSchneider42/tactile-mnist-touch-real-single-t256-320x240"), every simulated touch draws one recorded touch
    # without object contact from its train split, estimates its depth map with the CycleGAN depth estimator on the
    # fly, and overlays it onto the simulated depth map via an element-wise minimum, so simulated contact deeper
    # than the artifacts still shows through, exactly like real contact does in the real-snap environments. Touches
    # without object contact are identified by their recorded gel z stopping within EMPTY_TOUCH_Z_TOLERANCE_M of the
    # round's minimum, which means that they pressed down to the platform.
    real_base_depth_dataset: str | None = None


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

    If `real_base_depth_dataset` is set, the environment has to call `init` before the first touch and `destroy`
    when it is closed; the base depth maps are prefetched by a feeder thread in between.
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
        if config.real_base_depth_dataset is not None:
            self.__base_depth_dataset = load_dataset(
                config.real_base_depth_dataset, split="train"
            )
            empty_touches = []
            for round_idx, positions in enumerate(
                self.__base_depth_dataset["gel_pose_cell_frame.position"]
            ):
                z = np.asarray(positions)[:, 2]
                for touch_idx in np.flatnonzero(
                    z <= z.min() + EMPTY_TOUCH_Z_TOLERANCE_M
                ):
                    empty_touches.append((round_idx, touch_idx))
            self.__base_depth_empty_touches = np.array(empty_touches)
        self.__base_depth_estimator: Any = None
        self.__base_depth_feeder: threading.Thread | None = None
        self.__base_depth_queue: queue.Queue[np.ndarray] | None = None
        self.__base_depth_stop: threading.Event | None = None

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
    def init(
        self,
        num_envs: int,
        depth_map_size: tuple[int, int],
        estimator_backend: str = "auto",
        estimator_device: str | None = None,
        estimator_device_index: int = 0,
    ) -> None:
        """
        Load the depth estimator and start the feeder thread that keeps base depth map batches ready.

        The environment has to call this before the first touch if `real_base_depth_dataset` is set (a no-op
        otherwise) and `destroy` when it is closed.

        :param num_envs:                Number of depth maps per prefetched batch, i.e. the batch size of the
                                        environment.
        :param depth_map_size:          Size (width, height) of the depth maps the environment renders.
        :param estimator_backend:       Backend of the depth estimator. Environments should pass the backend of
                                        their tactile renderer here, so that both share a single framework on the
                                        GPU.
        :param estimator_device:        Device to run the depth estimator on (None selects automatically).
        :param estimator_device_index:  Index of the device to run the depth estimator on.
        """
        if self.__config.real_base_depth_dataset is None:
            return
        if self.__base_depth_feeder is not None:
            raise RuntimeError("init() was called twice without destroy() in between.")
        self.__base_depth_estimator = mk_depth_estimator(
            backend=estimator_backend,
            device=estimator_device,
            device_index=estimator_device_index,
        )
        self.__base_depth_queue = queue.Queue(maxsize=BASE_DEPTH_PREFETCH_BATCHES)
        self.__base_depth_stop = threading.Event()
        self.__base_depth_feeder = threading.Thread(
            target=self.__feed_base_depth,
            args=(num_envs, tuple(depth_map_size)),
            daemon=True,
        )
        self.__base_depth_feeder.start()

    def destroy(self) -> None:
        """Stop the feeder thread and release the depth estimator. The counterpart of `init`."""
        if self.__base_depth_feeder is None:
            return
        self.__base_depth_stop.set()
        self.__base_depth_feeder.join()
        self.__base_depth_feeder = None
        self.__base_depth_queue = None
        self.__base_depth_stop = None
        self.__base_depth_estimator = None

    def __feed_base_depth(
        self, num_envs: int, depth_map_size: tuple[int, int]
    ) -> None:
        """Continuously estimate batches of base depth maps from random empty touches into the queue."""
        rng = np.random.default_rng()
        while not self.__base_depth_stop.is_set():
            picks = self.__base_depth_empty_touches[
                rng.integers(len(self.__base_depth_empty_touches), size=num_envs)
            ]
            frames = np.stack(
                [
                    np.asarray(
                        self.__base_depth_dataset[int(round_idx)]["sensor_image"][
                            int(touch_idx)
                        ].convert("RGB")
                    )
                    for round_idx, touch_idx in picks
                ]
            )
            batch = self.__base_depth_estimator.estimate(frames, depth_map_size)
            while not self.__base_depth_stop.is_set():
                try:
                    self.__base_depth_queue.put(batch, timeout=0.1)
                    break
                except queue.Full:
                    pass

    def apply_base_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Overlay depth estimates of real touches without object contact onto a batch of simulated depth maps.

        Every touch receives the estimate of a freshly drawn recorded touch without object contact, prefetched by
        the feeder thread `init` spawned. The depth estimator producing the estimates is the one whose artifacts the
        overlay is meant to emulate. The overlay is an element-wise minimum, so any simulated contact deeper than
        the artifacts of the depth estimator still shows through, exactly like real contact does in the
        re-rendering real-snap environments.

        :param depth:   (num_envs, height, width) batch of depth maps in meters.
        :return:        Batch of depth maps of the same shape with the empty-frame estimates overlaid, or `depth`
                        unchanged if `real_base_depth_dataset` is not set.
        """
        if self.__config.real_base_depth_dataset is None:
            return depth
        if self.__base_depth_feeder is None:
            raise RuntimeError(
                "apply_base_depth() requires init() to be called first."
            )
        while True:
            try:
                base = self.__base_depth_queue.get(timeout=1.0)
                break
            except queue.Empty:
                if not self.__base_depth_feeder.is_alive():
                    raise RuntimeError("The base depth feeder thread died unexpectedly.")
        if base.shape != depth.shape:
            raise ValueError(
                f"Expected depth maps of shape {base.shape} as announced to init(), but got {depth.shape}."
            )
        return np.minimum(depth, base)

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
