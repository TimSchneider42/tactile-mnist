from __future__ import annotations

import io
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import cv2
import numpy as np
import PIL.Image
import scipy.ndimage
from datasets import Image, List, load_dataset

from .constants import (
    CACHE_BASE_DIR,
    GELSIGHT_MINI_GEL_THICKNESS_MM,
    GELSIGHT_MINI_SENSOR_SURFACE_SIZE,
)
from .depth_estimator import mk_depth_estimator
from .tactile_renderer import RenderDirectOutput, TactileRenderer

# Touches whose recorded gel z stops within this distance of the round's minimum pressed down to the platform and
# thus did not touch the object. The deepest touch of a round is always a platform press (see
# TactileRealSnapConfig.recalibrate_sensor_z), and the per-touch spread of the platform presses (90th percentile
# 1mm) stays below the height of the objects.
EMPTY_TOUCH_Z_TOLERANCE_M = 0.001
# Minimum indentation (in m) of a simulated depth map for the touch to count as an object contact when scaling the
# base depth overlay (see SensorNoiseConfig.base_depth_contact_scale_range)
BASE_DEPTH_CONTACT_MIN_INDENTATION_M = 0.0001
# Number of recorded touches the depth estimator processes at once while building the base depth cache
BASE_DEPTH_CACHE_BATCH_SIZE = 32
# The contact variation is computed at no less than this resolution (pixels along the shorter side), so that
# boundary shifts of a fraction of a depth-map pixel (e.g. the 64x64 maps of the depth sensor) are still resolved
CONTACT_VARIATION_MIN_RESOLUTION = 256

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SensorNoiseConfig:
    """
    Model of the differences between simulated depth maps and the depth maps the re-rendering real-snap environments
    estimate from real tactile images (see TactileRealSnapConfig.sensor_type).

    The re-rendering real-snap environments obtain their depth maps from the CycleGAN depth estimator, so a policy
    trained in simulation sees real touches through that estimator. The model emulates its two main effects on the
    simulated depth maps before they are rendered:

    1. Contact geometry variation: the estimator under-recovers real contacts (thinner strokes, smoother profiles), so
       every touch shrinks or grows its contact region by a random boundary shift and optionally blurs it.
    2. Base depth overlay: the estimator writes artifacts into every estimate (most prominently a phantom
       indentation in the bottom-right corner), which are overlaid from estimates of real touches without contact.

    The contact variation is applied first, so the emulated artifacts are not affected by it. All terms default to
    off; the -DR variants of the environments with real recordings enable them (see registration.py).
    """

    # Per-touch variation of the contact geometry. Estimated real strokes are ~34% thinner than simulated ones and
    # the estimator recovers only ~65-72% of the contact area (domain-gap analysis on the real recordings). Every
    # touch draws a boundary shift (in m, positive shrinks the contact region by a grey-scale erosion of the
    # indentation, negative grows it by a dilation) from a normal distribution with contact_shift_mean and
    # contact_shift_std. With probability contact_blur_prob, the indentation is additionally blurred with a Gaussian
    # whose standard deviation (in m) is drawn uniformly from [0, contact_blur_sigma_max], as the estimated real depth
    # profiles are smoother than the sharp-edged simulated ones.
    contact_shift_mean: float = 0.0
    contact_shift_std: float = 0.0
    contact_blur_prob: float = 0.0
    contact_blur_sigma_max: float = 0.0
    # The CycleGAN depth estimator does not produce a clean no-contact estimate on real images: it writes a phantom
    # indentation into the bottom-right corner of every estimate (~8% of the frame, stable across recording sessions)
    # and a spatially structured noise floor elsewhere. If real_base_depth_dataset names a touch dataset (e.g.
    # "TimSchneider42/tactile-mnist-touch-real-single-t256-320x240"), the depth maps of all its recorded touches
    # without object contact (train split) are estimated once and cached on disk, and every simulated touch overlays
    # a randomly drawn one onto its depth map via an element-wise minimum, so simulated contact deeper than the
    # artifacts still shows through, exactly like real contact does in the real-snap environments. Touches without
    # object contact are identified by their recorded gel z stopping within EMPTY_TOUCH_Z_TOLERANCE_M of the round's
    # minimum, which means that they pressed down to the platform.
    real_base_depth_dataset: str | None = None
    # Range of the per-touch factor by which the base depth overlay is scaled for touches with object contact (drawn
    # uniformly). On the real recordings, the estimator artifacts are as deep in every touch without contact, but
    # 6-25% shallower in touches with object contact. The overlay pool consists of touches without contact, so contact
    # touches scale it down. (1, 1) disables the scaling.
    base_depth_contact_scale_range: tuple[float, float] = (1.0, 1.0)

    @property
    def varies_contact(self) -> bool:
        return (
            self.contact_shift_mean != 0.0
            or self.contact_shift_std > 0.0
            or (self.contact_blur_prob > 0.0 and self.contact_blur_sigma_max > 0.0)
        )


class SensorNoiseModel:
    """
    Applies the depth map variations described by [SensorNoiseConfig] to batches of simulated depth maps.

    All random draws use the generator passed to the apply methods. If `real_base_depth_dataset` is set, the
    environment has to call `init` before the first touch and `destroy` when it is closed.
    """

    def __init__(self, config: SensorNoiseConfig, num_envs: int):
        self.__config = config
        self.__num_envs = num_envs
        if config.real_base_depth_dataset is not None:
            dataset = load_dataset(config.real_base_depth_dataset, split="train")
            empty_touches = []
            for round_idx, positions in enumerate(
                dataset["gel_pose_cell_frame.position"]
            ):
                z = np.asarray(positions)[:, 2]
                for touch_idx in np.flatnonzero(
                    z <= z.min() + EMPTY_TOUCH_Z_TOLERANCE_M
                ):
                    empty_touches.append((round_idx, touch_idx))
            self.__base_depth_empty_touches = np.array(empty_touches)
            self.__base_depth_dataset_fingerprint = dataset._fingerprint
            # Materializing a row of the dataset decodes all images of that round (256 of them), of which only the
            # ones without object contact are needed. Hence, keep the images undecoded and decode the needed ones
            # manually in __build_base_depth_cache.
            self.__base_depth_dataset = dataset.select_columns(
                ["sensor_image"]
            ).cast_column("sensor_image", List(Image(decode=False)))
        self.__base_depth_maps: np.ndarray | None = None
        self.__contact_pool: ThreadPoolExecutor | None = None

    def init(
        self,
        num_envs: int,
        depth_map_size: tuple[int, int],
        estimator_backend: str = "auto",
        estimator_device: str | None = None,
        estimator_device_index: int = 0,
    ) -> None:
        """
        Load the depth estimates of all recorded touches without object contact for the base depth overlay.

        The estimates are cached on disk, keyed by the dataset and the depth map size. If the cache does not exist
        yet, the depth estimator is loaded once to build it. The environment has to call this before the first touch
        if `real_base_depth_dataset` is set (a no-op otherwise) and `destroy` when it is closed.

        :param num_envs:                Number of depth maps per batch, i.e. the batch size of the environment.
        :param depth_map_size:          Size (width, height) of the depth maps the environment renders.
        :param estimator_backend:       Backend of the depth estimator. Environments should pass the backend of
                                        their tactile renderer here, so that both share a single framework on the
                                        GPU.
        :param estimator_device:        Device to run the depth estimator on (None selects automatically).
        :param estimator_device_index:  Index of the device to run the depth estimator on.
        """
        if self.__config.real_base_depth_dataset is None:
            return
        if self.__base_depth_maps is not None:
            raise RuntimeError("init() was called twice without destroy() in between.")
        width, height = depth_map_size
        cache_file = (
            CACHE_BASE_DIR
            / "base-depth"
            / f"{self.__base_depth_dataset_fingerprint}_{EMPTY_TOUCH_Z_TOLERANCE_M}_{width}x{height}.npy"
        )
        expected_shape = (len(self.__base_depth_empty_touches), height, width)
        base_depth_maps = None
        if cache_file.exists():
            try:
                base_depth_maps = np.load(cache_file)
            except (OSError, ValueError) as e:
                logger.warning(
                    f"Ignoring unreadable base depth cache {cache_file}: {e}"
                )
            if base_depth_maps is not None and base_depth_maps.shape != expected_shape:
                logger.warning(
                    f"Ignoring base depth cache {cache_file} of shape {base_depth_maps.shape} (expected "
                    f"{expected_shape})."
                )
                base_depth_maps = None
        if base_depth_maps is None:
            base_depth_maps = self.__build_base_depth_cache(
                cache_file,
                tuple(depth_map_size),
                mk_depth_estimator(
                    backend=estimator_backend,
                    device=estimator_device,
                    device_index=estimator_device_index,
                ),
            )
        self.__base_depth_maps = base_depth_maps

    def destroy(self) -> None:
        """Release the base depth maps and the worker threads of the contact variation. The counterpart of `init`."""
        self.__base_depth_maps = None
        if self.__contact_pool is not None:
            self.__contact_pool.shutdown()
            self.__contact_pool = None

    def __build_base_depth_cache(
        self, cache_file: Path, depth_map_size: tuple[int, int], estimator: Any
    ) -> np.ndarray:
        """Estimate the depth maps of all recorded touches without object contact and store them in cache_file."""
        width, height = depth_map_size
        empty_touches = self.__base_depth_empty_touches
        logger.info(
            f"Building the base depth cache {cache_file} from {len(empty_touches)} recorded touches."
        )
        base_depth_maps = np.empty((len(empty_touches), height, width), np.float32)
        frames = []

        def flush(end: int) -> None:
            base_depth_maps[end - len(frames) : end] = estimator.estimate(
                np.stack(frames), depth_map_size
            )
            frames.clear()

        # The empty touches are sorted by round, so every row of the dataset is read only once
        current_round = None
        images = None
        for i, (round_idx, touch_idx) in enumerate(empty_touches):
            if round_idx != current_round:
                current_round = round_idx
                images = self.__base_depth_dataset[int(round_idx)]["sensor_image"]
            entry = images[touch_idx]
            if entry["bytes"] is None:
                image = PIL.Image.open(entry["path"])
            else:
                image = PIL.Image.open(io.BytesIO(entry["bytes"]))
            frames.append(np.asarray(image.convert("RGB")))
            if len(frames) == BASE_DEPTH_CACHE_BATCH_SIZE:
                flush(i + 1)
        if len(frames) > 0:
            flush(len(empty_touches))
        # Several processes may build the same cache at once, so each writes its own file and moves it into place
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = cache_file.with_name(f"{cache_file.stem}.{uuid.uuid4().hex}.tmp.npy")
        np.save(tmp_file, base_depth_maps)
        os.replace(tmp_file, cache_file)
        return base_depth_maps

    def apply_contact_variation(
        self, depth: np.ndarray, np_random: np.random.Generator
    ) -> np.ndarray:
        """
        Shrink, grow, and blur the contact region of every simulated depth map by a freshly drawn amount (see the
        contact_* fields of SensorNoiseConfig).

        :param depth:       (num_envs, height, width) batch of depth maps in meters (gel thickness = no contact),
                            covering GELSIGHT_MINI_SENSOR_SURFACE_SIZE.
        :param np_random:   Random number generator used to draw the per-touch variation.
        :return:            Batch of depth maps of the same shape, or `depth` unchanged if the variation is disabled.
        """
        config = self.__config
        if not config.varies_contact:
            return depth
        depth = np.asarray(depth)
        count = depth.shape[0]
        shifts = np_random.normal(
            config.contact_shift_mean, config.contact_shift_std, size=count
        )
        blur_sigmas = np.where(
            np_random.uniform(size=count) < config.contact_blur_prob,
            np_random.uniform(0.0, config.contact_blur_sigma_max, size=count),
            0.0,
        )
        height, width = depth.shape[-2:]
        zoom = max(
            1, int(np.ceil(CONTACT_VARIATION_MIN_RESOLUTION / min(height, width)))
        )
        # Size of a pixel (x, y) in m at the working resolution; the sensor surface is not square, so neither are
        # the pixels
        pixel_size = GELSIGHT_MINI_SENSOR_SURFACE_SIZE / (
            np.array([width, height]) * zoom
        )
        no_contact = GELSIGHT_MINI_GEL_THICKNESS_MM / 1000

        def vary(i: int) -> np.ndarray:
            indentation = np.maximum(no_contact - depth[i].astype(np.float64), 0.0)
            if not np.any(indentation > 0):
                return depth[i]
            if indentation.min() == indentation.max():
                # A constant indentation (the gel pressed flat onto the platform) is left unchanged by the filters
                return (no_contact - indentation).astype(depth.dtype)
            if zoom > 1:
                indentation = scipy.ndimage.zoom(indentation, zoom, order=1)
            radius_x, radius_y = np.abs(shifts[i]) / pixel_size
            morph_radius = np.zeros(2, dtype=int)
            footprint = None
            if radius_x >= 0.5 or radius_y >= 0.5:
                rx, ry = int(np.ceil(radius_x)), int(np.ceil(radius_y))
                y, x = np.mgrid[-ry : ry + 1, -rx : rx + 1]
                footprint = (x / max(radius_x, 1e-6)) ** 2 + (
                    y / max(radius_y, 1e-6)
                ) ** 2 <= 1.0
                morph_radius = np.array([ry, rx])
            blur_sigma = None
            blur_radius = np.zeros(2, dtype=int)
            if blur_sigmas[i] > 0:
                blur_sigma = (
                    blur_sigmas[i] / pixel_size[1],
                    blur_sigmas[i] / pixel_size[0],
                )
                # The support of scipy's Gaussian kernel (truncate=4)
                blur_radius = np.array([int(4.0 * s + 0.5) for s in blur_sigma])
            # Both filters only reach a bounded distance, and the indentation is zero outside its contact region, so
            # it suffices to filter a crop around the contact region: the output can only be non-zero within
            # out_margin of it, and computing the output there only reads the input within in_margin of it. Where the
            # crop is clipped by the image border, the "nearest" mode extends the image exactly as it does uncropped.
            out_margin = (morph_radius if shifts[i] <= 0 else 0) + blur_radius
            in_margin = out_margin + morph_radius + blur_radius
            rows, cols = np.nonzero(indentation)
            lo = np.array([rows.min(), cols.min()])
            hi = np.array([rows.max(), cols.max()]) + 1
            shape = np.array(indentation.shape)
            in_lo, in_hi = np.maximum(lo - in_margin, 0), np.minimum(
                hi + in_margin, shape
            )
            out_lo, out_hi = np.maximum(lo - out_margin, 0), np.minimum(
                hi + out_margin, shape
            )
            crop = indentation[in_lo[0] : in_hi[0], in_lo[1] : in_hi[1]]
            if footprint is not None:
                # OpenCV computes the same flat-footprint grey erosion/dilation as scipy.ndimage (BORDER_REPLICATE
                # is its "nearest" mode), but several times faster
                morph = cv2.erode if shifts[i] > 0 else cv2.dilate
                crop = morph(
                    crop, footprint.astype(np.uint8), borderType=cv2.BORDER_REPLICATE
                )
            if blur_sigma is not None:
                crop = scipy.ndimage.gaussian_filter(
                    crop, sigma=blur_sigma, mode="nearest"
                )
            indentation = np.zeros_like(indentation)
            indentation[out_lo[0] : out_hi[0], out_lo[1] : out_hi[1]] = crop[
                out_lo[0] - in_lo[0] : out_hi[0] - in_lo[0],
                out_lo[1] - in_lo[1] : out_hi[1] - in_lo[1],
            ]
            if zoom > 1:
                indentation = indentation.reshape(height, zoom, width, zoom).mean(
                    axis=(1, 3)
                )
            # Regions without contact keep their depth (it may lie beyond the gel thickness), regions that lost their
            # contact through the erosion move back to the gel thickness
            return np.where(
                indentation > 0,
                no_contact - indentation,
                np.maximum(depth[i], no_contact),
            ).astype(depth.dtype)

        # The filters release the GIL, so the touches are processed in parallel
        if self.__contact_pool is None:
            self.__contact_pool = ThreadPoolExecutor(
                max_workers=min(self.__num_envs, os.cpu_count() or 1)
            )
        return np.stack(list(self.__contact_pool.map(vary, range(count))))

    def apply_base_depth(
        self, depth: np.ndarray, np_random: np.random.Generator
    ) -> np.ndarray:
        """
        Overlay depth estimates of real touches without object contact onto a batch of simulated depth maps.

        Every touch receives the cached estimate of a freshly drawn recorded touch without object contact. The overlay
        is an element-wise minimum, so any simulated contact deeper than the artifacts of the depth estimator still
        shows through, exactly like real contact does in the re-rendering real-snap environments. For touches with
        object contact, the indentation of the overlay is first scaled by a factor drawn from
        `base_depth_contact_scale_range`, as the real estimator artifacts are weaker on those.

        :param depth:       (num_envs, height, width) batch of depth maps in meters.
        :param np_random:   Random number generator used to draw the overlays and their scaling.
        :return:            Batch of depth maps of the same shape with the empty-frame estimates overlaid, or `depth`
                            unchanged if `real_base_depth_dataset` is not set.
        """
        if self.__config.real_base_depth_dataset is None:
            return depth
        if self.__base_depth_maps is None:
            raise RuntimeError("apply_base_depth() requires init() to be called first.")
        if depth.shape[1:] != self.__base_depth_maps.shape[1:]:
            raise ValueError(
                f"Expected depth maps of shape {self.__base_depth_maps.shape[1:]} as announced to init(), but got "
                f"{depth.shape[1:]}."
            )
        base = self.__base_depth_maps[
            np_random.integers(len(self.__base_depth_maps), size=len(depth))
        ]
        scale_range = tuple(self.__config.base_depth_contact_scale_range)
        if scale_range != (1.0, 1.0):
            no_contact = GELSIGHT_MINI_GEL_THICKNESS_MM / 1000
            # A simulated touch has object contact if it indents the gel anywhere (presses on the bare platform do
            # not, see TactilePerceptionConfig.max_platform_penetration)
            contact = np.any(
                depth < no_contact - BASE_DEPTH_CONTACT_MIN_INDENTATION_M, axis=(-1, -2)
            )
            scales = np.where(
                contact, np_random.uniform(*scale_range, size=depth.shape[0]), 1.0
            )
            base = no_contact - scales[:, None, None] * (no_contact - base)
        return np.minimum(depth, base)

    def apply(self, depth: np.ndarray, np_random: np.random.Generator) -> np.ndarray:
        """Apply the contact variation and then the base depth overlay to a batch of depth maps."""
        depth = self.apply_contact_variation(depth, np_random)
        return self.apply_base_depth(depth, np_random)

    @property
    def config(self) -> SensorNoiseConfig:
        return self.__config


OutputType = TypeVar("OutputType")


class SensorNoiseRenderer(TactileRenderer[OutputType], Generic[OutputType]):
    """
    TactileRenderer wrapper that applies the depth map variations described by [SensorNoiseConfig] before passing the
    depth maps on to another renderer. `close` has to be called when the renderer is no longer needed.
    """

    def __init__(
        self,
        inner: TactileRenderer[OutputType],
        config: SensorNoiseConfig,
        num_envs: int,
        output_size: tuple[int, int],
        estimator_backend: str | None = None,
        estimator_device: str | None = None,
        estimator_device_index: int = 0,
    ):
        """
        :param inner:                   Renderer the depth map variations are applied in front of.
        :param config:                  Configuration of the depth map variations.
        :param num_envs:                Number of environments, i.e. the batch size of the rendered images.
        :param output_size:             Size (width, height) of the rendered images.
        :param estimator_backend:       Backend of the depth estimator of the base depth overlay. Defaults to the
                                        backend of the wrapped renderer, so that both share a single framework on
                                        the GPU ("auto" if the wrapped renderer runs on numpy, as the estimator has
                                        no numpy implementation).
        :param estimator_device:        Device to run the depth estimator on (None selects automatically).
        :param estimator_device_index:  Index of the device to run the depth estimator on.
        """
        super().__init__(
            device=inner.device,
            backend_name=inner.backend_name,
            channels=inner.channels,
        )
        self.__inner = inner
        self.__model = SensorNoiseModel(config, num_envs)
        if estimator_backend is None:
            estimator_backend = (
                "auto" if inner.backend_name == "numpy" else inner.backend_name
            )
        self.__model.init(
            num_envs,
            inner.get_desired_depth_map_size(output_size),
            estimator_backend=estimator_backend,
            estimator_device=estimator_device,
            estimator_device_index=estimator_device_index,
        )

    @staticmethod
    def __require_rng(rng: np.random.Generator | None) -> np.random.Generator:
        if rng is None:
            raise ValueError(
                "SensorNoiseRenderer requires an RNG to be passed to render()/render_direct()."
            )
        return rng

    def get_desired_depth_map_size(
        self, output_size: tuple[int, int]
    ) -> tuple[int, int]:
        return self.__inner.get_desired_depth_map_size(output_size)

    def render(
        self,
        depth: np.ndarray,
        output_size: tuple[int, int],
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        depth = self.__model.apply(depth, self.__require_rng(rng))
        return self.__inner.render(depth, output_size)

    def render_direct(
        self,
        depth: np.ndarray,
        output_size: tuple[int, int],
        rng: np.random.Generator | None = None,
    ) -> RenderDirectOutput[OutputType]:
        depth = self.__model.apply(depth, self.__require_rng(rng))
        return self.__inner.render_direct(depth, output_size)

    def close(self) -> None:
        self.__model.destroy()
        self.__inner.close()

    @property
    def inner(self) -> TactileRenderer[OutputType]:
        return self.__inner

    @property
    def config(self) -> SensorNoiseConfig:
        return self.__model.config
