from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .depth_estimator import DEPTH_OFFSET_M, DEPTH_SCALE_M, DepthEstimator
from ..tactile_renderer import Device
from ..tactile_renderer.cycle_gan_jax import (
    apply_generator_jax,
    load_generator_params_jax,
)

CYCLE_GAN_INVERSE_CHECKPOINT = Path(
    files("tactile_mnist.resources").joinpath("cycle_gan_inverse_tactile_mnist_v0.npz")
)


class CycleGANDepthEstimatorJAX(DepthEstimator):
    """Estimates depth maps with the inverse of the generator CycleGANRendererJAX uses.

    The two generators are the two directions of the same CycleGAN, so chaining this estimator and
    CycleGANRendererJAX maps a real tactile image into the domain of the simulated environments.
    """

    def __init__(self, device: Device | None = None):
        if device is None:
            self.__jax_device = jax.devices()[0]
        else:
            self.__jax_device = jax.devices(device.platform)[device.device_index]
        super().__init__(
            device=Device(self.__jax_device.platform, self.__jax_device.id),
            backend_name="jax",
        )
        self.__params = jax.device_put(
            load_generator_params_jax(CYCLE_GAN_INVERSE_CHECKPOINT), self.__jax_device
        )
        self.__estimate = jax.jit(self.__estimate_impl, static_argnames=("output_size",))

    def __estimate_impl(
        self, images: jax.Array, output_size: tuple[int, int]
    ) -> jax.Array:
        img = images.astype(jnp.float32)
        if images.dtype == jnp.uint8:
            img = img / 255
        input_width, input_height = self.input_size
        if img.shape[-3:-1] != (input_height, input_width):
            img = jax.image.resize(
                img,
                img.shape[:-3] + (input_height, input_width, img.shape[-1]),
                method="bicubic",
                antialias=True,
            ).clip(0, 1)
        # The generator was trained on images normalized to [-1, 1] and predicts the image coordinates in its first
        # two channels and the depth in its third one
        nocs = apply_generator_jax(self.__params, jnp.moveaxis((img - 0.5) / 0.5, -1, -3))
        depth_scaled = (nocs[..., 2, :, :] + 1) / 2
        if depth_scaled.shape[-2:] != (output_size[1], output_size[0]):
            depth_scaled = jax.image.resize(
                depth_scaled,
                depth_scaled.shape[:-2] + (output_size[1], output_size[0]),
                method="bicubic",
                antialias=True,
            )
        return depth_scaled.clip(0, 1) * DEPTH_SCALE_M + DEPTH_OFFSET_M

    def estimate(self, images: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
        depth = self.__estimate(
            jax.device_put(images, self.__jax_device), tuple(output_size)
        )
        return np.asarray(depth, dtype=np.float32)

    @property
    def jax_device(self) -> jax.Device:
        return self.__jax_device
