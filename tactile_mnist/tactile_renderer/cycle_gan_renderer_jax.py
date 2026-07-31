from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import jax
import jax.numpy as jnp

from .cycle_gan_jax import apply_generator_jax, load_generator_params_jax
from .depth_renderer_jax import DepthRendererJAX
from .tactile_renderer import Device
from .tactile_renderer_jax import TactileRendererJAX

CYCLE_GAN_CHECKPOINT = Path(
    files("tactile_mnist.resources").joinpath("cycle_gan_tactile_mnist_v0.npz")
)


def encode_depth_jax(depth_scaled: jax.Array) -> jax.Array:
    """Turn a depth map scaled to [0, 1] into the 3-channel input of the CycleGAN generator.

    The first two channels hold the normalized image coordinates and the third one the depth. Shapes are
    (..., H, W, 1) in and (..., 3, H, W) out, as the generator operates on channel-first data.
    """
    # If the depth map is completely 0, make it completely 1
    depth_scaled = jnp.where(
        jnp.all(depth_scaled == 0, axis=(-3, -2), keepdims=True), 1.0, depth_scaled
    )
    height, width = depth_scaled.shape[-3:-1]
    y, x = jnp.meshgrid(
        jnp.linspace(0, 1, height), jnp.linspace(0, 1, width), indexing="ij"
    )
    img_coords = jnp.broadcast_to(
        jnp.stack((y, x), axis=-1), depth_scaled.shape[:-1] + (2,)
    )
    nocs_coords = jnp.concatenate([img_coords, depth_scaled], axis=-1)
    return jnp.moveaxis((nocs_coords - 0.5) / 0.5, -1, -3)


class CycleGANRendererJAX(TactileRendererJAX):
    def __init__(self, device: Device | None = None):
        super().__init__(channels=3, device=device)
        self.__params = jax.device_put(
            load_generator_params_jax(CYCLE_GAN_CHECKPOINT), self.jax_device
        )
        self.__depth_renderer = DepthRendererJAX(device=device)
        self.__generator = jax.jit(apply_generator_jax)

    def get_desired_depth_map_size(
        self, output_size: tuple[int, int]
    ) -> tuple[int, int]:
        return 256, 256

    def _render_direct(
        self, depth: jax.Array, output_size: tuple[int, int]
    ) -> jax.Array:
        depth_scaled = self.__depth_renderer._render_direct(depth, output_size)
        tactile_img_norm = self.__generator(
            self.__params, encode_depth_jax(depth_scaled)
        )
        return jnp.moveaxis((tactile_img_norm + 1.0) / 2.0, -3, -1)
