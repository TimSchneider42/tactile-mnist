from abc import abstractmethod, ABC

import jax
import numpy as np

from .tactile_renderer import TactileRenderer, Device, RenderDirectOutput


class TactileRendererJAX(TactileRenderer[jax.Array], ABC):
    def __init__(
        self, jit_all: bool = False, channels: int = 3, device: Device | None = None
    ):
        if device is None:
            self.__jax_device = jax.devices()[0]
        else:
            self.__jax_device = jax.devices(device.platform)[device.device_index]
        super().__init__(
            device=Device(self.__jax_device.platform, self.__jax_device.id),
            backend_name="jax",
            channels=channels,
        )
        self.__render_direct_fn = (
            jax.jit(self.__render_direct, static_argnames=("output_size",))
            if jit_all
            else lambda d, s: self.__render_direct(
                jax.device_put(d, self.__jax_device), s
            )
        )

    def render(self, depth: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
        return np.array(self.__render_direct_fn(depth, output_size)[0])

    def render_direct(
        self, depth: np.ndarray, output_size: tuple[int, int]
    ) -> RenderDirectOutput[jax.Array]:
        return RenderDirectOutput(*self.__render_direct_fn(depth, output_size))

    def __render_direct(
        self, depth: jax.Array, output_size: tuple[int, int]
    ) -> tuple[jax.Array, jax.Array]:
        img = self._render_direct(depth, output_size)
        target_shape = img.shape[:-3] + (output_size[1], output_size[0], img.shape[-1])
        if img.shape != target_shape:
            img = jax.image.resize(
                img, target_shape, method="bicubic", antialias=True
            ).clip(0, 1)
        return img, depth

    @abstractmethod
    def _render_direct(
        self, depth: jax.Array, output_size: tuple[int, int]
    ) -> jax.Array:
        pass

    @property
    def jax_device(self) -> jax.Device:
        return self.__jax_device
