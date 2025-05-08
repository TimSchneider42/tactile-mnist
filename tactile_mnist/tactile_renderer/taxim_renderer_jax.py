import jax
from taxim import CALIB_GELSIGHT_MINI
from taxim.taxim_jax import TaximJax

from tactile_mnist import GELSIGHT_MINI_GEL_THICKNESS_MM
from .tactile_renderer import Device
from .tactile_renderer_jax import TactileRendererJAX


class TaximRendererJAX(TactileRendererJAX):
    def __init__(self, device: Device | None = None):
        super().__init__(jit_all=True, channels=3, device=device)
        self.__taxim = TaximJax(
            calib_folder=CALIB_GELSIGHT_MINI,
            device=self.jax_device,
            params={"simulator": {"contact_scale": 0.6}},
        )

    def get_desired_depth_map_size(
        self, output_size: tuple[int, int]
    ) -> tuple[int, int]:
        return output_size

    def _render_direct(
        self, depth: jax.Array, output_size: tuple[int, int]
    ) -> jax.Array:
        return self.__taxim.render_direct(depth * 1000 - GELSIGHT_MINI_GEL_THICKNESS_MM)
