import torch
from taxim import CALIB_GELSIGHT_MINI
from taxim.taxim_torch import TaximTorch

from tactile_mnist import GELSIGHT_MINI_GEL_THICKNESS_MM
from .tactile_renderer_torch import TactileRendererTorch
from .tactile_renderer import Device


class TaximRendererTorch(TactileRendererTorch):
    def __init__(self, device: Device | None = None):
        super().__init__(channels=3, device=device)
        self.__taxim = TaximTorch(
            calib_folder=CALIB_GELSIGHT_MINI,
            device=self.torch_device,
            params={"simulator": {"contact_scale": 0.6}},
        )

    def get_desired_depth_map_size(
        self, output_size: tuple[int, int]
    ) -> tuple[int, int]:
        return output_size

    def _render_direct(
        self, depth: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        return self.__taxim.render_direct(depth * 1000 - GELSIGHT_MINI_GEL_THICKNESS_MM)
