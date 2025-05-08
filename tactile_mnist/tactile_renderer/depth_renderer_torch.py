import torch

from tactile_mnist import GELSIGHT_MINI_GEL_THICKNESS_MM
from tactile_mnist.tactile_renderer.tactile_renderer_torch import TactileRendererTorch
from .tactile_renderer import Device


class DepthRendererTorch(TactileRendererTorch):
    def __init__(self, device: Device | None = None):
        super().__init__(channels=1, device=device)

    def get_desired_depth_map_size(
        self, output_size: tuple[int, int]
    ) -> tuple[int, int]:
        return output_size

    def _render_direct(
        self, depth: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        gel_thickness_m = GELSIGHT_MINI_GEL_THICKNESS_MM / 1000
        depth = torch.clip(depth, 0, gel_thickness_m)
        return depth[..., None, :, :] / gel_thickness_m
