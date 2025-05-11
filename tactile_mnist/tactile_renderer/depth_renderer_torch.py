import torch

from tactile_mnist import GELSIGHT_MINI_GEL_THICKNESS_MM, GEL_PENETRATION_DEPTH_MM
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
        gel_penetration_depth_m = GEL_PENETRATION_DEPTH_MM / 1000
        depth = torch.clip(depth, gel_penetration_depth_m, gel_thickness_m)
        return (depth[..., None, :, :] - gel_penetration_depth_m) / (
            gel_thickness_m - gel_penetration_depth_m
        )
