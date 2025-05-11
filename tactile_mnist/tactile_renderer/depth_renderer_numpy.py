import numpy as np

from tactile_mnist import GELSIGHT_MINI_GEL_THICKNESS_MM, GEL_PENETRATION_DEPTH_MM
from .tactile_renderer import TactileRenderer, Device, RenderDirectOutput


class DepthRendererNumpy(TactileRenderer[np.ndarray]):
    def __init__(self, device: Device | None = None):
        if device is None:
            device = Device("cpu")
        elif device.platform != "cpu":
            raise ValueError("DepthRendererNumpy only supports CPU.")
        super().__init__(device=device, backend_name="numpy", channels=1)

    def get_desired_depth_map_size(
        self, output_size: tuple[int, int]
    ) -> tuple[int, int]:
        return output_size

    def render_direct(
        self, depth: np.ndarray, output_size: tuple[int, int]
    ) -> RenderDirectOutput[np.ndarray]:
        gel_thickness_m = GELSIGHT_MINI_GEL_THICKNESS_MM / 1000
        gel_penetration_depth_m = GEL_PENETRATION_DEPTH_MM / 1000
        depth_clipped = np.clip(depth, gel_penetration_depth_m, gel_thickness_m)
        return RenderDirectOutput(
            (depth_clipped[..., None] - gel_penetration_depth_m)
            / (gel_thickness_m - gel_penetration_depth_m),
            depth,
        )

    def render(self, depth: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
        return self.render_direct(depth, output_size).tactile_image
