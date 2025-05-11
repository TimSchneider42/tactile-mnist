import functools
from pathlib import Path

import PIL.Image
import torch
from torchvision.transforms import transforms

from tactile_mnist import GELSIGHT_MINI_GEL_THICKNESS_MM
from tactile_mnist.tactile_renderer.depth_renderer_torch import DepthRendererTorch
from tactile_mnist.tactile_renderer.tactile_renderer_torch import TactileRendererTorch
from .cycle_gan_torch import create_g_net
from .tactile_renderer import Device
from importlib.resources import files


class CycleGANRendererTorch(TactileRendererTorch):
    def __init__(self, device: Device | None = None):
        super().__init__(channels=3, device=device)

        state_dict = torch.load(
            Path(
                files("tactile_mnist.resources").joinpath(
                    "cycle_gan_tactile_mnist_v0.pth"
                )
            ),
            map_location=self.torch_device,
        )

        self.__g_model = torch.jit.script(
            create_g_net(
                input_nc=3,
                output_nc=3,
                ngf=64,
                net_g="resnet_9blocks",
                norm="instance",
                use_dropout=False,
            )
        ).to(self.torch_device)
        self.__g_model.load_state_dict(state_dict)
        self.__g_model.eval()

        self.__depth_renderer = DepthRendererTorch(device=device)

        normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        def preprocess_depth(depth: torch.Tensor) -> torch.Tensor:
            # If the depth map is completely 0, make it completely 1
            depth[torch.all(depth == 0, dim=(-1, -2))] = 1

            y, x = torch.meshgrid(
                torch.linspace(0, 1, depth.shape[-2], device=depth.device),
                torch.linspace(0, 1, depth.shape[-1], device=depth.device),
                indexing="ij",
            )
            img_coords = torch.stack((y, x), dim=0)
            img_coords_broadcasted = torch.broadcast_to(
                img_coords[None, :, :, :],
                (
                    depth.shape[0],
                    2,
                    depth.shape[-2],
                    depth.shape[-1],
                ),
            )
            nocs_coords = torch.cat([img_coords_broadcasted, depth], dim=1)
            return normalizer(nocs_coords)

        self.__preprocess_depth = torch.jit.script(preprocess_depth)

    def get_desired_depth_map_size(
        self, output_size: tuple[int, int]
    ) -> tuple[int, int]:
        return 256, 256

    def _render_direct(
        self, depth: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        with torch.no_grad():
            depth_scaled = self.__depth_renderer._render_direct(depth, output_size)
            nocs_coords_norm = self.__preprocess_depth(depth_scaled)
            tactile_img_norm = self.__g_model(nocs_coords_norm)
            tactile_img = (tactile_img_norm + 1.0) / 2.0
            return transforms.Resize(output_size)(tactile_img)
