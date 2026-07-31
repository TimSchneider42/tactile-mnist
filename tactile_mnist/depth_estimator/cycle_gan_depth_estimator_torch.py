from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional

from .depth_estimator import DEPTH_OFFSET_M, DEPTH_SCALE_M, DepthEstimator
from ..tactile_renderer import Device
from ..tactile_renderer.cycle_gan_renderer_torch import load_generator_state_dict_torch
from ..tactile_renderer.cycle_gan_torch import create_g_net

CYCLE_GAN_INVERSE_CHECKPOINT = Path(
    files("tactile_mnist.resources").joinpath("cycle_gan_inverse_tactile_mnist_v0.npz")
)


class CycleGANDepthEstimatorTorch(DepthEstimator):
    """Estimates depth maps with the inverse of the generator CycleGANRendererTorch uses.

    The two generators are the two directions of the same CycleGAN, so chaining this estimator and
    CycleGANRendererTorch maps a real tactile image into the domain of the simulated environments.
    """

    def __init__(self, device: Device | None = None):
        if device is None:
            self.__torch_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif device.platform == "cpu":
            # If we pass the index here, checkpoint loading stops working
            self.__torch_device = torch.device("cpu")
        else:
            self.__torch_device = torch.device(str(device))
        super().__init__(
            device=Device(self.__torch_device.type, self.__torch_device.index),
            backend_name="torch",
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
        ).to(self.__torch_device)
        self.__g_model.load_state_dict(
            load_generator_state_dict_torch(
                CYCLE_GAN_INVERSE_CHECKPOINT, self.__torch_device
            )
        )
        self.__g_model.eval()

    def estimate(self, images: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
        with torch.no_grad():
            images = np.ascontiguousarray(images)
            if not images.flags.writeable:
                # torch.from_numpy does not accept read-only arrays
                images = images.copy()
            img = torch.from_numpy(images).to(self.__torch_device)
            if img.dtype == torch.uint8:
                img = img.to(torch.float32) / 255
            else:
                img = img.to(torch.float32)
            img = torch.movedim(img, -1, -3)
            input_width, input_height = self.input_size
            if img.shape[-2:] != (input_height, input_width):
                img = torchvision.transforms.functional.resize(
                    img,
                    [input_height, input_width],
                    torchvision.transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ).clip(0, 1)
            # The generator was trained on images normalized to [-1, 1] and predicts the image coordinates in its
            # first two channels and the depth in its third one
            depth_scaled = (self.__g_model((img - 0.5) / 0.5)[..., 2, :, :] + 1) / 2
            if depth_scaled.shape[-2:] != (output_size[1], output_size[0]):
                depth_scaled = torchvision.transforms.functional.resize(
                    depth_scaled[..., None, :, :],
                    [output_size[1], output_size[0]],
                    torchvision.transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                )[..., 0, :, :]
            depth = depth_scaled.clip(0, 1) * DEPTH_SCALE_M + DEPTH_OFFSET_M
            return depth.cpu().numpy().astype(np.float32)

    @property
    def torch_device(self) -> torch.device:
        return self.__torch_device
