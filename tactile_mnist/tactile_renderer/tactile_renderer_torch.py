from abc import abstractmethod, ABC

import numpy as np
import torch
import torchvision.transforms.functional

from .tactile_renderer import TactileRenderer, Device, RenderDirectOutput


class TactileRendererTorch(TactileRenderer[torch.Tensor], ABC):
    def __init__(self, channels: int = 3, device: Device | None = None):
        if device is None:
            if torch.cuda.is_available():
                self.__torch_device = torch.device("cuda")
            else:
                torch.device("cpu")
        else:
            if device.platform == "cpu":
                # If we pass the index here, checkpoint loading stops working
                self.__torch_device = torch.device("cpu")
            else:
                self.__torch_device = torch.device(str(device))
        super().__init__(
            device=Device(self.__torch_device.type, self.__torch_device.index),
            backend_name="torch",
            channels=channels,
        )

    def render(self, depth: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
        return np.moveaxis(
            self.render_direct(depth, output_size).tactile_image.cpu().numpy(),
            [-3],
            [-1],
        )

    def render_direct(
        self, depth: np.ndarray, output_size: tuple[int, int]
    ) -> RenderDirectOutput[torch.Tensor]:
        depth_conv = torch.from_numpy(depth.astype(np.float32)).to(self.__torch_device)
        img = self._render_direct(depth_conv, output_size)
        target_shape = img.shape[:-3] + (img.shape[-3], output_size[1], output_size[0])
        if img.shape != target_shape:
            img = torchvision.transforms.functional.resize(
                img,
                [output_size[1], output_size[0]],
                torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ).clip(0, 1)
        return RenderDirectOutput(img, depth_conv)

    @abstractmethod
    def _render_direct(
        self, depth: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        pass

    @property
    def torch_device(self) -> torch.device:
        return self.__torch_device
