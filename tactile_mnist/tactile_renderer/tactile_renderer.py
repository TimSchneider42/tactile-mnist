from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

import numpy as np


@dataclass(frozen=True)
class Device:
    platform: str
    device_index: int = 0

    def __str__(self):
        return f"{self.platform}:{self.device_index}"


OutputType = TypeVar("OutputType")


@dataclass(frozen=True)
class RenderDirectOutput(Generic[OutputType]):
    tactile_image: OutputType
    depth_map: OutputType


class TactileRenderer(ABC, Generic[OutputType]):
    """
    Renders batches of depth maps into tactile images.

    Renderers may be randomized (see SensorNoiseRenderer): all random draws happen through the `rng` argument of
    `render`/`render_direct` and the `reset` hook, which deterministic renderers ignore.
    """

    def __init__(self, device: Device, backend_name: str, channels: int = 3):
        self.__channels = channels
        self.__device = device
        self.__backend_name = backend_name

    @abstractmethod
    def get_desired_depth_map_size(
        self, output_size: tuple[int, int]
    ) -> tuple[int, int]:
        pass

    @abstractmethod
    def render(
        self,
        depth: np.ndarray,
        output_size: tuple[int, int],
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def render_direct(
        self,
        depth: np.ndarray,
        output_size: tuple[int, int],
        rng: np.random.Generator | None = None,
    ) -> RenderDirectOutput[OutputType]:
        pass

    def reset(self, mask: Sequence[bool], rng: np.random.Generator) -> None:
        """Notify the renderer that the environments selected by `mask` start a new episode. No-op by default."""

    def close(self) -> None:
        """Release any resources the renderer holds. No-op by default."""

    def __call__(
        self,
        depth: np.ndarray,
        output_size: tuple[int, int],
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        return self.render(depth, output_size, rng)

    @property
    def channels(self) -> int:
        return self.__channels

    @property
    def device(self) -> Device:
        return self.__device

    @property
    def backend_name(self) -> str:
        return self.__backend_name
