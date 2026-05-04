from __future__ import annotations

from abc import abstractmethod
from typing import SupportsInt, Sequence

import PIL.Image
import numpy as np

from ap_gym.envs.dataset import Dataset


class ImageClassificationDataset(
    Dataset[tuple[np.ndarray, int], tuple[np.ndarray, np.ndarray]]
):
    @abstractmethod
    def _get_num_classes(self) -> int:
        pass

    @abstractmethod
    def _get_num_channels(self) -> int:
        pass

    def __has_overridden(self, method_name: str) -> bool:
        """Checks if the method is overridden in a subclass"""
        super_method = getattr(ImageClassificationDataset, method_name)
        actual_method = getattr(type(self), method_name)
        return super_method != actual_method

    def _get_data_point(self, idx: int) -> tuple[np.ndarray | PIL.Image, SupportsInt]:
        if self.__has_overridden("_get_data_point_batch"):
            imgs, labels = self._get_data_point_batch(np.array([idx]))
            return next(iter(imgs)), next(iter(labels))
        else:
            raise TypeError(
                "At least one of _get_data_point or _get_data_point_batch must be implemented."
            )

    def _get_data_point_batch(
        self, idx: np.ndarray
    ) -> tuple[
        Sequence[np.ndarray] | Sequence[PIL.Image] | np.ndarray, Sequence[SupportsInt]
    ]:
        if self.__has_overridden("_get_data_point"):
            return tuple(map(list, zip(*(self._get_data_point(int(i)) for i in idx))))
        else:
            raise TypeError(
                "At least one of _get_data_point or _get_data_point_batch must be implemented."
            )

    def get_data_point(self, idx: SupportsInt) -> tuple[np.ndarray, int]:
        img, label = self._get_data_point(int(idx))
        return self._process_img(img), int(label)

    def get_data_point_batch(
        self, idx: Sequence[SupportsInt] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(idx)
        if idx.shape[0] == 0:
            raise ValueError("Empty index array")
        imgs, labels = self._get_data_point_batch(idx)
        return self._process_img_batch(imgs), np.asarray(labels).astype(np.int32)

    def _process_img(self, img: PIL.Image | np.ndarray) -> np.ndarray:
        return self._process_img_batch([img])[0]

    def _process_imgs_np(self, imgs: np.ndarray) -> np.ndarray:
        if imgs.dtype == np.uint8:
            imgs = imgs.astype(np.float32) / 255
        elif imgs.dtype != np.float32:
            imgs = imgs.astype(np.float32)
        if len(imgs.shape) == 3:
            imgs = imgs[..., None]
        target_channels = self._get_num_channels()
        if target_channels not in [1, 3]:
            raise ValueError(
                f"Target channels must be either 1 or 3 but is {target_channels}."
            )
        if imgs.shape[-1] == 1 and target_channels == 3:
            imgs = np.repeat(imgs, 3, axis=-1)
        if imgs.shape[-1] != target_channels:
            raise ValueError(
                f"Invalid image format. Expected {target_channels} channels but got {imgs.shape[-1]}"
            )
        return imgs

    def _process_img_batch(
        self, imgs: Sequence[np.ndarray] | Sequence[PIL.Image] | np.ndarray
    ) -> np.ndarray:
        if isinstance(imgs, np.ndarray):
            return self._process_imgs_np(imgs)
        else:
            return np.stack(
                [self._process_imgs_np(np.asarray([img]))[0] for img in imgs]
            )

    @property
    def num_classes(self) -> int:
        return self._get_num_classes()
