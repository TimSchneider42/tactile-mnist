"""Tests for the silhouette deduplication of the Minecraft item dataset."""

import numpy as np
import pytest

from tactile_mnist.minecraft_dataset import (
    group_textures_by_silhouette,
    silhouettes_equal,
)

# An asymmetric L-shaped silhouette, so that every rotation is distinct
L_SHAPE = np.zeros((16, 16), dtype=np.bool_)
L_SHAPE[3:10, 4] = True
L_SHAPE[9, 4:8] = True


def _texture(mask: np.ndarray) -> np.ndarray:
    texture = np.zeros(mask.shape + (4,), dtype=np.uint8)
    texture[mask] = (255, 0, 0, 255)
    return texture


@pytest.mark.parametrize("k", range(4))
@pytest.mark.parametrize("shift", [(0, 0), (2, -3), (-1, 3)])
def test_shifted_and_rotated_silhouettes_are_equal(k, shift):
    other = np.roll(np.rot90(L_SHAPE, k), shift, axis=(0, 1))
    assert silhouettes_equal(L_SHAPE, other)
    assert silhouettes_equal(other, L_SHAPE)


def test_mirrored_silhouette_is_not_equal():
    assert not silhouettes_equal(L_SHAPE, L_SHAPE[:, ::-1])


def test_different_pixel_count_is_not_equal():
    other = L_SHAPE.copy()
    other[0, 0] = True
    assert not silhouettes_equal(L_SHAPE, other)


def test_same_pixel_count_different_shape_is_not_equal():
    line = np.zeros_like(L_SHAPE)
    line[2, : np.count_nonzero(L_SHAPE)] = True
    assert not silhouettes_equal(L_SHAPE, line)


def test_shifted_items_are_grouped():
    textures = {
        "bucket": _texture(L_SHAPE),
        "tadpole_bucket": _texture(np.roll(L_SHAPE, 1, axis=0)),
        "water_bucket": _texture(np.rot90(L_SHAPE)),
        "stick": _texture(L_SHAPE[:, ::-1]),
    }
    names = [name for name, _ in group_textures_by_silhouette(textures)]
    assert names == ["bucket", "stick"]
