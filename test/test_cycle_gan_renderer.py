"""Tests for the CycleGAN tactile renderer.

The JAX renderer is a re-implementation of the PyTorch one and runs on the same weights, so both backends have to
produce the same images.
"""

import numpy as np
import pytest

from tactile_mnist import GELSIGHT_MINI_GEL_THICKNESS_MM
from tactile_mnist.tactile_renderer import mk_tactile_renderer
from tactile_mnist.tactile_renderer.factory import (
    TACTILE_RENDERER_FACTORIES,
    ModuleNotLoaded,
)

OUTPUT_SIZE = (64, 64)

AVAILABLE_BACKENDS = [
    backend
    for backend, factory in TACTILE_RENDERER_FACTORIES["cycle_gan"].items()
    if not isinstance(factory, ModuleNotLoaded)
]

requires_both_backends = pytest.mark.skipif(
    len(AVAILABLE_BACKENDS) < 2, reason="requires both the torch and the jax backend"
)


@pytest.fixture(scope="module")
def depth() -> np.ndarray:
    """Two depth maps in meters: a rounded indentation and a surface the sensor does not touch."""
    y, x = np.mgrid[0:256, 0:256] / 255
    indented = GELSIGHT_MINI_GEL_THICKNESS_MM / 1000 - 2e-3 * np.exp(
        -(((y - 0.5) ** 2 + (x - 0.4) ** 2) / 0.02)
    )
    untouched = np.full_like(indented, GELSIGHT_MINI_GEL_THICKNESS_MM / 1000)
    return np.stack([indented, untouched]).astype(np.float32)


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
def test_renders_images_of_the_requested_size(depth, backend):
    renderer = mk_tactile_renderer("cycle_gan", backend=backend)
    image = renderer.render(depth, OUTPUT_SIZE)
    assert image.shape == (len(depth), *OUTPUT_SIZE[::-1], 3)
    assert np.all((image >= 0) & (image <= 1))
    # The indented and the untouched depth map must not render to the same image
    assert not np.allclose(image[0], image[1])


@requires_both_backends
def test_backends_agree(depth):
    images = [
        mk_tactile_renderer("cycle_gan", backend=backend).render(depth, OUTPUT_SIZE)
        for backend in AVAILABLE_BACKENDS
    ]
    # The backends resize with different bicubic implementations, so they differ by well below one uint8 step
    assert images[0] == pytest.approx(images[1], abs=1 / 255)


@requires_both_backends
def test_backends_agree_before_the_resize(depth):
    """Compare the raw generator output, without the resampling the backends do differently."""
    images = [
        mk_tactile_renderer("cycle_gan", backend=backend).render(depth, (256, 256))
        for backend in AVAILABLE_BACKENDS
    ]
    assert images[0] == pytest.approx(images[1], abs=1e-3)
