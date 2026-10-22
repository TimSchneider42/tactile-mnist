"""Tests for the cached per-object dataset statistics (tactile_mnist.util.get_dataset_stats)."""

import numpy as np
import pytest
from datasets import load_dataset

import tactile_mnist.util as util
from tactile_mnist import SimpleMeshDataset
from tactile_mnist.util import get_dataset_stats

NUM_OBJECTS = 8


def _extract(idx: int, ds, scale: float):
    mesh = SimpleMeshDataset(ds)[idx].mesh
    return {"volume": scale * mesh.volume, "extents": np.asarray(mesh.extents)}


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(util, "CACHE_BASE_DIR", tmp_path)
    return SimpleMeshDataset(
        load_dataset("TimSchneider42/tactile-mnist-mnist3d", split="train").select(
            range(NUM_OBJECTS)
        )
    )


def test_values_are_stacked_in_dataset_order(dataset):
    stats = get_dataset_stats(dataset, "test", _extract, kwargs=dict(scale=2.0))
    assert stats["volume"].shape == (NUM_OBJECTS,)
    assert stats["extents"].shape == (NUM_OBJECTS, 3)
    for i in range(NUM_OBJECTS):
        assert stats["volume"][i] == pytest.approx(2.0 * dataset[i].volume)
        np.testing.assert_allclose(stats["extents"][i], dataset[i].mesh.extents)


def test_cache_roundtrip(dataset, tmp_path):
    computed = get_dataset_stats(dataset, "test", _extract, kwargs=dict(scale=1.0))
    assert len(list((tmp_path / "test").glob("*.npz"))) == 1
    cached = get_dataset_stats(dataset, "test", _extract, kwargs=dict(scale=1.0))
    assert cached.keys() == computed.keys()
    for k in computed:
        np.testing.assert_array_equal(cached[k], computed[k])
    # Different kwargs get their own cache entry
    get_dataset_stats(dataset, "test", _extract, kwargs=dict(scale=3.0))
    assert len(list((tmp_path / "test").glob("*.npz"))) == 2
