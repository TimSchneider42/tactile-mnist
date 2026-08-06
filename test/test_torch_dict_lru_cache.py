"""Tests for TorchDictLRUCache."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tactile_mnist import TorchDictLRUCache

NUM_ROWS = 8


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    return torch.device(request.param)


@pytest.fixture
def data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "points": rng.standard_normal((NUM_ROWS, 5, 3)).astype(np.float32),
        "half": rng.standard_normal((NUM_ROWS, 4)).astype(np.float16),
        "label": rng.integers(0, 256, (NUM_ROWS, 7)).astype(np.uint8),
        "scale": rng.standard_normal(NUM_ROWS).astype(np.float32),
        "start": rng.integers(0, 1000, NUM_ROWS),
    }


def assert_rows_equal(
    result: dict[str, torch.Tensor], data: dict[str, np.ndarray], index
) -> None:
    assert result.keys() == data.keys()
    for name, value in result.items():
        expected = data[name][index]
        assert value.shape == expected.shape
        assert value.dtype == torch.from_numpy(expected.reshape(-1)[:0]).dtype
        np.testing.assert_array_equal(value.cpu().numpy(), expected)


def test_validation(data, device):
    with pytest.raises(ValueError, match="at least one array"):
        TorchDictLRUCache({}, capacity=4, device=device)
    with pytest.raises(ValueError, match="capacity"):
        TorchDictLRUCache(data, capacity=0, device=device)
    with pytest.raises(ValueError, match="leading axis length"):
        TorchDictLRUCache(
            {**data, "odd": np.zeros(NUM_ROWS + 1)}, capacity=4, device=device
        )
    with pytest.raises(ValueError, match="leading row axis"):
        TorchDictLRUCache({"scalar": np.float32(0)}, capacity=4, device=device)


def test_sequence_protocol(data, device):
    cache = TorchDictLRUCache(data, capacity=4, device=device)
    assert len(cache) == NUM_ROWS
    assert_rows_equal(cache[2], data, 2)
    assert_rows_equal(cache[-1], data, NUM_ROWS - 1)
    with pytest.raises(IndexError):
        cache[NUM_ROWS]
    with pytest.raises(IndexError):
        cache[-NUM_ROWS - 1]
    for index, element in enumerate(cache):
        assert_rows_equal(element, data, index)


def test_batched_lookup(data, device):
    cache = TorchDictLRUCache(data, capacity=4, device=device)
    indices = [3, 1, 3, -2]
    result = cache.get(indices)
    assert_rows_equal(result, data, np.asarray(indices) % NUM_ROWS)
    # Numpy array, torch tensor, and slice indices are equivalent.
    assert_rows_equal(cache[np.asarray(indices)], data, np.asarray(indices) % NUM_ROWS)
    assert_rows_equal(
        cache[torch.as_tensor(indices, device=device)],
        data,
        np.asarray(indices) % NUM_ROWS,
    )
    assert_rows_equal(cache[1:7:2], data, np.arange(1, 7, 2))
    empty = cache.get([])
    assert all(value.shape[0] == 0 for value in empty.values())


def test_scalar_index_types(data, device):
    cache = TorchDictLRUCache(data, capacity=4, device=device)
    assert_rows_equal(cache[np.int64(3)], data, 3)
    assert_rows_equal(cache[torch.tensor(3)], data, 3)
    with pytest.raises(TypeError, match="integers"):
        cache[1.5]
    with pytest.raises(TypeError, match="integers"):
        cache[np.asarray([True, False])]
    with pytest.raises(ValueError, match="flat"):
        cache[np.zeros((2, 2), np.int64)]


def test_lru_eviction(data, device):
    cache = TorchDictLRUCache(data, capacity=2, device=device)
    cache.get([0, 1])
    assert (cache.misses, cache.hits) == (2, 0)
    cache.get(0)  # Refreshes row 0, making row 1 the eviction victim.
    assert (cache.misses, cache.hits) == (2, 1)
    cache.get(2)
    assert (cache.misses, cache.hits) == (3, 1)
    cache.get(0)
    assert (cache.misses, cache.hits) == (3, 2)
    assert_rows_equal(cache.get(1), data, 1)  # Row 1 was evicted, so this misses.
    assert (cache.misses, cache.hits) == (4, 2)


def test_lookup_pins_requested_rows(data, device):
    cache = TorchDictLRUCache(data, capacity=2, device=device)
    cache.get([0, 1])
    # Row 1 is requested (and refreshed) in the same lookup that misses row 2, so
    # row 0 must be evicted even though row 1 is the nominal LRU victim first.
    result = cache.get([1, 2, 1])
    assert_rows_equal(result, data, [1, 2, 1])
    assert (cache.misses, cache.hits) == (3, 2)
    cache.get([1, 2])
    assert (cache.misses, cache.hits) == (3, 4)


def test_lookup_larger_than_capacity(data, device):
    cache = TorchDictLRUCache(data, capacity=2, device=device)
    with pytest.raises(ValueError, match="exceeds the cache capacity"):
        cache.get([0, 1, 2])
    # Duplicates only count once towards the distinct row limit.
    assert_rows_equal(cache.get([0, 1, 0, 1]), data, [0, 1, 0, 1])


def test_results_survive_eviction(data, device):
    cache = TorchDictLRUCache(data, capacity=1, device=device)
    result = cache.get(0)
    cache.get(1)  # Overwrites row 0's slot in place.
    assert_rows_equal(result, data, 0)


def test_capacity_clamped_to_rows(data, device):
    cache = TorchDictLRUCache(data, capacity=1000, device=device)
    assert cache.capacity == NUM_ROWS
    assert_rows_equal(cache.get(list(range(NUM_ROWS))), data, np.arange(NUM_ROWS))
