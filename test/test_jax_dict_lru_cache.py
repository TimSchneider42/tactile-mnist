"""Tests for JaxDictLRUCache."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from tactile_mnist import JaxDictLRUCache

NUM_ROWS = 8


@pytest.fixture(params=["cpu", "gpu"])
def device(request):
    try:
        return jax.devices(request.param)[0]
    except RuntimeError:
        pytest.skip(f"No {request.param} device is available.")


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
    result: "dict[str, jax.Array]", data: dict[str, np.ndarray], index
) -> None:
    assert result.keys() == data.keys()
    for name, value in result.items():
        expected = data[name][index]
        assert value.shape == expected.shape
        assert value.dtype == jax.dtypes.canonicalize_dtype(expected.dtype)
        np.testing.assert_array_equal(np.asarray(value), expected)


def test_validation(data, device):
    with pytest.raises(ValueError, match="at least one array"):
        JaxDictLRUCache({}, capacity=4, device=device)
    with pytest.raises(ValueError, match="capacity"):
        JaxDictLRUCache(data, capacity=0, device=device)
    with pytest.raises(ValueError, match="leading axis length"):
        JaxDictLRUCache(
            {**data, "odd": np.zeros(NUM_ROWS + 1)}, capacity=4, device=device
        )
    with pytest.raises(ValueError, match="leading row axis"):
        JaxDictLRUCache({"scalar": np.float32(0)}, capacity=4, device=device)


def test_sequence_protocol(data, device):
    cache = JaxDictLRUCache(data, capacity=4, device=device)
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
    cache = JaxDictLRUCache(data, capacity=4, device=device)
    indices = [3, 1, 3, -2]
    result = cache.get(indices)
    assert_rows_equal(result, data, np.asarray(indices) % NUM_ROWS)
    # Numpy array, jax array, and slice indices are equivalent.
    assert_rows_equal(cache[np.asarray(indices)], data, np.asarray(indices) % NUM_ROWS)
    assert_rows_equal(
        cache[jax.device_put(np.asarray(indices), device)],
        data,
        np.asarray(indices) % NUM_ROWS,
    )
    assert_rows_equal(cache[1:7:2], data, np.arange(1, 7, 2))
    empty = cache.get([])
    assert all(value.shape[0] == 0 for value in empty.values())


def test_scalar_index_types(data, device):
    cache = JaxDictLRUCache(data, capacity=4, device=device)
    assert_rows_equal(cache[np.int64(3)], data, 3)
    assert_rows_equal(cache[jax.numpy.asarray(3)], data, 3)
    with pytest.raises(TypeError, match="integers"):
        cache[1.5]
    with pytest.raises(TypeError, match="integers"):
        cache[np.asarray([True, False])]
    with pytest.raises(ValueError, match="flat"):
        cache[np.zeros((2, 2), np.int64)]


def test_device_placement(data, device):
    cache = JaxDictLRUCache(data, capacity=4, device=device)
    assert cache.device == device
    result = cache.get([0, 1])
    assert all(value.devices() == {device} for value in result.values())


def test_lru_eviction(data, device):
    cache = JaxDictLRUCache(data, capacity=2, device=device)
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
    cache = JaxDictLRUCache(data, capacity=2, device=device)
    cache.get([0, 1])
    # Row 1 is requested (and refreshed) in the same lookup that misses row 2, so
    # row 0 must be evicted even though row 1 is the nominal LRU victim first.
    result = cache.get([1, 2, 1])
    assert_rows_equal(result, data, [1, 2, 1])
    assert (cache.misses, cache.hits) == (3, 2)
    cache.get([1, 2])
    assert (cache.misses, cache.hits) == (3, 4)


def test_lookup_larger_than_capacity(data, device):
    cache = JaxDictLRUCache(data, capacity=2, device=device)
    with pytest.raises(ValueError, match="exceeds the cache capacity"):
        cache.get([0, 1, 2])
    # Duplicates only count once towards the distinct row limit.
    assert_rows_equal(cache.get([0, 1, 0, 1]), data, [0, 1, 0, 1])


def test_results_survive_eviction(data, device):
    cache = JaxDictLRUCache(data, capacity=1, device=device)
    result = cache.get(0)
    cache.get(1)  # Overwrites row 0's slot in place.
    assert_rows_equal(result, data, 0)


def test_capacity_clamped_to_rows(data, device):
    cache = JaxDictLRUCache(data, capacity=1000, device=device)
    assert cache.capacity == NUM_ROWS
    assert_rows_equal(cache.get(list(range(NUM_ROWS))), data, np.arange(NUM_ROWS))


def test_jit_lookup(data, device):
    cache = JaxDictLRUCache(data, capacity=4, device=device)
    lookup = jax.jit(cache.get)

    def put(indices):
        return jax.device_put(np.asarray(indices), device)

    assert_rows_equal(lookup(put([0, 1, 2])), data, [0, 1, 2])
    assert (cache.misses, cache.hits) == (3, 0)
    # Same shape reuses the compiled executable; the cache state must still
    # advance per call.
    assert_rows_equal(lookup(put([2, 1, 3])), data, [2, 1, 3])
    assert (cache.misses, cache.hits) == (4, 2)
    # Fully hit lookups take the callback-free branch.
    assert_rows_equal(lookup(put([3, 3, 0])), data, [3, 3, 0])
    assert (cache.misses, cache.hits) == (4, 5)
    # Rows 1 and 2 are the least recently used residents, so they are evicted.
    assert_rows_equal(lookup(put([4, 5, 0])), data, [4, 5, 0])
    assert (cache.misses, cache.hits) == (6, 6)
    assert_rows_equal(cache.get(1), data, 1)  # Evicted under jit, so this misses.
    assert (cache.misses, cache.hits) == (7, 6)


def test_jit_full_hit_skips_callback(data, device, monkeypatch):
    cache = JaxDictLRUCache(data, capacity=4, device=device)
    calls = []
    original = cache._JaxDictLRUCache__serve_missing

    def counting(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(cache, "_JaxDictLRUCache__serve_missing", counting)
    lookup = jax.jit(cache.get)
    indices = jax.device_put(np.asarray([0, 1, 2]), device)
    lookup(indices)
    assert calls
    calls.clear()
    lookup(indices)
    jax.block_until_ready(lookup(indices))
    assert not calls


def instrument_staging_sizes(cache, monkeypatch) -> list:
    sizes = []
    original = cache._JaxDictLRUCache__serve_missing

    def recording(size, *args):
        sizes.append(size)
        return original(size, *args)

    monkeypatch.setattr(cache, "_JaxDictLRUCache__serve_missing", recording)
    return sizes


def test_jit_staging_bucket_sizes(device, monkeypatch):
    data = {"x": np.arange(128, dtype=np.float32)}
    cache = JaxDictLRUCache(data, capacity=64, device=device)
    sizes = instrument_staging_sizes(cache, monkeypatch)
    lookup = jax.jit(cache.get)

    def put(indices):
        return jax.device_put(np.asarray(indices, np.int32), device)

    first = list(range(64))
    lookup(put(first))  # 64 misses -> full bucket
    second = list(range(48)) + list(range(64, 80))  # 16 misses
    lookup(put(second))
    third = list(range(48)) + list(range(64, 72)) + list(range(88, 96))  # 8 misses
    result = lookup(put(third))
    np.testing.assert_array_equal(np.asarray(result["x"]), np.asarray(third))
    jax.block_until_ready(lookup(put(third))["x"])  # all hit
    assert sizes == [64, 16, 16]
    assert (cache.misses, cache.hits) == (88, 168)


def test_jit_small_batches_use_full_bucket(data, device, monkeypatch):
    # Batches below the minimum bucket size still get a (request-sized) miss branch.
    cache = JaxDictLRUCache(data, capacity=4, device=device)
    sizes = instrument_staging_sizes(cache, monkeypatch)
    lookup = jax.jit(cache.get)
    indices = jax.device_put(np.asarray([0, 1, 2, 3]), device)
    assert_rows_equal(lookup(indices), data, [0, 1, 2, 3])
    jax.block_until_ready(lookup(indices)["points"])  # all hit
    assert sizes == [4]
    assert (cache.misses, cache.hits) == (4, 4)


def test_jit_and_eager_share_state(data, device):
    cache = JaxDictLRUCache(data, capacity=2, device=device)
    cache.get([0, 1])
    lookup = jax.jit(cache.get)
    result = lookup(jax.device_put(np.asarray([0, 1]), device))
    assert_rows_equal(result, data, [0, 1])
    assert (cache.misses, cache.hits) == (2, 2)


@pytest.mark.parametrize("indices", [[0, 1], [0, 1, 0, 1]])
def test_lookup_is_differentiable(data, device, indices):
    # Anything computed from a lookup must remain differentiable, which requires the
    # storage updates to stay out of the staging switch: jax cannot partial-evaluate
    # a cond whose branches carry state effects.
    cache = JaxDictLRUCache(data, capacity=2, device=device)

    def loss(weights, index):
        return jax.numpy.sum(weights * cache.get(index)["points"])

    index = jax.device_put(np.asarray(indices), device)
    weights = jax.device_put(np.ones((len(indices), 5, 3), np.float32), device)
    expected = data["points"][indices]
    for grad_fn in (jax.grad(loss), jax.jit(jax.grad(loss)), jax.grad(jax.jit(loss))):
        np.testing.assert_allclose(np.asarray(grad_fn(weights, index)), expected)


def test_repeated_eviction_churn(data, device):
    # Rows staged into previously used slots must fully overwrite the evicted rows.
    cache = JaxDictLRUCache(data, capacity=2, device=device)
    for _ in range(3):
        for index in range(NUM_ROWS):
            assert_rows_equal(cache.get(index), data, index)
    assert cache.misses == 3 * NUM_ROWS
