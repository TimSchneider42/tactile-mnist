import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import threading
import time
from typing import Callable

import datasets
import numpy as np
import pytest
import trimesh

from tactile_mnist import (
    PrefetchedDataset,
    SimpleMeshDataset,
    TactileClassificationVectorEnv,
    TactilePerceptionConfig,
)
from tactile_mnist.dataset import Dataset

NUM_ENVS = 2
NUM_OBJECTS = 8
STEP_LIMIT = 2


class RecordingDataset(Dataset[int, "RecordingDataset"]):
    def __init__(self, length: int = NUM_OBJECTS, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.load_log: list[tuple[int, str]] = []
        self.load_lock = threading.Lock()

    def _get_item(self, index: int) -> int:
        with self.load_lock:
            self.load_log.append((index, threading.current_thread().name))
        return index * 2

    def _select(self, indices: np.ndarray) -> "RecordingDataset":
        raise NotImplementedError

    def _get_length(self) -> int:
        return self.length

    @property
    def loaded_indices(self) -> list[int]:
        with self.load_lock:
            return [index for index, _ in self.load_log]

    @property
    def load_threads(self) -> list[str]:
        with self.load_lock:
            return [thread for _, thread in self.load_log]


def _wait_for(condition: Callable[[], bool], timeout: float = 5.0):
    end = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < end, "Timed out waiting for condition."
        time.sleep(0.005)


@pytest.fixture(scope="module")
def mesh_dataset() -> SimpleMeshDataset:
    meshes = [
        trimesh.creation.box(extents=(0.02, 0.03, 0.005 + 0.001 * i))
        for i in range(NUM_OBJECTS)
    ]
    return SimpleMeshDataset(
        datasets.Dataset.from_dict(
            {
                "id": list(range(NUM_OBJECTS)),
                "label": [i % 2 for i in range(NUM_OBJECTS)],
                "mesh.vertices": [m.vertices.tolist() for m in meshes],
                "mesh.faces": [m.faces.tolist() for m in meshes],
            },
            features=datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "label": datasets.ClassLabel(names=["even", "odd"]),
                    "mesh.vertices": datasets.Sequence(
                        datasets.Sequence(datasets.Value("float64"), length=3)
                    ),
                    "mesh.faces": datasets.Sequence(
                        datasets.Sequence(datasets.Value("int64"), length=3)
                    ),
                }
            ),
        )
    )


def test_in_order_collection_returns_prefetched_values():
    ds = RecordingDataset()
    with PrefetchedDataset(ds, capacity=1) as prefetched:
        prefetched.prefetch(range(len(ds)))
        assert [prefetched[i] for i in range(len(ds))] == [
            i * 2 for i in range(len(ds))
        ]
    assert ds.loaded_indices == list(range(len(ds)))
    assert all(name.startswith("RecordingDataset-prefetch") for name in ds.load_threads)


def test_capacity_limits_lookahead():
    ds = RecordingDataset()
    with PrefetchedDataset(ds) as prefetched:
        prefetched.prefetch(range(len(ds)))
        # With the default capacity of 0, only the next element may be loaded before anything is collected
        _wait_for(lambda: len(ds.load_log) == 1)
        time.sleep(0.1)
        assert ds.loaded_indices == [0]
        assert prefetched[0] == 0
        _wait_for(lambda: len(ds.load_log) == 2)
        time.sleep(0.1)
        assert ds.loaded_indices == [0, 1]
        assert [prefetched[i] for i in range(1, len(ds))] == [
            i * 2 for i in range(1, len(ds))
        ]
    assert ds.loaded_indices == list(range(len(ds)))


def test_out_of_order_collection_discards_preceding_elements():
    ds = RecordingDataset()
    with PrefetchedDataset(ds, capacity=len(ds)) as prefetched:
        prefetched.prefetch(range(len(ds)))
        assert prefetched[3] == 6
        assert prefetched[4] == 8
        # Skipping backwards leaves the requested element unscheduled, so it is loaded on the fly
        assert prefetched[0] == 0
        assert ds.load_threads[-1] == threading.current_thread().name
        # The on-the-fly load cleared the queues, so later elements load on the fly as well
        assert prefetched[6] == 12
        assert ds.load_threads[-1] == threading.current_thread().name


def test_out_of_order_collection_clears_input_queue():
    ds = RecordingDataset()
    with PrefetchedDataset(ds) as prefetched:
        # With capacity 0, most of the input queue is not even loaded when element 5 is requested
        prefetched.prefetch(range(len(ds)))
        assert prefetched[5] == 10
        assert prefetched[6] == 12
        assert prefetched[7] == 14
    assert 5 in ds.loaded_indices
    # The elements between the last loaded in-order element and the requested one were never loaded
    assert 4 not in ds.loaded_indices


def test_unscheduled_access_loads_on_the_fly():
    ds = RecordingDataset()
    with PrefetchedDataset(ds) as prefetched:
        assert prefetched[3] == 6
    assert ds.load_threads == [threading.current_thread().name]


def test_prefetch_propagates_exceptions():
    class FailingDataset(RecordingDataset):
        def _get_item(self, index: int) -> int:
            raise ValueError("boom")

    with PrefetchedDataset(FailingDataset()) as prefetched:
        prefetched.prefetch([0])
        with pytest.raises(ValueError, match="boom"):
            prefetched[0]


def test_negative_and_out_of_bounds_indices():
    ds = RecordingDataset()
    with PrefetchedDataset(ds) as prefetched:
        prefetched.prefetch([-1])
        assert prefetched[len(ds) - 1] == (len(ds) - 1) * 2
        assert len(ds.load_log) == 1
        with pytest.raises(IndexError):
            prefetched.prefetch([len(ds)])


def test_prefetch_requires_open():
    ds = RecordingDataset()
    prefetched = PrefetchedDataset(ds)
    with pytest.raises(RuntimeError):
        prefetched.prefetch([0])
    # Plain access works without opening
    assert prefetched[1] == 2
    with prefetched:
        prefetched.prefetch([0])
        assert prefetched[0] == 0
    with pytest.raises(RuntimeError):
        prefetched.prefetch([0])


def test_close_joins_loader_thread():
    ds = RecordingDataset()
    with PrefetchedDataset(ds) as prefetched:
        prefetched.prefetch(range(len(ds)))
    assert not any(
        t.name.startswith("RecordingDataset-prefetch") for t in threading.enumerate()
    )


def test_mesh_load_fn_forces_mesh_load(mesh_dataset):
    with PrefetchedDataset(mesh_dataset, load_fn=lambda dp: dp.mesh) as prefetched:
        prefetched.prefetch([2])
        dp = prefetched[2]
    # The mesh must already be materialized in the data point's lazy field cache by the loader thread
    assert dp._HuggingfaceDatapoint__fetch_value_cached.cache_info().currsize >= 1
    assert dp.id == 2
    np.testing.assert_allclose(
        dp.mesh.extents, (0.02, 0.03, 0.005 + 0.001 * 2), atol=1e-9
    )


def _mk_env(dataset: SimpleMeshDataset) -> TactileClassificationVectorEnv:
    return TactileClassificationVectorEnv(
        TactilePerceptionConfig(
            dataset,
            step_limit=STEP_LIMIT,
            sensor_output_size=(32, 32),
            allow_sensor_rotation=False,
            sensor_type="depth",
            sensor_backend="numpy",
        ),
        NUM_ENVS,
    )


def test_env_uses_prefetched_datapoints(mesh_dataset):
    env = _mk_env(mesh_dataset)
    env.reset(seed=5)
    env.action_space.seed(5)
    for _ in range(4 * STEP_LIMIT):
        prefetched_indices = list(
            env._TactilePerceptionVectorEnv__prefetched_data_point_indices
        )
        assert all(idx is not None for idx in prefetched_indices)
        previous_indices = env.current_data_point_indices.copy()
        env.step(env.action_space.sample())
        # Whenever an env resets, it must use the datapoint whose load was prefetched during the previous reset
        for i in range(NUM_ENVS):
            if env.current_data_point_indices[i] != previous_indices[i]:
                assert env.current_data_point_indices[i] == prefetched_indices[i]
    env.close()


def test_env_datapoint_idx_option_overrides_prefetch(mesh_dataset):
    env = _mk_env(mesh_dataset)
    env.reset(seed=7)
    env.reset(options={"datapoint_idx": [3, 1]})
    np.testing.assert_array_equal(env.current_data_point_indices, [3, 1])
    assert [dp.id for dp in env.current_data_points] == [3, 1]
    env.close()


def test_env_close_shuts_down_prefetching(mesh_dataset):
    threads_before = set(threading.enumerate())
    env = _mk_env(mesh_dataset)
    env.reset(seed=3)
    env.close()
    new_prefetch_threads = [
        t
        for t in threading.enumerate()
        if t not in threads_before and t.name.startswith("SimpleMeshDataset-prefetch")
    ]
    assert not new_prefetch_threads
