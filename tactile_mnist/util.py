from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import pickle
from functools import partial
from typing import Sequence, Any, Callable, TypeVar, Generic, overload

import filelock
import numpy as np
import tqdm
from transformation import Transformation

from tactile_mnist import CACHE_BASE_DIR, MeshDataset

logger = logging.getLogger(__name__)


def transformation_where(
    condition: Sequence[bool], true_trans: Transformation, false_trans: Transformation
):
    return Transformation.batch_concatenate(
        [t if c else f for c, t, f in zip(condition, true_trans, false_trans)]
    )


def dict_where(
    condition: Sequence[bool],
    true_dict: dict[str, np.ndarray],
    false_dict: dict[str, np.ndarray],
):
    return {k: dynamic_where(condition, true_dict[k], false_dict[k]) for k in true_dict}


def dynamic_where(condition: Sequence[bool], true_val: Any, false_val: Any):
    condition = np.asarray(condition)
    if isinstance(true_val, dict):
        return dict_where(condition, true_val, false_val)
    if isinstance(true_val, Transformation):
        return transformation_where(condition, true_val, false_val)
    return np.where(
        condition.reshape((condition.shape[0],) + (1,) * (len(true_val.shape) - 1)),
        true_val,
        false_val,
    )


StaticType = TypeVar("StaticType")
DynamicType = TypeVar("DynamicType")
InstanceType = TypeVar("InstanceType")


class OverridableStaticField(Generic[InstanceType, StaticType, DynamicType]):
    def __init__(self, static_value: StaticType):
        self._dynamic_value_fn: Callable[[InstanceType], DynamicType] | None = None
        self._static_value = static_value

    def dynamic_update(self, fn: Callable[[InstanceType], DynamicType]):
        self._dynamic_value_fn = fn
        return self

    @overload
    def __get__(self, instance: InstanceType, owner: Any) -> DynamicType: ...

    @overload
    def __get__(self, instance: None, owner: Any) -> StaticType: ...

    def __get__(self, instance: InstanceType | None = None, owner: Any = None):
        if self._dynamic_value_fn is None or instance is None:
            return self._static_value
        return self._dynamic_value_fn(instance)


def get_dataset_stats(
    dataset: MeshDataset,
    stats_name: str,
    extraction_fn: Callable[[int, MeshDataset, ...], dict[str, float]],
    kwargs: dict[str, Any] | None = None,
):
    if kwargs is None:
        kwargs = {}
    kwargs_signature_tuple = tuple((k, kwargs[k]) for k in sorted(kwargs.keys()))
    kwargs_signature_string = hashlib.sha256(
        pickle.dumps(kwargs_signature_tuple)
    ).hexdigest()[:16]
    cache_dir = CACHE_BASE_DIR / stats_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds_fingerprint = dataset.huggingface_dataset._fingerprint
    cache_file = cache_dir / f"{ds_fingerprint}_{kwargs_signature_string}.json"
    with filelock.FileLock(cache_dir / f"{ds_fingerprint}.lock"):
        if cache_file.exists():
            try:
                with cache_file.open() as f:
                    return json.load(f)
            except Exception as ex:
                logger.warning(
                    f"Loading {stats_name} statistics from cache failed with the following exception: {ex}"
                )
        print(f"Computing {stats_name} statistics (the results will be cached)...")
        with multiprocessing.pool.Pool(
            processes=min(multiprocessing.cpu_count(), 8)
        ) as pool:
            statistics = list(
                tqdm.tqdm(
                    pool.imap_unordered(
                        partial(
                            extraction_fn,
                            ds=dataset.huggingface_dataset,
                            **kwargs,
                        ),
                        range(len(dataset)),
                    ),
                    total=len(dataset),
                )
            )

        if len(statistics) == 0:
            raise ValueError("Cannot compute statistics on empty dataset.")

        def extract_stats(data: list[float]) -> dict[str, float]:
            arr = np.asarray(data)
            return {
                "mean": np.mean(arr),
                "std": np.std(arr),
                "min": np.min(arr),
                "max": np.max(arr),
            }

        statistics = {
            k: extract_stats([s[k] for s in statistics]) for k in statistics[0]
        }

        with cache_file.open("w") as f:
            json.dump(statistics, f)

        return statistics
