from __future__ import annotations

from typing import Mapping, Any, Sequence, TypeVar

import numpy as np


def update_dict_recursive(d: dict, u: Mapping):
    return {
        **d,
        **{
            k: update_dict_recursive(d.get(k, {}), v) if isinstance(v, Mapping) else v
            for k, v in u.items()
        },
    }


def update_info_metrics(
    info: dict[str, Any], metrics: dict[str, Sequence[float] | np.ndarray]
) -> dict[str, Any]:
    return update_dict_recursive(
        info,
        {
            "stats": {
                "scalar": {
                    **{f"avg_{n}": float(np.mean(v)) for n, v in metrics.items()},
                    **{f"final_{n}": float(v[-1]) for n, v in metrics.items()},
                },
                "vector": {
                    # Need to use a list and not a numpy array here as otherwise SyncVectorEnv will try to stack the
                    # arrays, which throws an error if the arrays are of different lengths
                    n: list(v)
                    for n, v in metrics.items()
                },
            }
        },
    )


def update_info_metrics_vec(
    info: dict[str, Any],
    metrics: dict[str, Sequence[Sequence[float] | np.ndarray]],
    done: np.ndarray,
) -> dict[str, Any]:
    return update_dict_recursive(
        info,
        {
            "stats": {
                "scalar": {
                    **{
                        f"final_{n}": np.array(
                            [e[-1] if t else np.nan for t, e in zip(done, v)],
                            dtype=np.float32,
                        )
                        for n, v in metrics.items()
                    },
                    **{f"_final_{n}": done for n in metrics.keys()},
                    **{
                        f"avg_{n}": np.array(
                            [np.mean(e) if t else np.nan for t, e in zip(done, v)],
                            dtype=np.float32,
                        )
                        for n, v in metrics.items()
                    },
                    **{f"_avg_{n}": done for n in metrics.keys()},
                },
                "_scalar": done,
                "vector": {
                    **{
                        # The None trick is to ensure that numpy does not try to stack the lists if they happen to have
                        # the same length.
                        n: np.array(
                            [(list(e) if t else []) for e, t in zip(v, done)] + [None],
                            dtype=object,
                        )[:-1]
                        for n, v in metrics.items()
                    },
                    **{f"_{n}": done for n in metrics.keys()},
                },
                "_vector": done,
            }
        },
    )


T = TypeVar("T")


def idoc(obj: T, doc: Any) -> T:
    obj.__idoc__ = doc
    return obj


def project_sphere(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
    magnitude = np.linalg.norm(x, axis=-1, keepdims=True)
    direction = x / np.maximum(magnitude, radius)
    return np.where(magnitude > radius, direction * radius, x)
