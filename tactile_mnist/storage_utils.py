from __future__ import annotations

import dataclasses
import itertools
import typing
from collections import OrderedDict
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, List, Iterator, Union, Optional

import numpy as np

SupportedDtypes = np.ndarray | str | int | float
NestedDict = dict[str, Union[SupportedDtypes, "NestedDict"]]

FieldsType = typing.OrderedDict[str, Optional["FieldType"]]


def nested_dict_to_list(d: dict[str, Any], fields: FieldsType):
    return list(
        chain.from_iterable(
            nested_dict_to_list(d[k], v) if isinstance(d[k], dict) else (d[k],)
            for k, v in fields.items()
        )
    )


def get_fields(d: dict[str, Any]) -> FieldsType:
    return OrderedDict(
        [(k, get_fields(d[k]) if isinstance(d[k], dict) else None) for k in sorted(d)]
    )


def flatten_fields(
    fields: FieldsType, prefix: tuple[str, ...] = ()
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        chain.from_iterable(
            flatten_fields(v, prefix=prefix + (k,))
            if v is not None
            else (prefix + (k,),)
            for k, v in fields.items()
        )
    )


def _maybe_pad(
    val: SupportedDtypes, shape: tuple[int, ...]
) -> tuple[SupportedDtypes, ...]:
    if isinstance(val, np.ndarray):
        val_padded = np.pad(
            val, [(0, s_t - s_v) for s_t, s_v in zip(shape, val.shape)], mode="constant"
        )
        return val_padded, np.array(val.shape)
    else:
        return (val,)


def _pad_numpy_arrays(
    data: List[SupportedDtypes], target_shapes: List[tuple[int, ...] | None]
):
    return tuple(
        e for d, shape in zip(data, target_shapes) for e in _maybe_pad(d, shape)
    )


def _todict(obj: object | NestedDict):
    if isinstance(obj, dict):
        return obj
    else:
        return dataclasses.asdict(obj)


def data_to_struct_array(data: Iterable[object]):
    try:
        first_dp = next(iter(data))
    except StopIteration:
        raise ValueError("Need at least one data point.")
    fields = get_fields(_todict(first_dp))
    fields_flat = flatten_fields(fields)
    data = [nested_dict_to_list(_todict(e), fields) for e in data]
    entry_dtypes = [tuple(type(v) for v in e) for e in data]
    dtypes = entry_dtypes[0]
    assert all(dtypes == e for e in entry_dtypes)

    shapes = []
    out_dtypes = []
    for i, (field_path, dtype) in enumerate(zip(fields_flat, dtypes)):
        name = ".".join(field_path)
        assert dtype in [
            int,
            float,
            str,
            np.ndarray,
        ], f"Unsupported dtype {dtype} for field {name}."
        if dtype == np.ndarray:
            padded_shape = tuple(np.max([e[i].shape for e in data], axis=0).tolist())
            shapes.append(padded_shape)
            out_dtypes.append((name, data[0][i].dtype, padded_shape))
            out_dtypes.append((name + ".shape", np.int_, (len(padded_shape),)))
        else:
            if dtype == str:
                str_maxlen = max(len(e[i]) for e in data)
                out_dtypes.append((name, np.dtype("U"), str_maxlen))
            else:
                out_dtypes.append((name, np.dtype(dtype)))
            shapes.append(None)
    for i in range(len(data)):
        data[i] = _pad_numpy_arrays(data[i], shapes)
    return np.array(data, dtype=out_dtypes)


def flat_dict_to_nested_dict(array_dict: dict[str, SupportedDtypes]) -> NestedDict:
    output = {k: v for k, v in array_dict.items() if "." not in k}
    sub_dicts = {k: v for k, v in array_dict.items() if "." in k}
    sub_dicts_grouped = itertools.groupby(
        sub_dicts.items(), lambda e: e[0].split(".", maxsplit=1)[0]
    )
    output.update(
        {
            k: flat_dict_to_nested_dict(
                {ki.split(".", maxsplit=1)[1]: vi for ki, vi in v}
            )
            for k, v in sub_dicts_grouped
        }
    )
    return output


def _restore_dtypes(array_dict: dict[str, Any]) -> dict[str, SupportedDtypes]:
    return {
        k: v[tuple(slice(0, s) for s in array_dict[f"{k}.shape"])].copy()
        if isinstance(v, np.ndarray)
        else v.item()
        for k, v in array_dict.items()
        if not k.endswith(".shape")
    }


def save_data(
    path: Path,
    data: Iterable[object | NestedDict],
    metadata: NestedDict | None = None,
):
    if metadata is None:
        metadata = {}
    np.savez_compressed(
        path, data=data_to_struct_array(data), metadata=data_to_struct_array([metadata])
    )


def load_data(path: Path) -> tuple[Iterator[NestedDict], NestedDict]:
    with np.load(path, allow_pickle=False) as f:
        struct_array = f["data"]
        metadata = f["metadata"]
    metadata_restored = _restore_dtypes(dict(zip(metadata.dtype.names, metadata[0])))
    metadata_output = flat_dict_to_nested_dict(metadata_restored)
    data_output = (
        flat_dict_to_nested_dict(
            _restore_dtypes(dict(zip(struct_array.dtype.names, e)))
        )
        for e in struct_array
    )
    return data_output, metadata_output
