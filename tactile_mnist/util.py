from __future__ import annotations

from typing import Sequence, Any, Callable, TypeVar, Generic, overload

import numpy as np
from transformation import Transformation


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
    def __get__(self, instance: InstanceType, owner: Any) -> DynamicType:
        ...

    @overload
    def __get__(self, instance: None, owner: Any) -> StaticType:
        ...

    def __get__(self, instance: InstanceType | None = None, owner: Any = None):
        if self._dynamic_value_fn is None or instance is None:
            return self._static_value
        return self._dynamic_value_fn(instance)
