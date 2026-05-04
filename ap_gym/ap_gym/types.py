from __future__ import annotations

from typing import TypeVar, Literal

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
PredType = TypeVar("PredType")
PredTargetType = TypeVar("PredTargetType")
ArrayType = TypeVar("ArrayType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")
WrapperPredType = TypeVar("WrapperPredType")
WrapperPredTargetType = TypeVar("WrapperPredTargetType")
WrapperArrayType = TypeVar("WrapperArrayType")

FullActType = dict[Literal["action", "prediction"], ActType | PredType]
