from __future__ import annotations

import logging
from typing import Callable, Literal

from .depth_estimator import DepthEstimator
from ..tactile_renderer.factory import ModuleNotLoaded, device_available
from ..tactile_renderer.tactile_renderer import Device

logger = logging.getLogger(__name__)

_DEPTH_ESTIMATORS_STR = {
    "jax": {
        "cycle_gan": ("cycle_gan_depth_estimator_jax", "CycleGANDepthEstimatorJAX"),
    },
    "torch": {
        "cycle_gan": ("cycle_gan_depth_estimator_torch", "CycleGANDepthEstimatorTorch"),
    },
}

_DISPLAY_NAMES = {"jax": "JAX", "torch": "PyTorch"}

_ESTIMATORS_BY_BACKEND: dict[str, dict[str, Callable[[Device], DepthEstimator]]] = {}

for _backend, _estimators_str in _DEPTH_ESTIMATORS_STR.items():
    _estimators = {}
    for _name, (_sub_module, _cls) in _estimators_str.items():
        try:
            _estimators[_name] = getattr(
                __import__(_sub_module, globals(), locals(), [""], 1), _cls
            )
        except ImportError as e:
            logger.info(
                f"Could not import {_DISPLAY_NAMES[_backend]}-based depth estimator {_name}: {e}"
            )
            _estimators[_name] = ModuleNotLoaded(e.name)
    _ESTIMATORS_BY_BACKEND[_backend] = _estimators

DEPTH_ESTIMATOR_FACTORIES: dict[
    str, dict[str, Callable[[Device], DepthEstimator] | ModuleNotLoaded]
] = {
    "cycle_gan": {
        backend: estimators["cycle_gan"]
        for backend, estimators in _ESTIMATORS_BY_BACKEND.items()
    }
}

# The depth estimators are neural networks, so unlike the depth renderer they benefit from a GPU
DEVICE_PREFERENCE_ORDER = (Device("cuda"), Device("cpu"))
BACKEND_PREFERENCE_ORDER = ("jax", "torch")


def resolve_backend_and_device(
    estimator_type: Literal["cycle_gan"],
    backend: Literal["jax", "torch", "auto"],
    device: Device | None = None,
) -> tuple[Literal["jax", "torch"], Device]:
    if backend not in BACKEND_PREFERENCE_ORDER + ("auto",):
        raise ValueError(
            f"Backend {backend} is not in the list of supported backends: "
            f"{BACKEND_PREFERENCE_ORDER + ('auto',)}"
        )

    factories = DEPTH_ESTIMATOR_FACTORIES[estimator_type]
    device_preference_order = DEVICE_PREFERENCE_ORDER if device is None else (device,)
    backend_preference_order = (
        BACKEND_PREFERENCE_ORDER if backend == "auto" else (backend,)
    )

    viable_backends = []
    reasons = []
    for be in backend_preference_order:
        if be not in factories:
            reasons.append(f"Backend {be} does not support depth estimator {estimator_type}.")
            continue
        factory = factories[be]
        if isinstance(factory, ModuleNotLoaded):
            reasons.append(
                f"Depth estimator {estimator_type} supports backend {be}, but backend {be} could not be "
                f"loaded, because {factory.missing_module} could not be imported."
            )
            continue
        viable_backends.append(be)

    for dev in device_preference_order:
        for be in viable_backends:
            if device_available(dev, be):
                return be, dev
            reasons.append(f"Device {dev} is not available for backend {be}.")

    dev_msg = "" if device is None else f" with device {device}"
    reasons_joined = "\n".join(reasons)
    raise ValueError(
        f"Could not find a suitable backend-device combination for depth estimator "
        f"{estimator_type}{dev_msg}. Reasons:\n{reasons_joined}"
    )


def mk_depth_estimator(
    estimator_type: Literal["cycle_gan"] = "cycle_gan",
    backend: Literal["jax", "torch", "auto"] = "auto",
    device: str | None = None,
    device_index: int = 0,
) -> DepthEstimator:
    if estimator_type not in DEPTH_ESTIMATOR_FACTORIES:
        raise ValueError(f"Depth estimator {estimator_type} does not exist.")
    device = None if device is None else Device(device, device_index)
    backend, device = resolve_backend_and_device(estimator_type, backend, device)
    logger.info(
        f'Initializing depth estimator of type "{estimator_type}" for backend {backend} on device {device}.'
    )
    return DEPTH_ESTIMATOR_FACTORIES[estimator_type][backend](device)
