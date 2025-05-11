import itertools as _itertools
import logging as logging
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Literal, Callable

from .depth_renderer_numpy import DepthRendererNumpy
from .tactile_renderer import TactileRenderer, Device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModuleNotLoaded:
    missing_module: str


_TACTILE_RENDERERS_STR = {
    "jax": {
        "depth": ("depth_renderer_jax", "DepthRendererJAX"),
        "taxim": ("taxim_renderer_jax", "TaximRendererJAX"),
    },
    "torch": {
        "depth": ("depth_renderer_torch", "DepthRendererTorch"),
        "taxim": ("taxim_renderer_torch", "TaximRendererTorch"),
        "cycle_gan": ("cycle_gan_renderer_torch", "CycleGANRendererTorch"),
    },
}

_DISPLAY_NAMES = {
    "jax": "JAX",
    "torch": "PyTorch",
}


def device_available_jax(device: Device, module) -> bool:
    return len(module.devices(device.platform)) > device.device_index


def device_available_torch(device: Device, module) -> bool:
    try:
        torch_device = module.device(f"{device.platform}:{device.device_index}")
        with module.no_grad():
            module.empty((), device=torch_device)
        return True
    except:
        return False


_DEVICE_AVAILABLE_FNS_RAW = {
    "jax": device_available_jax,
    "torch": device_available_torch,
}

_DEVICE_AVAILABLE_FNS = {
    "numpy": lambda device: device.platform == "cpu",
}
_RENDERERS_BY_BACKEND = {"numpy": {"depth": DepthRendererNumpy}}

for name, renderers_str in _TACTILE_RENDERERS_STR.items():
    try:
        module = __import__(name, globals(), locals(), [name])

        _DEVICE_AVAILABLE_FNS[name] = partial(
            _DEVICE_AVAILABLE_FNS_RAW[name], module=module
        )

        renderers_dict = {}

        for renderer_name, (sub_module, renderer) in renderers_str.items():
            try:
                renderers_dict[renderer_name] = getattr(
                    __import__(sub_module, globals(), locals(), [""], 1), renderer
                )
            except ImportError as e:
                logger.info(
                    f"Could not import {_DISPLAY_NAMES[name]}-based tactile renderer {renderer_name}: {e}"
                )
                renderers_dict[renderer_name] = ModuleNotLoaded(e.name)

        _RENDERERS_BY_BACKEND[name] = renderers_dict

    except ImportError as e:
        logger.info(f"Could not import {_DISPLAY_NAMES[name]}: {e}")

        _DEVICE_AVAILABLE_FNS[name] = lambda device: False


def device_available(device: Device, backend: Literal["jax", "torch", "numpy"]) -> bool:
    if backend not in _DEVICE_AVAILABLE_FNS:
        raise ValueError(
            f"Backend {backend} is not in the list of supported backends: {_DEVICE_AVAILABLE_FNS.keys()}"
        )
    return _DEVICE_AVAILABLE_FNS[backend](device)


TACTILE_RENDERER_FACTORIES: dict[
    str, dict[str, Callable[[Device], TactileRenderer] | ModuleNotLoaded]
] = {
    renderer_name: {
        backend_name: renderers[renderer_name]
        for backend_name, renderers in _RENDERERS_BY_BACKEND.items()
        if renderer_name in renderers
    }
    for renderer_name in set(
        _itertools.chain(*(r.keys() for r in _RENDERERS_BY_BACKEND.values()))
    )
}

DEFAULT_DEVICE_PREFERENCE_ORDER = (Device("cuda"), Device("cpu"))
BACKEND_PREFERENCE_ORDER = ("numpy", "jax", "torch")

DEVICE_PREFERENCE_ORDER = defaultdict(lambda: DEFAULT_DEVICE_PREFERENCE_ORDER)
DEVICE_PREFERENCE_ORDER["depth"] = (Device("cpu"), Device("cuda"))


def resolve_backend_and_device(
    renderer_type: Literal["depth", "taxim", "cycle_gan"],
    backend: Literal["jax", "torch", "numpy", "auto"],
    device: Device | None = None,
) -> tuple[Literal["jax", "torch", "numpy"], Device]:
    if backend not in BACKEND_PREFERENCE_ORDER + ("auto",):
        raise ValueError(
            f"Backend {backend} is not in the list of supported backends: {BACKEND_PREFERENCE_ORDER + ('auto',)}"
        )

    factories = TACTILE_RENDERER_FACTORIES[renderer_type]

    device_preference_order = (
        DEVICE_PREFERENCE_ORDER[renderer_type] if device is None else (device,)
    )

    backend_preference_order = (
        BACKEND_PREFERENCE_ORDER if backend == "auto" else (backend,)
    )

    viable_backends = []

    reasons = []
    for backend in backend_preference_order:
        if backend not in factories:
            reasons.append(
                f"Backend {backend} does not support tactile renderer {renderer_type}."
            )
            continue

        factory = factories[backend]

        if isinstance(factory, ModuleNotLoaded):
            reasons.append(
                f"Tactile renderer {renderer_type} supports backend {backend}, but backend {backend} could not "
                f"be loaded, because {factory.missing_module} could not be imported."
            )
            continue
        viable_backends.append(backend)

    for dev in device_preference_order:
        for be in viable_backends:
            if device_available(dev, be):
                return be, dev
            else:
                reasons.append(f"Device {dev} is not available for backend {be}.")

    if device is not None:
        dev_msg = f" with device {device}"
    else:
        dev_msg = ""

    reasons_joined = "\n".join(reasons)
    raise ValueError(
        f"Could not find a suitable backend-device combination for tactile renderer {renderer_type}{dev_msg}. "
        f"Reasons:\n"
        f"{reasons_joined}"
    )


def mk_tactile_renderer(
    renderer_type: Literal["depth", "taxim", "cycle_gan"],
    backend: Literal["jax", "torch", "numpy", "auto"],
    device: str | None = None,
    device_index: int = 0,
):
    if device is None:
        device = None
    else:
        device = Device(device, device_index)

    if renderer_type not in TACTILE_RENDERER_FACTORIES:
        raise ValueError(f"Tactile renderer {renderer_type} does not exist.")

    backend, device = resolve_backend_and_device(renderer_type, backend, device)
    logger.info(
        f'Initializing tactile renderer of type "{renderer_type}" for backend {backend} on device {device}.'
    )
    return TACTILE_RENDERER_FACTORIES[renderer_type][backend](device)
