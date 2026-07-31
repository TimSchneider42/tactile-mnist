# JAX re-implementation of the CycleGAN ResNet generator. The architecture follows the CycleGAN project
# (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), whose license is reproduced in cycle_gan_torch.py.
#
# The parameters are the ones the PyTorch generator was trained with, so this implementation has to reproduce the
# PyTorch layer semantics exactly: instance normalization without affine parameters, reflection padding, and
# transposed convolutions with output padding.

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Layer indices in the nn.Sequential of the resnet_9blocks generator with ngf=64
_STEM_CONV = 1
_DOWN_CONVS = (4, 7)
_RESNET_BLOCKS = tuple(range(10, 19))
_UP_CONVS = (19, 22)
_OUT_CONV = 26

Params = dict[str, jax.Array]


def load_generator_params_jax(path: Path | str) -> Params:
    """Load the parameters of a resnet_9blocks generator from one of the .npz checkpoints in resources."""
    with np.load(path) as data:
        params = {k: jnp.asarray(data[k]) for k in data.files}
    missing = [
        f"model.{i}.{s}"
        for i in (_STEM_CONV, *_DOWN_CONVS, *_UP_CONVS, _OUT_CONV)
        for s in ("weight", "bias")
    ] + [
        f"model.{i}.conv_block.{j}.{s}"
        for i in _RESNET_BLOCKS
        for j in (1, 5)
        for s in ("weight", "bias")
    ]
    absent = [k for k in missing if k not in params]
    if absent:
        raise ValueError(
            f"Checkpoint {path} is not a resnet_9blocks generator: {len(absent)} parameters are missing "
            f"(e.g. {absent[0]})."
        )
    return params


def _instance_norm(x: jax.Array, eps: float = 1e-5) -> jax.Array:
    # nn.InstanceNorm2d(affine=False, track_running_stats=False): normalize each (sample, channel) over H and W
    mean = jnp.mean(x, axis=(-2, -1), keepdims=True)
    var = jnp.var(x, axis=(-2, -1), keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps)


def _reflection_pad(x: jax.Array, padding: int) -> jax.Array:
    return jnp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), "reflect")


def _conv(
    x: jax.Array, weight: jax.Array, bias: jax.Array, stride: int = 1, padding: int = 0
) -> jax.Array:
    y = jax.lax.conv_general_dilated(
        x,
        weight,
        (stride, stride),
        ((padding, padding), (padding, padding)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return y + bias[None, :, None, None]


def _conv_transpose(
    x: jax.Array,
    weight: jax.Array,
    bias: jax.Array,
    stride: int = 2,
    padding: int = 1,
    output_padding: int = 1,
) -> jax.Array:
    """nn.ConvTranspose2d, expressed as a dilated convolution with the spatially flipped kernel.

    A transposed convolution is the gradient of a convolution, which is equivalent to dilating the input by the
    stride, padding it by (kernel_size - 1 - padding) on each side plus output_padding at the end, and convolving
    with the flipped kernel. This yields the same output size as PyTorch: (in - 1) * stride - 2 * padding +
    kernel_size + output_padding.
    """
    kernel_size = weight.shape[-1]
    assert weight.shape[-2] == kernel_size, "only square kernels are supported"
    # PyTorch stores transposed convolution weights as (in_channels, out_channels, kh, kw)
    weight_eff = jnp.flip(jnp.swapaxes(weight, 0, 1), axis=(-2, -1))
    pad_low = kernel_size - 1 - padding
    pad = ((pad_low, pad_low + output_padding),) * 2
    y = jax.lax.conv_general_dilated(
        x,
        weight_eff,
        (1, 1),
        pad,
        lhs_dilation=(stride, stride),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return y + bias[None, :, None, None]


def _resnet_block(x: jax.Array, params: Params, prefix: str) -> jax.Array:
    y = _reflection_pad(x, 1)
    y = _conv(y, params[f"{prefix}.1.weight"], params[f"{prefix}.1.bias"])
    y = jax.nn.relu(_instance_norm(y))
    y = _reflection_pad(y, 1)
    y = _conv(y, params[f"{prefix}.5.weight"], params[f"{prefix}.5.bias"])
    return x + _instance_norm(y)


def apply_generator_jax(params: Params, x: jax.Array) -> jax.Array:
    """Run the resnet_9blocks generator on NCHW input in [-1, 1] and return NCHW output in [-1, 1]."""
    y = _reflection_pad(x, 3)
    y = _conv(y, params[f"model.{_STEM_CONV}.weight"], params[f"model.{_STEM_CONV}.bias"])
    y = jax.nn.relu(_instance_norm(y))

    for i in _DOWN_CONVS:
        y = _conv(
            y, params[f"model.{i}.weight"], params[f"model.{i}.bias"], stride=2, padding=1
        )
        y = jax.nn.relu(_instance_norm(y))

    for i in _RESNET_BLOCKS:
        y = _resnet_block(y, params, f"model.{i}.conv_block")

    for i in _UP_CONVS:
        y = _conv_transpose(y, params[f"model.{i}.weight"], params[f"model.{i}.bias"])
        y = jax.nn.relu(_instance_norm(y))

    y = _reflection_pad(y, 3)
    y = _conv(y, params[f"model.{_OUT_CONV}.weight"], params[f"model.{_OUT_CONV}.bias"])
    return jnp.tanh(y)
