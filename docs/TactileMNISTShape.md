# TactileMNISTShape

<p align="center"><img src="img/env/TactileMNISTShape-v0.webp" alt="TactileMNISTShape-v0" width="200px"/></p>

This environment is part of the tactile regression environments.
Refer to the [tactile regression environments overview](TactileRegressionEnv.md) for a general description of these environments.

|                              |                                  |
|------------------------------|----------------------------------|
| **Environment ID**           | TactileMNISTShape-v0             |
| **Dataset**                  | [MNIST 3D](datasets.md#mnist-3d) |
| **Prediction Dimensions**    | 128                              |
| **Step limit**               | 16                               |
| **Sensor rotation**          | disabled                         |
| **Object pose perturbation** | enabled                          |

## Description

In the TactileMNISTShape environment, the agent's objective is to reconstruct the full 3D shape of 3D models of handwritten digits by touch alone.
The shape is represented by a compact latent embedding of a COD-VAE shape autoencoder, which the agent has to regress to.
Since every touch reveals only a small patch of the object's surface, the agent has to integrate information from many touches to reconstruct the global shape.
Object pose perturbation is enabled, meaning that the object shifts around slightly while being touched.
This requires the agent to use robust strategies that are invariant to small shifts in the object's pose.

## Prediction Target Space

The prediction target is the flattened [COD-VAE](https://github.com/TimSchneider42/cod-vae) latent representation of the object's mesh (Cho et al., ICCV 2025).
The mesh, posed in the platform frame, is normalized into the COD-VAE model's $[-1, 1]$ cube and encoded into $k$ latent vectors of dimension $d$.
Flattened, this yields a $k \cdot d = 128$-element `np.ndarray` for the default model ($k = 4$, $d = 32$).
Since the cube normalization removes the object's position and scale, the target jointly encodes the object's shape and its current (randomized and perturbed) orientation on the platform.
Unlike a factored orientation-plus-shape representation, this joint representation remains well-defined even for (rotationally) symmetric objects, for which the orientation alone would be ambiguous.
The latent is a deterministic function of the posed mesh: encoding uses the deterministic posterior mean, the surface sampling is seeded, and the latent tokens are put into a canonical order.
A mesh can be reconstructed from a predicted latent by decoding it into an occupancy field with the same COD-VAE model (see `cod_vae.CODVAEBase.decode_mesh`).
If the `renderer_show_shadow_objects` option is enabled, the environment decodes the agent's current prediction every step and renders it as a translucent shadow object; this is disabled by default, as it requires a COD-VAE decoder pass per step.

COD-VAE latents are KL-regularized towards a standard normal, so they are approximately unit scale and are regressed without further normalization.

The COD-VAE model can be changed via the `model` argument, which accepts a Hugging Face Hub repository id or a local npz checkpoint:

```python
import ap_gym

env = ap_gym.make("TactileMNISTShape-v0", model="TimSchneider42/cod-vae-4x32")
```

## Metrics

On top of the standard regression metrics, this environment logs the following metrics in the `info` dictionary:

- `latent_rmse`: the RMS error between the predicted and the target latent.
- `rel_error`: the Euclidean norm of the latent error relative to the norm of the target latent.

## Example Usage

```python
import ap_gym

env = ap_gym.make("TactileMNISTShape-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("TactileMNISTShape-v0", num_envs=4)
```

## Version History

- `v0`: Initial release.

## Variants

| Environment ID                      | Description                                                                                                                                                                            | Preview                                                                                                            |
|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| TactileMNISTShape-train-v0          | Alias for TactileMNISTShape-v0.                                                                                                                                                        | <img src="img/env/TactileMNISTShape-v0.webp" alt="TactileMNISTShape-v0" width="200px"/>                             |
| TactileMNISTShape-DR-v0 | Same as TactileMNISTShape-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. TactileMNISTShape-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNISTShape-DR-v0.webp" alt="TactileMNISTShape-DR-v0" width="200px"/> |
| TactileMNISTShape-test-v0           | Uses the test split of _MNIST 3D_ instead of the train split.                                                                                                                          | <img src="img/env/TactileMNISTShape-test-v0.webp" alt="TactileMNISTShape-test-v0" width="200px"/>                   |
| TactileMNISTShape-DR-test-v0 | Same as TactileMNISTShape-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/TactileMNISTShape-DR-test-v0.webp" alt="TactileMNISTShape-DR-test-v0" width="200px"/> |
| TactileMNISTShape-CycleGAN-train-v0 | Uses a [CycleGAN](https://junyanz.github.io/CycleGAN/) trained on data points from the [Real Tactile MNIST](datasets.md#available-touch-datasets) dataset instead of Taxim to render the tactile images. | <img src="img/env/TactileMNISTShape-CycleGAN-v0.webp" alt="TactileMNISTShape-CycleGAN-v0" width="200px"/>           |
| TactileMNISTShape-CycleGAN-DR-v0 | Same as TactileMNISTShape-CycleGAN-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. TactileMNISTShape-CycleGAN-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNISTShape-CycleGAN-DR-v0.webp" alt="TactileMNISTShape-CycleGAN-DR-v0" width="200px"/> |
| TactileMNISTShape-CycleGAN-test-v0  | Same as TactileMNISTShape-CycleGAN-train-v0 but uses the test split of _MNIST 3D_ instead of the train split.                                                                          | <img src="img/env/TactileMNISTShape-CycleGAN-test-v0.webp" alt="TactileMNISTShape-CycleGAN-test-v0" width="200px"/> |
| TactileMNISTShape-CycleGAN-DR-test-v0 | Same as TactileMNISTShape-CycleGAN-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/TactileMNISTShape-CycleGAN-DR-test-v0.webp" alt="TactileMNISTShape-CycleGAN-DR-test-v0" width="200px"/> |
| TactileMNISTShape-Depth-train-v0    | Uses a depth image instead of rendering tactile images.                                                                                                                                | <img src="img/env/TactileMNISTShape-Depth-v0.webp" alt="TactileMNISTShape-Depth-v0" width="200px"/>                 |
| TactileMNISTShape-Depth-DR-v0 | Same as TactileMNISTShape-Depth-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. TactileMNISTShape-Depth-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNISTShape-Depth-DR-v0.webp" alt="TactileMNISTShape-Depth-DR-v0" width="200px"/> |
| TactileMNISTShape-Depth-test-v0     | Same as TactileMNISTShape-Depth-train-v0 but uses the test split of _MNIST 3D_ instead of the train split.                                                                             | <img src="img/env/TactileMNISTShape-Depth-test-v0.webp" alt="TactileMNISTShape-Depth-test-v0" width="200px"/>       |
| TactileMNISTShape-Depth-DR-test-v0 | Same as TactileMNISTShape-Depth-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/TactileMNISTShape-Depth-DR-test-v0.webp" alt="TactileMNISTShape-Depth-DR-test-v0" width="200px"/> |
| TactileMNISTShapeSnap-train-v0 | Snap variant of TactileMNISTShape-v0: instead of positioning the sensor freely, in every step, 23 touch positions are sampled uniformly over the cell and the one closest to the requested target position is chosen. This simulates the touch selection scheme of the [TactileMNISTRealSnap](TactileMNISTRealSnap.md) environment. | <img src="img/env/TactileMNISTShapeSnap-v0.webp" alt="TactileMNISTShapeSnap-v0" width="200px"/> |
| TactileMNISTShapeSnap-DR-v0 | Same as TactileMNISTShapeSnap-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. TactileMNISTShapeSnap-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNISTShapeSnap-DR-v0.webp" alt="TactileMNISTShapeSnap-DR-v0" width="200px"/> |
| TactileMNISTShapeSnap-test-v0 | Same as TactileMNISTShapeSnap-train-v0 but uses the test split of _MNIST 3D_ instead of the train split. | <img src="img/env/TactileMNISTShapeSnap-test-v0.webp" alt="TactileMNISTShapeSnap-test-v0" width="200px"/> |
| TactileMNISTShapeSnap-DR-test-v0 | Same as TactileMNISTShapeSnap-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/TactileMNISTShapeSnap-DR-test-v0.webp" alt="TactileMNISTShapeSnap-DR-test-v0" width="200px"/> |
| TactileMNISTShapeSnap-CycleGAN-train-v0 | Same as TactileMNISTShapeSnap-train-v0 but uses a [CycleGAN](https://junyanz.github.io/CycleGAN/) trained on data points from the [Real Tactile MNIST](datasets.md#available-touch-datasets) dataset instead of Taxim to render the tactile images. | <img src="img/env/TactileMNISTShapeSnap-CycleGAN-v0.webp" alt="TactileMNISTShapeSnap-CycleGAN-v0" width="200px"/> |
| TactileMNISTShapeSnap-CycleGAN-DR-v0 | Same as TactileMNISTShapeSnap-CycleGAN-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. TactileMNISTShapeSnap-CycleGAN-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNISTShapeSnap-CycleGAN-DR-v0.webp" alt="TactileMNISTShapeSnap-CycleGAN-DR-v0" width="200px"/> |
| TactileMNISTShapeSnap-CycleGAN-test-v0 | Same as TactileMNISTShapeSnap-CycleGAN-train-v0 but uses the test split of _MNIST 3D_ instead of the train split. | <img src="img/env/TactileMNISTShapeSnap-CycleGAN-test-v0.webp" alt="TactileMNISTShapeSnap-CycleGAN-test-v0" width="200px"/> |
| TactileMNISTShapeSnap-CycleGAN-DR-test-v0 | Same as TactileMNISTShapeSnap-CycleGAN-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/TactileMNISTShapeSnap-CycleGAN-DR-test-v0.webp" alt="TactileMNISTShapeSnap-CycleGAN-DR-test-v0" width="200px"/> |
| TactileMNISTShapeSnap-Depth-train-v0 | Same as TactileMNISTShapeSnap-train-v0 but uses a depth image instead of rendering tactile images. | <img src="img/env/TactileMNISTShapeSnap-Depth-v0.webp" alt="TactileMNISTShapeSnap-Depth-v0" width="200px"/> |
| TactileMNISTShapeSnap-Depth-DR-v0 | Same as TactileMNISTShapeSnap-Depth-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. TactileMNISTShapeSnap-Depth-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNISTShapeSnap-Depth-DR-v0.webp" alt="TactileMNISTShapeSnap-Depth-DR-v0" width="200px"/> |
| TactileMNISTShapeSnap-Depth-test-v0 | Same as TactileMNISTShapeSnap-Depth-train-v0 but uses the test split of _MNIST 3D_ instead of the train split. | <img src="img/env/TactileMNISTShapeSnap-Depth-test-v0.webp" alt="TactileMNISTShapeSnap-Depth-test-v0" width="200px"/> |
| TactileMNISTShapeSnap-Depth-DR-test-v0 | Same as TactileMNISTShapeSnap-Depth-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/TactileMNISTShapeSnap-Depth-DR-test-v0.webp" alt="TactileMNISTShapeSnap-Depth-DR-test-v0" width="200px"/> |
