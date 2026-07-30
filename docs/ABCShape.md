# ABCShape

<p align="center"><img src="img/env/ABCShape-v0.webp" alt="ABCShape-v0" width="200px"/></p>

This environment is part of the tactile regression environments.
Refer to the [tactile regression environments overview](TactileRegressionEnv.md) for a general description of these environments.

|                              |                                        |
|------------------------------|----------------------------------------|
| **Environment ID**           | ABCShape-v0                            |
| **Dataset**                  | [ABC Dataset](datasets.md#abc-dataset) |
| **Prediction Dimensions**    | 128                                    |
| **Step limit**               | 32                                     |
| **Sensor rotation**          | enabled                                |
| **Object pose perturbation** | enabled                                |

## Description

In the ABCShape environment, the agent's objective is to reconstruct the full 3D shape of realistic industrial 3D CAD models by touch alone.
The shape is represented by a compact latent embedding of a COD-VAE shape autoencoder, which the agent has to regress to.
Since every touch reveals only a small patch of the object's surface, the agent has to integrate information from many touches to reconstruct the global shape.
Object pose perturbation is enabled, meaning that the object shifts around slightly while being touched.
This requires the agent to use robust strategies that are invariant to small shifts in the object's pose.
Unlike the TactileMNISTShape environment, in ABCShape, the objects are much more complex, realistic, and diverse in shape, making it a more challenging task.
For this reason, we allow the agent to rotate the sensor, which can be crucial to effectively explore complex objects.

## Prediction Target Space

The prediction target is the flattened [COD-VAE](https://github.com/TimSchneider42/cod-vae) latent representation of the object's mesh (Cho et al., ICCV 2025).
The mesh, posed in the platform frame, is normalized into the COD-VAE model's $[-1, 1]$ cube and encoded into $k$ latent vectors of dimension $d$.
Flattened, this yields a $k \cdot d = 128$-element `np.ndarray` for the default model ($k = 4$, $d = 32$).
Since the cube normalization removes the object's position and scale, the target jointly encodes the object's shape and its current (randomized and perturbed) orientation on the platform.
Note that encoding the orientation is important in ABCShape, as the models of the ABC dataset have no canonical orientation, so a prediction target in the model frame would not be identifiable from touch observations alone.
Unlike a factored orientation-plus-shape representation, this joint representation remains well-defined even for (rotationally) symmetric objects, for which the orientation alone would be ambiguous.
The latent is a deterministic function of the posed mesh: encoding uses the deterministic posterior mean, the surface sampling is seeded, and the latent tokens are put into a canonical order.
A mesh can be reconstructed from a predicted latent by decoding it into an occupancy field with the same COD-VAE model (see `cod_vae.CODVAEBase.decode_mesh`).
If the `renderer_show_shadow_objects` option is enabled, the environment decodes the agent's current prediction every step and renders it as a translucent shadow object; this is disabled by default, as it requires a COD-VAE decoder pass per step.

COD-VAE latents are KL-regularized towards a standard normal, so they are approximately unit scale and are regressed without further normalization.

The COD-VAE model can be changed via the `model` argument, which accepts a Hugging Face Hub repository id or a local npz checkpoint:

```python
import ap_gym

env = ap_gym.make("ABCShape-v0", model="TimSchneider42/cod-vae-4x32")
```

## Metrics

On top of the standard regression metrics, this environment logs the following metrics in the `info` dictionary:

- `latent_rmse`: the RMS error between the predicted and the target latent.
- `rel_error`: the Euclidean norm of the latent error relative to the norm of the target latent.

## Example Usage

```python
import ap_gym

env = ap_gym.make("ABCShape-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("ABCShape-v0", num_envs=4)
```

## Version History

- `v0`: Initial release.

## Variants

| Environment ID          | Description                                                                                              | Preview                                                                                    |
|-------------------------|----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| ABCShape-train-v0       | Alias for ABCShape-v0.                                                                                   | <img src="img/env/ABCShape-v0.webp" alt="ABCShape-v0" width="200px"/>                       |
| ABCShape-test-v0        | Uses the test split of the _ABC Dataset_ instead of the train split.                                     | <img src="img/env/ABCShape-test-v0.webp" alt="ABCShape-test-v0" width="200px"/>             |
| ABCShape-Depth-train-v0 | Uses a depth image instead of rendering tactile images.                                                  | <img src="img/env/ABCShape-Depth-v0.webp" alt="ABCShape-Depth-v0" width="200px"/>           |
| ABCShape-Depth-test-v0  | Same as ABCShape-Depth-train-v0 but uses the test split of the _ABC Dataset_ instead of the train split. | <img src="img/env/ABCShape-Depth-test-v0.webp" alt="ABCShape-Depth-test-v0" width="200px"/> |
