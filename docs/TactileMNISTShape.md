# TactileMNISTShape

<p align="center"><img src="img/env/TactileMNISTShape-v0.webp" alt="TactileMNISTShape-v0" width="200px"/></p>

This environment is part of the tactile shape reconstruction environments.
Refer to the [tactile shape reconstruction environments overview](TactileShapeReconstructionEnv.md) for a general description of these environments, including their prediction space, prediction target, loss, and metrics.

|                              |                                  |
|------------------------------|----------------------------------|
| **Environment ID**           | TactileMNISTShape-v0             |
| **Dataset**                  | [MNIST 3D](datasets.md#mnist-3d) |
| **Prediction Dimensions**    | 132                              |
| **Step limit**               | 16                               |
| **Sensor rotation**          | disabled                         |
| **Object pose perturbation** | enabled                          |

## Description

In the TactileMNISTShape environment, the agent's objective is to reconstruct the full 3D shape of 3D models of handwritten digits by touch alone.
Object pose perturbation is enabled, meaning that the object shifts around slightly while being touched.
This requires the agent to use robust strategies that are invariant to small shifts in the object's pose.
The TactileMNISTShapeStatic variants disable this perturbation, so the object stays fixed in place for the entire episode, though its initial pose is still randomized.

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
| TactileMNISTShapeStatic-v0 | Same as TactileMNISTShape-v0 but the object pose stays fixed while it is being touched (object pose perturbation disabled). TactileMNISTShapeStatic-train-v0 is an alias for it. | <img src="img/env/TactileMNISTShapeStatic-v0.webp" alt="TactileMNISTShapeStatic-v0" width="200px"/> |
| TactileMNISTShapeStatic-train-v0 | Alias for TactileMNISTShapeStatic-v0. | <img src="img/env/TactileMNISTShapeStatic-v0.webp" alt="TactileMNISTShapeStatic-v0" width="200px"/> |
| TactileMNISTShapeStatic-DR-v0 | Same as TactileMNISTShape-DR-v0 but the object pose stays fixed while it is being touched. TactileMNISTShapeStatic-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNISTShapeStatic-DR-v0.webp" alt="TactileMNISTShapeStatic-DR-v0" width="200px"/> |
| TactileMNISTShapeStatic-test-v0 | Same as TactileMNISTShape-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStatic-test-v0.webp" alt="TactileMNISTShapeStatic-test-v0" width="200px"/> |
| TactileMNISTShapeStatic-DR-test-v0 | Same as TactileMNISTShape-DR-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStatic-DR-test-v0.webp" alt="TactileMNISTShapeStatic-DR-test-v0" width="200px"/> |
| TactileMNISTShapeStatic-CycleGAN-train-v0 | Same as TactileMNISTShape-CycleGAN-train-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStatic-CycleGAN-v0.webp" alt="TactileMNISTShapeStatic-CycleGAN-v0" width="200px"/> |
| TactileMNISTShapeStatic-CycleGAN-DR-v0 | Same as TactileMNISTShape-CycleGAN-DR-v0 but the object pose stays fixed while it is being touched. TactileMNISTShapeStatic-CycleGAN-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNISTShapeStatic-CycleGAN-DR-v0.webp" alt="TactileMNISTShapeStatic-CycleGAN-DR-v0" width="200px"/> |
| TactileMNISTShapeStatic-CycleGAN-test-v0 | Same as TactileMNISTShape-CycleGAN-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStatic-CycleGAN-test-v0.webp" alt="TactileMNISTShapeStatic-CycleGAN-test-v0" width="200px"/> |
| TactileMNISTShapeStatic-CycleGAN-DR-test-v0 | Same as TactileMNISTShape-CycleGAN-DR-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStatic-CycleGAN-DR-test-v0.webp" alt="TactileMNISTShapeStatic-CycleGAN-DR-test-v0" width="200px"/> |
| TactileMNISTShapeStatic-Depth-train-v0 | Same as TactileMNISTShape-Depth-train-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStatic-Depth-v0.webp" alt="TactileMNISTShapeStatic-Depth-v0" width="200px"/> |
| TactileMNISTShapeStatic-Depth-DR-v0 | Same as TactileMNISTShape-Depth-DR-v0 but the object pose stays fixed while it is being touched. TactileMNISTShapeStatic-Depth-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNISTShapeStatic-Depth-DR-v0.webp" alt="TactileMNISTShapeStatic-Depth-DR-v0" width="200px"/> |
| TactileMNISTShapeStatic-Depth-test-v0 | Same as TactileMNISTShape-Depth-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStatic-Depth-test-v0.webp" alt="TactileMNISTShapeStatic-Depth-test-v0" width="200px"/> |
| TactileMNISTShapeStatic-Depth-DR-test-v0 | Same as TactileMNISTShape-Depth-DR-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStatic-Depth-DR-test-v0.webp" alt="TactileMNISTShapeStatic-Depth-DR-test-v0" width="200px"/> |
| TactileMNISTShapeStaticSnap-train-v0 | Same as TactileMNISTShapeSnap-train-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStaticSnap-v0.webp" alt="TactileMNISTShapeStaticSnap-v0" width="200px"/> |
| TactileMNISTShapeStaticSnap-DR-v0 | Same as TactileMNISTShapeSnap-DR-v0 but the object pose stays fixed while it is being touched. TactileMNISTShapeStaticSnap-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNISTShapeStaticSnap-DR-v0.webp" alt="TactileMNISTShapeStaticSnap-DR-v0" width="200px"/> |
| TactileMNISTShapeStaticSnap-test-v0 | Same as TactileMNISTShapeSnap-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStaticSnap-test-v0.webp" alt="TactileMNISTShapeStaticSnap-test-v0" width="200px"/> |
| TactileMNISTShapeStaticSnap-DR-test-v0 | Same as TactileMNISTShapeSnap-DR-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStaticSnap-DR-test-v0.webp" alt="TactileMNISTShapeStaticSnap-DR-test-v0" width="200px"/> |
| TactileMNISTShapeStaticSnap-CycleGAN-train-v0 | Same as TactileMNISTShapeSnap-CycleGAN-train-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStaticSnap-CycleGAN-v0.webp" alt="TactileMNISTShapeStaticSnap-CycleGAN-v0" width="200px"/> |
| TactileMNISTShapeStaticSnap-CycleGAN-DR-v0 | Same as TactileMNISTShapeSnap-CycleGAN-DR-v0 but the object pose stays fixed while it is being touched. TactileMNISTShapeStaticSnap-CycleGAN-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNISTShapeStaticSnap-CycleGAN-DR-v0.webp" alt="TactileMNISTShapeStaticSnap-CycleGAN-DR-v0" width="200px"/> |
| TactileMNISTShapeStaticSnap-CycleGAN-test-v0 | Same as TactileMNISTShapeSnap-CycleGAN-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStaticSnap-CycleGAN-test-v0.webp" alt="TactileMNISTShapeStaticSnap-CycleGAN-test-v0" width="200px"/> |
| TactileMNISTShapeStaticSnap-CycleGAN-DR-test-v0 | Same as TactileMNISTShapeSnap-CycleGAN-DR-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStaticSnap-CycleGAN-DR-test-v0.webp" alt="TactileMNISTShapeStaticSnap-CycleGAN-DR-test-v0" width="200px"/> |
| TactileMNISTShapeStaticSnap-Depth-train-v0 | Same as TactileMNISTShapeSnap-Depth-train-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStaticSnap-Depth-v0.webp" alt="TactileMNISTShapeStaticSnap-Depth-v0" width="200px"/> |
| TactileMNISTShapeStaticSnap-Depth-DR-v0 | Same as TactileMNISTShapeSnap-Depth-DR-v0 but the object pose stays fixed while it is being touched. TactileMNISTShapeStaticSnap-Depth-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNISTShapeStaticSnap-Depth-DR-v0.webp" alt="TactileMNISTShapeStaticSnap-Depth-DR-v0" width="200px"/> |
| TactileMNISTShapeStaticSnap-Depth-test-v0 | Same as TactileMNISTShapeSnap-Depth-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStaticSnap-Depth-test-v0.webp" alt="TactileMNISTShapeStaticSnap-Depth-test-v0" width="200px"/> |
| TactileMNISTShapeStaticSnap-Depth-DR-test-v0 | Same as TactileMNISTShapeSnap-Depth-DR-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/TactileMNISTShapeStaticSnap-Depth-DR-test-v0.webp" alt="TactileMNISTShapeStaticSnap-Depth-DR-test-v0" width="200px"/> |
