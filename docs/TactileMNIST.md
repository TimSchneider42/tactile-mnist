# TactileMNIST

<p align="center"><img src="img/env/TactileMNIST-v0.webp" alt="TactileMNIST-v0" width="200px"/></p>

This environment is part of the tactile classification environments.
Refer to the [tactile classification environments overview](TactileClassificationEnv.md) for a general description of these environments.

|                              |                                    |
|------------------------------|------------------------------------|
| **Environment ID**           | TactileMNIST-v0                    |
| **Dataset**                  | [MNIST 3D](datasets.md#mnist-3d) |
| **Number of classes**        | 10                                 |
| **Step limit**               | 16                                 |
| **Sensor rotation**          | disabled                           |
| **Object pose perturbation** | enabled                            |

## Description

In the TactileMNIST environment, the agent's objective is to classify 3D models of handwritten digits by touch alone.
Aside from finding the object, the main challenge in the TactileMNIST environment is to learn contour following strategies to efficiently classify it once found.
Object pose perturbation is enabled, meaning that the object shifts around slightly while being touched.
This requires the agent to use robust strategies that are invariant to small shifts in the object's pose.

## Example Usage

```python
import ap_gym

env = ap_gym.make("TactileMNIST-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("TactileMNIST-v0", num_envs=4)
```

## Version History

- `v0`: Initial release.

## Variants

| Environment ID                 | Description                                                                                                                                                                             | Preview                                                                                                  |
|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| TactileMNIST-train-v0          | Alias for TactileMNIST-v0.                                                                                                                                                              | <img src="img/env/TactileMNIST-v0.webp" alt="TactileMNIST-v0" width="200px"/>                             |
| TactileMNIST-DR-v0 | Same as TactileMNIST-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. TactileMNIST-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNIST-DR-v0.webp" alt="TactileMNIST-DR-v0" width="200px"/> |
| TactileMNIST-test-v0           | Uses the test split of _MNIST 3D_ instead of the train split.                                                                                                                           | <img src="img/env/TactileMNIST-test-v0.webp" alt="TactileMNIST-test-v0" width="200px"/>                   |
| TactileMNIST-DR-test-v0 | Same as TactileMNIST-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/TactileMNIST-DR-test-v0.webp" alt="TactileMNIST-DR-test-v0" width="200px"/> |
| TactileMNIST-CycleGAN-train-v0 | Uses a [CycleGAN](https://junyanz.github.io/CycleGAN/) trained on data points from the [Real Tactile MNIST](datasets.md#available-touch-datasets) dataset instead of Taxim to render the tactile images. | <img src="img/env/TactileMNIST-CycleGAN-v0.webp" alt="TactileMNIST-CycleGAN-v0" width="200px"/>           |
| TactileMNIST-CycleGAN-DR-v0 | Same as TactileMNIST-CycleGAN-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. TactileMNIST-CycleGAN-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNIST-CycleGAN-DR-v0.webp" alt="TactileMNIST-CycleGAN-DR-v0" width="200px"/> |
| TactileMNIST-CycleGAN-test-v0  | Same as TactileMNIST-CycleGAN-train-v0 but uses the test split of _MNIST 3D_ instead of the train split.                                                                                | <img src="img/env/TactileMNIST-CycleGAN-test-v0.webp" alt="TactileMNIST-CycleGAN-test-v0" width="200px"/> |
| TactileMNIST-CycleGAN-DR-test-v0 | Same as TactileMNIST-CycleGAN-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/TactileMNIST-CycleGAN-DR-test-v0.webp" alt="TactileMNIST-CycleGAN-DR-test-v0" width="200px"/> |
| TactileMNIST-Depth-train-v0    | Uses a depth image instead of rendering tactile images.                                                                                                                                 | <img src="img/env/TactileMNIST-Depth-v0.webp" alt="TactileMNIST-Depth-v0" width="200px"/>                 |
| TactileMNIST-Depth-DR-v0 | Same as TactileMNIST-Depth-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. TactileMNIST-Depth-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNIST-Depth-DR-v0.webp" alt="TactileMNIST-Depth-DR-v0" width="200px"/> |
| TactileMNIST-Depth-test-v0     | Same as TactileMNIST-Depth-train-v0 but uses the test split of _MNIST 3D_ instead of the train split.                                                                                   | <img src="img/env/TactileMNIST-Depth-test-v0.webp" alt="TactileMNIST-Depth-test-v0" width="200px"/>       |
| TactileMNIST-Depth-DR-test-v0 | Same as TactileMNIST-Depth-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/TactileMNIST-Depth-DR-test-v0.webp" alt="TactileMNIST-Depth-DR-test-v0" width="200px"/> |
| TactileMNISTSnap-train-v0 | Snap variant of TactileMNIST-v0: instead of positioning the sensor freely, in every step, 23 touch positions are sampled uniformly over the cell and the one closest to the requested target position is chosen. This simulates the touch selection scheme of the [TactileMNISTRealSnap](TactileMNISTRealSnap.md) environment. | <img src="img/env/TactileMNISTSnap-v0.webp" alt="TactileMNISTSnap-v0" width="200px"/> |
| TactileMNISTSnap-DR-v0 | Same as TactileMNISTSnap-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. TactileMNISTSnap-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNISTSnap-DR-v0.webp" alt="TactileMNISTSnap-DR-v0" width="200px"/> |
| TactileMNISTSnap-test-v0 | Same as TactileMNISTSnap-train-v0 but uses the test split of _MNIST 3D_ instead of the train split. | <img src="img/env/TactileMNISTSnap-test-v0.webp" alt="TactileMNISTSnap-test-v0" width="200px"/> |
| TactileMNISTSnap-DR-test-v0 | Same as TactileMNISTSnap-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/TactileMNISTSnap-DR-test-v0.webp" alt="TactileMNISTSnap-DR-test-v0" width="200px"/> |
| TactileMNISTSnap-CycleGAN-train-v0 | Same as TactileMNISTSnap-train-v0 but uses a [CycleGAN](https://junyanz.github.io/CycleGAN/) trained on data points from the [Real Tactile MNIST](datasets.md#available-touch-datasets) dataset instead of Taxim to render the tactile images. | <img src="img/env/TactileMNISTSnap-CycleGAN-v0.webp" alt="TactileMNISTSnap-CycleGAN-v0" width="200px"/> |
| TactileMNISTSnap-CycleGAN-DR-v0 | Same as TactileMNISTSnap-CycleGAN-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. TactileMNISTSnap-CycleGAN-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNISTSnap-CycleGAN-DR-v0.webp" alt="TactileMNISTSnap-CycleGAN-DR-v0" width="200px"/> |
| TactileMNISTSnap-CycleGAN-test-v0 | Same as TactileMNISTSnap-CycleGAN-train-v0 but uses the test split of _MNIST 3D_ instead of the train split. | <img src="img/env/TactileMNISTSnap-CycleGAN-test-v0.webp" alt="TactileMNISTSnap-CycleGAN-test-v0" width="200px"/> |
| TactileMNISTSnap-CycleGAN-DR-test-v0 | Same as TactileMNISTSnap-CycleGAN-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/TactileMNISTSnap-CycleGAN-DR-test-v0.webp" alt="TactileMNISTSnap-CycleGAN-DR-test-v0" width="200px"/> |
| TactileMNISTSnap-Depth-train-v0 | Same as TactileMNISTSnap-train-v0 but uses a depth image instead of rendering tactile images. | <img src="img/env/TactileMNISTSnap-Depth-v0.webp" alt="TactileMNISTSnap-Depth-v0" width="200px"/> |
| TactileMNISTSnap-Depth-DR-v0 | Same as TactileMNISTSnap-Depth-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. TactileMNISTSnap-Depth-DR-train-v0 is an alias for it. | <img src="img/env/TactileMNISTSnap-Depth-DR-v0.webp" alt="TactileMNISTSnap-Depth-DR-v0" width="200px"/> |
| TactileMNISTSnap-Depth-test-v0 | Same as TactileMNISTSnap-Depth-train-v0 but uses the test split of _MNIST 3D_ instead of the train split. | <img src="img/env/TactileMNISTSnap-Depth-test-v0.webp" alt="TactileMNISTSnap-Depth-test-v0" width="200px"/> |
| TactileMNISTSnap-Depth-DR-test-v0 | Same as TactileMNISTSnap-Depth-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/TactileMNISTSnap-Depth-DR-test-v0.webp" alt="TactileMNISTSnap-Depth-DR-test-v0" width="200px"/> |
