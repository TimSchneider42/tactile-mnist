# ABCShape

<p align="center"><img src="img/env/ABCShape-v0.webp" alt="ABCShape-v0" width="200px"/></p>

This environment is part of the tactile shape reconstruction environments.
Refer to the [tactile shape reconstruction environments overview](TactileShapeReconstructionEnv.md) for a general description of these environments, including their prediction space, prediction target, loss, and metrics.

|                              |                                        |
|------------------------------|----------------------------------------|
| **Environment ID**           | ABCShape-v0                            |
| **Dataset**                  | [ABC Dataset](datasets.md#abc-dataset) |
| **Prediction Dimensions**    | 132                                    |
| **Step limit**               | 32                                     |
| **Sensor rotation**          | enabled                                |
| **Object pose perturbation** | enabled                                |

## Description

In the ABCShape environment, the agent's objective is to reconstruct the full 3D shape of realistic industrial 3D CAD models by touch alone.
Object pose perturbation is enabled, meaning that the object shifts around slightly while being touched.
This requires the agent to use robust strategies that are invariant to small shifts in the object's pose.
The ABCShapeStatic variants disable this perturbation, so the object stays fixed in place for the entire episode, though its initial pose is still randomized.
Unlike the TactileMNISTShape environment, in ABCShape, the objects are much more complex, realistic, and diverse in shape, making it a more challenging task.
For this reason, we allow the agent to rotate the sensor, which can be crucial to effectively explore complex objects.
Note that encoding the object's orientation in the prediction is particularly important in ABCShape, as the models of the ABC dataset have no canonical orientation, so a prediction target in the model frame would not be identifiable from touch observations alone.

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
| ABCShape-DR-v0 | Same as ABCShape-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. ABCShape-DR-train-v0 is an alias for it. | <img src="img/env/ABCShape-DR-v0.webp" alt="ABCShape-DR-v0" width="200px"/> |
| ABCShape-test-v0        | Uses the test split of the _ABC Dataset_ instead of the train split.                                     | <img src="img/env/ABCShape-test-v0.webp" alt="ABCShape-test-v0" width="200px"/>             |
| ABCShape-DR-test-v0 | Same as ABCShape-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/ABCShape-DR-test-v0.webp" alt="ABCShape-DR-test-v0" width="200px"/> |
| ABCShape-CycleGAN-train-v0 | Uses a [CycleGAN](https://junyanz.github.io/CycleGAN/) trained on data points from the [Real Tactile MNIST](datasets.md#available-touch-datasets) dataset instead of Taxim to render the tactile images. | <img src="img/env/ABCShape-CycleGAN-v0.webp" alt="ABCShape-CycleGAN-v0" width="200px"/> |
| ABCShape-CycleGAN-DR-v0 | Same as ABCShape-CycleGAN-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. ABCShape-CycleGAN-DR-train-v0 is an alias for it. | <img src="img/env/ABCShape-CycleGAN-DR-v0.webp" alt="ABCShape-CycleGAN-DR-v0" width="200px"/> |
| ABCShape-CycleGAN-test-v0 | Same as ABCShape-CycleGAN-train-v0 but uses the test split of the _ABC Dataset_ instead of the train split. | <img src="img/env/ABCShape-CycleGAN-test-v0.webp" alt="ABCShape-CycleGAN-test-v0" width="200px"/> |
| ABCShape-CycleGAN-DR-test-v0 | Same as ABCShape-CycleGAN-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/ABCShape-CycleGAN-DR-test-v0.webp" alt="ABCShape-CycleGAN-DR-test-v0" width="200px"/> |
| ABCShape-Depth-train-v0 | Uses a depth image instead of rendering tactile images.                                                  | <img src="img/env/ABCShape-Depth-v0.webp" alt="ABCShape-Depth-v0" width="200px"/>           |
| ABCShape-Depth-DR-v0 | Same as ABCShape-Depth-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. ABCShape-Depth-DR-train-v0 is an alias for it. | <img src="img/env/ABCShape-Depth-DR-v0.webp" alt="ABCShape-Depth-DR-v0" width="200px"/> |
| ABCShape-Depth-test-v0  | Same as ABCShape-Depth-train-v0 but uses the test split of the _ABC Dataset_ instead of the train split. | <img src="img/env/ABCShape-Depth-test-v0.webp" alt="ABCShape-Depth-test-v0" width="200px"/> |
| ABCShape-Depth-DR-test-v0 | Same as ABCShape-Depth-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. | <img src="img/env/ABCShape-Depth-DR-test-v0.webp" alt="ABCShape-Depth-DR-test-v0" width="200px"/> |
| ABCShapeStatic-v0 | Same as ABCShape-v0 but the object pose stays fixed while it is being touched (object pose perturbation disabled). ABCShapeStatic-train-v0 is an alias for it. | <img src="img/env/ABCShapeStatic-v0.webp" alt="ABCShapeStatic-v0" width="200px"/> |
| ABCShapeStatic-train-v0 | Alias for ABCShapeStatic-v0. | <img src="img/env/ABCShapeStatic-v0.webp" alt="ABCShapeStatic-v0" width="200px"/> |
| ABCShapeStatic-DR-v0 | Same as ABCShape-DR-v0 but the object pose stays fixed while it is being touched. ABCShapeStatic-DR-train-v0 is an alias for it. | <img src="img/env/ABCShapeStatic-DR-v0.webp" alt="ABCShapeStatic-DR-v0" width="200px"/> |
| ABCShapeStatic-test-v0 | Same as ABCShape-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/ABCShapeStatic-test-v0.webp" alt="ABCShapeStatic-test-v0" width="200px"/> |
| ABCShapeStatic-DR-test-v0 | Same as ABCShape-DR-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/ABCShapeStatic-DR-test-v0.webp" alt="ABCShapeStatic-DR-test-v0" width="200px"/> |
| ABCShapeStatic-CycleGAN-train-v0 | Same as ABCShape-CycleGAN-train-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/ABCShapeStatic-CycleGAN-v0.webp" alt="ABCShapeStatic-CycleGAN-v0" width="200px"/> |
| ABCShapeStatic-CycleGAN-DR-v0 | Same as ABCShape-CycleGAN-DR-v0 but the object pose stays fixed while it is being touched. ABCShapeStatic-CycleGAN-DR-train-v0 is an alias for it. | <img src="img/env/ABCShapeStatic-CycleGAN-DR-v0.webp" alt="ABCShapeStatic-CycleGAN-DR-v0" width="200px"/> |
| ABCShapeStatic-CycleGAN-test-v0 | Same as ABCShape-CycleGAN-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/ABCShapeStatic-CycleGAN-test-v0.webp" alt="ABCShapeStatic-CycleGAN-test-v0" width="200px"/> |
| ABCShapeStatic-CycleGAN-DR-test-v0 | Same as ABCShape-CycleGAN-DR-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/ABCShapeStatic-CycleGAN-DR-test-v0.webp" alt="ABCShapeStatic-CycleGAN-DR-test-v0" width="200px"/> |
| ABCShapeStatic-Depth-train-v0 | Same as ABCShape-Depth-train-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/ABCShapeStatic-Depth-v0.webp" alt="ABCShapeStatic-Depth-v0" width="200px"/> |
| ABCShapeStatic-Depth-DR-v0 | Same as ABCShape-Depth-DR-v0 but the object pose stays fixed while it is being touched. ABCShapeStatic-Depth-DR-train-v0 is an alias for it. | <img src="img/env/ABCShapeStatic-Depth-DR-v0.webp" alt="ABCShapeStatic-Depth-DR-v0" width="200px"/> |
| ABCShapeStatic-Depth-test-v0 | Same as ABCShape-Depth-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/ABCShapeStatic-Depth-test-v0.webp" alt="ABCShapeStatic-Depth-test-v0" width="200px"/> |
| ABCShapeStatic-Depth-DR-test-v0 | Same as ABCShape-Depth-DR-test-v0 but the object pose stays fixed while it is being touched. | <img src="img/env/ABCShapeStatic-Depth-DR-test-v0.webp" alt="ABCShapeStatic-Depth-DR-test-v0" width="200px"/> |
