# ABCVolume

<p align="center"><img src="img/env/ABCVolume-v0.gif" alt="ABCVolume-v0" width="200px"/></p>

This environment is part of the tactile regression environments.
Refer to the [tactile regression environments overview](TactileRegressionEnv.md) for a general description of these environments.

|                              |                                        |
|------------------------------|----------------------------------------|
| **Environment ID**           | ABCVolume-v0                           |
| **Dataset**                  | [ABC Dataset](datasets.md#abc-dataset) |
| **Prediction Dimensions**    | 1                                      |
| **Step limit**               | 32                                     |
| **Sensor rotation**          | enabled                                |
| **Object pose perturbation** | enabled                                |

## Description

In the ABCVolume environment, the agent's objective is to estimate the volume of realistic industrial 3D CAD models.
Aside from finding the object, the main challenge in the ABCVolume environment is to learn contour following strategies to efficiently explore it once found.
Object pose perturbation is enabled, meaning that the object shifts around slightly while being touched.
This requires the agent to use robust strategies that are invariant to small shifts in the object's pose.
Unlike the TactileMNISTVolume environment, in ABCVolume, the objects are much more complex, realistic, and diverse in shape, making it a more challenging task.
For this reason, we allow the agent to rotate the sensor, which can be crucial to effectively explore complex objects.

## Prediction Target Space

The prediction target is a 1-element `np.ndarray` containing the volume of the object.
We normalize the object volumes to have a mean of 0 and a standard deviation of 1 across the training set.
In case of the ABC Dataset, the normalization parameters are as follows:

- Mean: 15.958 cm $^3$
- Standard Deviation: 23.855 cm $^3$

Note that when running the environment for the first time, computing these values from the training dataset might take some time.
However, this is only done once and the values are cached for future runs.

## Example Usage

```python
import ap_gym

env = ap_gym.make("ABCVolume-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("ABCVolume-v0", num_envs=4)
```

## Version History

- `v0`: Initial release.

## Variants

| Environment ID           | Description                                                                                               | Preview                                                                                      |
|--------------------------|-----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| ABCVolume-train-v0       | Alias for ABCVolume-v0.                                                                                   | <img src="img/env/ABCVolume-v0.gif" alt="ABCVolume-v0" width="200px"/>                       |
| ABCVolume-test-v0        | Uses the test split of the _ABC Dataset_ instead of the train split.                                      | <img src="img/env/ABCVolume-test-v0.gif" alt="ABCVolume-test-v0" width="200px"/>             |
| ABCVolume-Depth-train-v0 | Uses a depth image instead of rendering tactile images.                                                   | <img src="img/env/ABCVolume-Depth-v0.gif" alt="ABCVolume-Depth-v0" width="200px"/>           |
| ABCVolume-Depth-test-v0  | Same as ABCVolume-Depth-train-v0 but uses the test split of the _ABC Dataset_ instead of the train split. | <img src="img/env/ABCVolume-Depth-test-v0.gif" alt="ABCVolume-Depth-test-v0" width="200px"/> |
