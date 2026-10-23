# PrimitiveGeometriesVolume

<p align="center"><img src="img/env/PrimitiveGeometriesVolume-v0.webp" alt="PrimitiveGeometriesVolume-v0" width="200px"/></p>

This environment is part of the tactile regression environments.
Refer to the [tactile regression environments overview](TactileRegressionEnv.md) for a general description of these environments.

|                              |                                                                         |
|------------------------------|-------------------------------------------------------------------------|
| **Environment ID**           | PrimitiveGeometriesVolume-v0                                                |
| **Dataset**                  | [PrimitiveGeometries](datasets.md#primitivegeometries) (procedurally generated) |
| **Prediction Dimensions**    | 1                                                                       |
| **Step limit**               | 32                                                                      |
| **Sensor rotation**          | enabled                                                                 |
| **Object pose perturbation** | enabled                                                                 |

## Description

In the PrimitiveGeometriesVolume environment, the agent's objective is to estimate the volume of a geometric primitive: a cylinder, a cuboid, a triangular prism, or a half cylinder with randomly drawn dimensions.
Unlike in the ABCVolume environment, the objects are simple, so a few well-placed touches suffice to measure them: the agent has to find out the shape of the object, measure its footprint, and determine its height, for which, with sensor rotation enabled, it can tilt the sensor to touch the sides of the object.
Since the volume is known in closed form for every primitive, the prediction targets are exact.

The objects are generated procedurally on the fly (see the [dataset description](datasets.md#primitivegeometries)), so the agent cannot memorize them.

## Prediction Target Space

The prediction target is a 1-element `np.ndarray` containing the volume of the object.
We normalize the object volumes to have a mean of 0 and a standard deviation of 1 across the training set; the test split uses the same normalization parameters.
In case of the PrimitiveGeometries dataset, the normalization parameters are as follows:

- Mean: 75.852 cm $^3$
- Standard Deviation: 20.232 cm $^3$

The volumes range from 38.1 cm $^3$ to 167.3 cm $^3$.
These values are computed from the closed-form volumes of the training objects when the environment is first created and cached for future runs.

## Loss Function

The loss (and hence the negative reward) is the normalized Huber loss of the normalized volume described in [ABCVolume](ABCVolume.md#loss-function).

## Example Usage

```python
import ap_gym

env = ap_gym.make("PrimitiveGeometriesVolume-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("PrimitiveGeometriesVolume-v0", num_envs=4)
```

## Version History

- `v0`: Initial release.

## Variants

| Environment ID                         | Description                                                                                                                                                                          | Preview                                                                                                                     |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| PrimitiveGeometriesVolume-train-v0         | Alias for PrimitiveGeometriesVolume-v0.                                                                                                                                                  | <img src="img/env/PrimitiveGeometriesVolume-v0.webp" alt="PrimitiveGeometriesVolume-v0" width="200px"/>                             |
| PrimitiveGeometriesVolume-DR-v0            | Same as PrimitiveGeometriesVolume-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. PrimitiveGeometriesVolume-DR-train-v0 is an alias for it.             | <img src="img/env/PrimitiveGeometriesVolume-DR-v0.webp" alt="PrimitiveGeometriesVolume-DR-v0" width="200px"/>                       |
| PrimitiveGeometriesVolume-test-v0          | Uses the test split of _PrimitiveGeometries_ instead of the train split.                                                                                                                 | <img src="img/env/PrimitiveGeometriesVolume-test-v0.webp" alt="PrimitiveGeometriesVolume-test-v0" width="200px"/>                   |
| PrimitiveGeometriesVolume-DR-test-v0       | Same as PrimitiveGeometriesVolume-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled.                                                              | <img src="img/env/PrimitiveGeometriesVolume-DR-test-v0.webp" alt="PrimitiveGeometriesVolume-DR-test-v0" width="200px"/>             |
| PrimitiveGeometriesVolume-Depth-v0         | Uses a depth image instead of rendering tactile images. PrimitiveGeometriesVolume-Depth-train-v0 is an alias for it.                                                                     | <img src="img/env/PrimitiveGeometriesVolume-Depth-v0.webp" alt="PrimitiveGeometriesVolume-Depth-v0" width="200px"/>                 |
| PrimitiveGeometriesVolume-Depth-DR-v0      | Same as PrimitiveGeometriesVolume-Depth-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. PrimitiveGeometriesVolume-Depth-DR-train-v0 is an alias for it. | <img src="img/env/PrimitiveGeometriesVolume-Depth-DR-v0.webp" alt="PrimitiveGeometriesVolume-Depth-DR-v0" width="200px"/>           |
| PrimitiveGeometriesVolume-Depth-test-v0    | Same as PrimitiveGeometriesVolume-Depth-v0 but uses the test split of _PrimitiveGeometries_ instead of the train split.                                                                      | <img src="img/env/PrimitiveGeometriesVolume-Depth-test-v0.webp" alt="PrimitiveGeometriesVolume-Depth-test-v0" width="200px"/>       |
| PrimitiveGeometriesVolume-Depth-DR-test-v0 | Same as PrimitiveGeometriesVolume-Depth-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled.                                                        | <img src="img/env/PrimitiveGeometriesVolume-Depth-DR-test-v0.webp" alt="PrimitiveGeometriesVolume-Depth-DR-test-v0" width="200px"/> |
