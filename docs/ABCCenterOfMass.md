# ABCCenterOfMass

<p align="center"><img src="img/env/ABCCenterOfMass-v0.gif" alt="ABCCenterOfMass-v0" width="200px"/></p>

This environment is part of the tactile regression environments.
Refer to the [tactile regression environments overview](TactileRegressionEnv.md) for a general description of these environments.

|                              |                                        |
|------------------------------|----------------------------------------|
| **Environment ID**           | ABCCenterOfMass-v0                     |
| **Dataset**                  | [ABC Dataset](datasets.md#abc-dataset) |
| **Prediction Dimensions**    | 2                                      |
| **Step limit**               | 32                                     |
| **Sensor rotation**          | enabled                                |
| **Object pose perturbation** | enabled                                |

## Description

In the ABCCenterOfMass environment, the agent's objective is to locate an object from the [ABC Dataset](datasets.md#abc-dataset) positioned randomly on a platform and estimate its precise 2D center of mass projected onto the plate.
Unlike the Toolbox task, in which the object shape is known a-priori, in ABCCenterOfMass, the agent faces a completely unknown object in every episode.
Hence, this task tests the agent’s ability to both find and thoroughly explore the shape of an unknown object through sequential tactile exploration.

## Prediction Target Space

The prediction target is a 2-element `np.ndarray` containing the 2D coordinates of the object's center of mass in platform coordinates, normalized to the range $[-1, 1]$.

## Example Usage

```python
import ap_gym

env = ap_gym.make("ABCCenterOfMass-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("ABCCenterOfMass-v0", num_envs=4)
```

## Version History

- `v0`: Initial release.

## Variants

| Environment ID                 | Description                                                                                                     | Preview                                                                                                  |
|--------------------------------|-----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| ABCCenterOfMass-train-v0       | Alias for ABCCenterOfMass-v0.                                                                                   | <img src="img/env/ABCCenterOfMass-v0.gif" alt="ABCCenterOfMass-v0" width="200px"/>                       |
| ABCCenterOfMass-test-v0        | Uses the test split of the _ABC Dataset_ instead of the train split.                                            | <img src="img/env/ABCCenterOfMass-test-v0.gif" alt="ABCCenterOfMass-test-v0" width="200px"/>             |
| ABCCenterOfMass-Depth-train-v0 | Uses a depth image instead of rendering tactile images.                                                         | <img src="img/env/ABCCenterOfMass-Depth-v0.gif" alt="ABCCenterOfMass-Depth-v0" width="200px"/>           |
| ABCCenterOfMass-Depth-test-v0  | Same as ABCCenterOfMass-Depth-train-v0 but uses the test split of the _ABC Dataset_ instead of the train split. | <img src="img/env/ABCCenterOfMass-Depth-test-v0.gif" alt="ABCCenterOfMass-Depth-test-v0" width="200px"/> |
