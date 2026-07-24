# TactileMNISTCenterOfMass

<p align="center"><img src="img/env/TactileMNISTCenterOfMass-v0.gif" alt="TactileMNISTCenterOfMass-v0" width="200px"/></p>

This environment is part of the tactile regression environments.
Refer to the [tactile regression environments overview](TactileRegressionEnv.md) for a general description of these environments.

|                              |                                  |
|------------------------------|----------------------------------|
| **Environment ID**           | TactileMNISTCenterOfMass-v0      |
| **Dataset**                  | [MNIST 3D](datasets.md#mnist-3d) |
| **Prediction Dimensions**    | 2                                |
| **Step limit**               | 32                               |
| **Sensor rotation**          | disabled                         |
| **Object pose perturbation** | enabled                          |

## Description

In the TactileMNISTCenterOfMass environment, the agent's objective is to locate a [3D MNIST](datasets.md#mnist-3d) digit positioned randomly on a platform and estimate its precise 2D center of mass projected onto the plate.
Unlike the Toolbox task, in which the object shape is known a-priori, in TactileMNISTCenterOfMass, the agent faces an unknown digit in every episode.
Hence, this task tests the agent’s ability to both find and thoroughly explore the shape of an unknown object through sequential tactile exploration.
Compared to the ABCCenterOfMass environment, the objects are simpler and less diverse in shape, making TactileMNISTCenterOfMass an easier variant of the same task.

## Prediction Target Space

The prediction target is a 2-element `np.ndarray` containing the 2D coordinates of the object's center of mass in platform coordinates, normalized to the range $[-1, 1]$.

## Example Usage

```python
import ap_gym

env = ap_gym.make("TactileMNISTCenterOfMass-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("TactileMNISTCenterOfMass-v0", num_envs=4)
```

## Version History

- `v0`: Initial release.

## Variants

| Environment ID                          | Description                                                                                              | Preview                                                                                                                    |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| TactileMNISTCenterOfMass-train-v0       | Alias for TactileMNISTCenterOfMass-v0.                                                                   | <img src="img/env/TactileMNISTCenterOfMass-v0.gif" alt="TactileMNISTCenterOfMass-v0" width="200px"/>                       |
| TactileMNISTCenterOfMass-test-v0        | Uses the test split of _MNIST 3D_ instead of the train split.                                            | <img src="img/env/TactileMNISTCenterOfMass-test-v0.gif" alt="TactileMNISTCenterOfMass-test-v0" width="200px"/>             |
| TactileMNISTCenterOfMass-Depth-train-v0 | Uses a depth image instead of rendering tactile images.                                                  | <img src="img/env/TactileMNISTCenterOfMass-Depth-v0.gif" alt="TactileMNISTCenterOfMass-Depth-v0" width="200px"/>           |
| TactileMNISTCenterOfMass-Depth-test-v0  | Same as TactileMNISTCenterOfMass-Depth-train-v0 but uses the test split of _MNIST 3D_ instead of the train split. | <img src="img/env/TactileMNISTCenterOfMass-Depth-test-v0.gif" alt="TactileMNISTCenterOfMass-Depth-test-v0" width="200px"/> |
