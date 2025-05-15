# Toolbox

<p align="center"><img src="img/env/Toolbox-v0.gif" alt="Toolbox-v0" width="200px"/></p>

This environment is part of the tactile regression environments.
Refer to the [tactile regression environments overview](TactileRegressionEnv.md) for a general description of these environments.

|                              |            |
|------------------------------|------------|
| **Environment ID**           | Toolbox-v0 |
| **Dataset**                  |            |
| **Prediction Dimensions**    | 4          |
| **Step limit**               | 64         |
| **Sensor rotation**          | disabled   |
| **Object pose perturbation** | enabled    |

## Description

In the Toolbox environment, the agent's objective is to locate a wrench positioned randomly on a platform and estimate its precise 2D position and 1D orientation.
Unlike the previous classification tasks, Toolbox is poses a regression problem that requires combining multiple touch observations to resolve ambiguities inherent in the wrench’s shape.
For example, touching the handle may reveal lateral placement but not longitudinal position or orientation, making it critical for the agent to explore strategically and seek out one of the wrench’s ends to accurately determine its pose.
Overall, the Toolbox tests the agent’s ability to both find and precisely localize an object through sequential tactile exploration.

## Prediction Target Space

The prediction target is a 4-element `np.ndarray` containing a pose representation of the wrench.
The first two elements are the 2D coordinates of the wrench's base frame (located in the center of the handle) in platform coordinates, normalized to the range $[-1, 1]$.
The last two elements are the sine and cosine of the wrench's rotation around the Z-axis.

## Example Usage

```python
import ap_gym

env = ap_gym.make("Toolbox-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("Toolbox-v0", num_envs=4)
```

## Version History

- `v0`: Initial release.

## Variants

| Environment ID   | Description                                             | Preview                                                                        |
|------------------|---------------------------------------------------------|--------------------------------------------------------------------------------|
| Toolbox-Depth-v0 | Uses a depth image instead of rendering tactile images. | <img src="img/env/Toolbox-Depth-v0.gif" alt="Toolbox-Depth-v0" width="200px"/> |
