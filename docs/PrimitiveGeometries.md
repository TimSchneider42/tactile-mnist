# PrimitiveGeometries

<p align="center"><img src="img/env/PrimitiveGeometries-v0.webp" alt="PrimitiveGeometries-v0" width="200px"/></p>

This environment is part of the tactile classification environments.
Refer to the [tactile classification environments overview](TactileClassificationEnv.md) for a general description of these environments.

|                              |                                                                         |
|------------------------------|-------------------------------------------------------------------------|
| **Environment ID**           | PrimitiveGeometries-v0                                                      |
| **Dataset**                  | [PrimitiveGeometries](datasets.md#primitivegeometries) (procedurally generated) |
| **Number of classes**        | 4                                                                       |
| **Step limit**               | 32                                                                      |
| **Sensor rotation**          | enabled                                                                 |
| **Object pose perturbation** | enabled                                                                 |

## Description

In the _PrimitiveGeometries_ environment, the agent must tell apart four geometric primitives: cylinders, cuboids, triangular prisms, and half cylinders.
The objects are large relative to the platform, so finding them is easy.
Instead, the challenge lies in telling the shapes apart: since the dimensions of every object are drawn at random, the classes overlap in size and only differ in their outline.
A touch on a flat top face looks the same for every class, so the agent has to seek out the edges and sides of the object, and with sensor rotation enabled, it can tilt the sensor to touch them.

The objects are not stored anywhere but generated procedurally on the fly: each of the 400,000 objects of the train split and the 400,000 objects of the test split is derived deterministically from its index with a stable pseudo random number generator (see the [dataset description](datasets.md#primitivegeometries)).
As the dimensions are drawn from a continuous distribution, the agent cannot memorize the objects.

## Example Usage

```python
import ap_gym

env = ap_gym.make("PrimitiveGeometries-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("PrimitiveGeometries-v0", num_envs=4)
```

## Version History

- `v0`: Initial release.

## Variants

| Environment ID                   | Description                                                                                                                                                              | Preview                                                                                                         |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| PrimitiveGeometries-train-v0         | Alias for PrimitiveGeometries-v0.                                                                                                                                            | <img src="img/env/PrimitiveGeometries-v0.webp" alt="PrimitiveGeometries-v0" width="200px"/>                             |
| PrimitiveGeometries-DR-v0            | Same as PrimitiveGeometries-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. PrimitiveGeometries-DR-train-v0 is an alias for it.             | <img src="img/env/PrimitiveGeometries-DR-v0.webp" alt="PrimitiveGeometries-DR-v0" width="200px"/>                       |
| PrimitiveGeometries-test-v0          | Uses the test split of _PrimitiveGeometries_ instead of the train split.                                                                                                     | <img src="img/env/PrimitiveGeometries-test-v0.webp" alt="PrimitiveGeometries-test-v0" width="200px"/>                   |
| PrimitiveGeometries-DR-test-v0       | Same as PrimitiveGeometries-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled.                                                        | <img src="img/env/PrimitiveGeometries-DR-test-v0.webp" alt="PrimitiveGeometries-DR-test-v0" width="200px"/>             |
| PrimitiveGeometries-Depth-v0         | Uses a depth image instead of rendering tactile images. PrimitiveGeometries-Depth-train-v0 is an alias for it.                                                               | <img src="img/env/PrimitiveGeometries-Depth-v0.webp" alt="PrimitiveGeometries-Depth-v0" width="200px"/>                 |
| PrimitiveGeometries-Depth-DR-v0      | Same as PrimitiveGeometries-Depth-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled. PrimitiveGeometries-Depth-DR-train-v0 is an alias for it. | <img src="img/env/PrimitiveGeometries-Depth-DR-v0.webp" alt="PrimitiveGeometries-Depth-DR-v0" width="200px"/>           |
| PrimitiveGeometries-Depth-test-v0    | Same as PrimitiveGeometries-Depth-v0 but uses the test split of _PrimitiveGeometries_ instead of the train split.                                                                | <img src="img/env/PrimitiveGeometries-Depth-test-v0.webp" alt="PrimitiveGeometries-Depth-test-v0" width="200px"/>       |
| PrimitiveGeometries-Depth-DR-test-v0 | Same as PrimitiveGeometries-Depth-test-v0 but with [domain randomization](TactilePerceptionConfig.md#sensor-noise) enabled.                                                  | <img src="img/env/PrimitiveGeometries-Depth-DR-test-v0.webp" alt="PrimitiveGeometries-Depth-DR-test-v0" width="200px"/> |
