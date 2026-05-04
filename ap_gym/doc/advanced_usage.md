# Advanced Usage

This section describes some more advanced usage of _ap_gym_, namely creating custom environments and wrappers.

## Defining Own Environments

Implementing custom environments entails subclassing `ap_gym.ActivePerceptionEnv` or `ap_gym.ActivePerceptionVectorEnv`.
An example implementation could look like this:

```python
from typing import Any

import numpy as np
import gymnasium as gym
import ap_gym


# Generic arguments are
# ObsType: Type of the observation
# ActType: Type of the action
# PredType: Type of the prediction
# PredTargetType: Type of the prediction target

class MyCustomEnv(ap_gym.ActivePerceptionEnv[np.ndarray, np.ndarray, np.ndarray, int]):
    def __init__(self):
        self.observation_space = ap_gym.ImageSpace(width=5, height=5, channels=1)
        inner_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        prediction_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(10,), dtype=np.float32
        )
        self.action_space = ap_gym.ActivePerceptionActionSpace(
            inner_action_space, prediction_space
        )
        self.prediction_target_space = gym.spaces.Discrete(10)
        self.loss_fn = ap_gym.CrossEntropyLossFn()
        self._current_class = None

    def _reset(
        self, *, options: dict[str, Any | None] = None
    ) -> tuple[np.ndarray, dict[str, Any], int]:
        self._current_class = ...  # Randomly choose a class
        obs = ...  # Generate the initial observation
        info = ...  # Additional information
        return obs, info, self._current_class

    def _step(
        self, action: np.ndarray, prediction: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any], int]:
        obs = ...  # Generate the next observation
        base_reward = ...  # Compute the base reward (the loss function will be
                           # evaluated by ap_gym.ActivePerceptionEnv)
        terminated = ...  # Whether the episode is terminated
        truncated = ...  # Whether the episode is truncated
        info = ...  # Additional information
        self._current_class = ...  # The prediction target may change over time
        return obs, base_reward, terminated, truncated, info, self._current_class
```

For vectorized environments, subclass `ap_gym.ActivePerceptionVectorEnv` instead:

```python
from typing import Any

import numpy as np
import gymnasium as gym
import ap_gym


# Generic arguments are
# ObsType: Type of the observation
# ActType: Type of the action
# PredType: Type of the prediction
# PredTargetType: Type of the prediction target
# ArrayType: Type of the arrays (typically np.ndarray)


class MyCustomVectorEnv(
    ap_gym.ActivePerceptionVectorEnv[
        np.ndarray, np.ndarray, np.ndarray, int, np.ndarray
    ]
):
    def __init__(self, num_envs: int):
        self.num_envs = num_envs

        self.single_observation_space = ap_gym.ImageSpace(width=5, height=5, channels=1)
        single_inner_action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )
        single_prediction_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(10,), dtype=np.float32
        )
        self.single_action_space = ap_gym.ActivePerceptionActionSpace(
            single_inner_action_space, single_prediction_space
        )
        self.single_prediction_target_space = gym.spaces.Discrete(10)

        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, num_envs
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, num_envs
        )
        self.prediction_target_space = gym.vector.utils.batch_space(
            self.single_prediction_target_space, num_envs
        )

        self.loss_fn = ap_gym.CrossEntropyLossFn()
        self._current_classes = None

    def _reset(
        self, *, options: dict[str, Any | None] = None
    ) -> tuple[np.ndarray, dict[str, Any], int]:
        self._current_class = ...  # Randomly choose classes (now an array)
        obs = ...  # Generate the initial observation
        info = ...  # Additional information
        return obs, info, self._current_class

    def _step(
        self, action: np.ndarray, prediction: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any], int]:
        obs = ...  # Generate the next observation
        base_reward = ...  # Compute the base reward (the loss function will be
                           # evaluated by ap_gym.ActivePerceptionEnv)
        terminated = ...  # Whether the episode is terminated
        truncated = ...  # Whether the episode is truncated
        info = ...  # Additional information
        self._current_class = ...  # The prediction target may change over time
        return obs, base_reward, terminated, truncated, info, self._current_class
```

### Custom Loss Functions

If you wish to define your own loss function, subclass `ap_gym.LossFn` and implement the `numpy`, `torch`, and `jax`
functions.
Alternatively, you can use `ap_gym.LambdaLossFn` to define a loss function based on a custom function:

```python
import numpy as np
import torch
import jax.numpy as jnp
import ap_gym

mse_loss_fn = ap_gym.LambdaLossFn(
    lambda prediction, target, batch_shape: np.mean(
        (prediction - target) ** 2
    ),  # Numpy implementation
    lambda prediction, target, batch_shape: torch.mean(
        (prediction - target) ** 2
    ),  # PyTorch implementation
    lambda prediction, target, batch_shape: jnp.mean(
        (prediction - target) ** 2
    ),  # JAX implementation
)
```

### Custom Classification Environments

Since a common class of tasks in active perception is classification, _ap_gym_ provides a base classes for classification environments: `ap_gym.ActiveClassificationEnv` and `ap_gym.ActiveClassificationVectorEnv`.
Aside from defining prediction and prediction target spaces, and using a cross entropy loss as loss function, `ap_gym.ActiveClassificationVectorEnv` also logs some statistics in the info dictionary.

Here is a brief usage example for a custom vectorized classification environment:

```python
from typing import Any

import numpy as np
import ap_gym
import gymnasium as gym


# Generic arguments are
# ObsType: Type of the observation
# ActType: Type of the action

class MyClassificationVectorEnv(
    ap_gym.ActiveClassificationVectorEnv[np.ndarray, np.ndarray]
):
    def __init__(self, num_envs: int):
        single_inner_action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )
        super().__init__(num_envs, 10, single_inner_action_space)  # 10 Classes

        self.single_observation_space = ap_gym.ImageSpace(width=5, height=5, channels=1)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, num_envs
        )

    def _reset(
        self, *, options: dict[str, Any | None] = None
    ) -> tuple[
        np.ndarray, dict[str, Any], np.ndarray
    ]: ...  # Implement _reset as in the MyCustomVectorEnv example

    def _step(
        self, action: np.ndarray, prediction: np.ndarray
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray
    ]: ...  # Implement _step as in the MyCustomVectorEnv example
```

The usage of `ap_gym.ActiveClassificationEnv` is analogous.

If you wish to see a full example, check out the [image classification](doc/ImageClassification.md) [implementation](ap_gym/envs/image_classification.py).


### Custom Regression Environments

Another common class of tasks in active perception are regression tasks.
Therefore, _ap_gym_ provides a base classes for regression environments: `ap_gym.ActiveRegressionEnv` and `ap_gym.ActiveRegressionVectorEnv`.
Aside from defining prediction and prediction target spaces, and using the mean squared error (MSE) as loss function, `ap_gym.ActiveRegressionVectorEnv` also logs some statistics in the info dictionary.

Here is a brief usage example for a custom vectorized regression environment:

```python
from typing import Any

import numpy as np
import ap_gym
import gymnasium as gym


# Generic arguments are
# ObsType: Type of the observation
# ActType: Type of the action

class MyRegressionVectorEnv(ap_gym.ActiveRegressionVectorEnv[np.ndarray, np.ndarray]):
    def __init__(self, num_envs: int):
        single_inner_action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )
        super().__init__(num_envs, 2, single_inner_action_space)  # 2D predictions

        self.single_observation_space = ap_gym.ImageSpace(width=5, height=5, channels=1)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, num_envs
        )

    def _reset(
        self, *, options: dict[str, Any | None] = None
    ) -> tuple[
        np.ndarray, dict[str, Any], np.ndarray
    ]: ...  # Implement _reset as in the MyCustomVectorEnv example

    def _step(
        self, action: np.ndarray, prediction: np.ndarray
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray
    ]: ...  # Implement _step as in the MyCustomVectorEnv example
```

The usage of `ap_gym.ActiveRegressionEnv` is analogous.

If you wish to see a full example, check out the [LightDark](doc/LightDark.md) [implementation](ap_gym/envs/light_dark.py).

### Converting Vectorized Environments into Single Environments

_ap_gym_ provides functionality to convert vectorized environments into single environments with the
`ap_gym.VectorToSingleWrapper`.
By defining a vectorized environment and wrapping it with the `ap_gym.VectorToSingleWrapper`, the environment can be
used as a single environment without the need of rewriting the environment's code.
`ap_gym.VectorToSingleWrapper` will take care of converting all spaces and transforming the inputs and outputs of the
step and reset functions accordingly.
Custom methods or fields will be mapped through to the original environment without conversion.

```python
vec_env = ap_gym.make_vec("MNIST-v0", num_envs=1)  # Is of type ap_gym.ActivePerceptionVectorEnv
env = ap_gym.VectorToSingleWrapper(vec_env)  # Is of type ap_gym.ActivePerceptionEnv
```

## Defining Custom Wrappers

Similar to Gymnasium, custom wrappers are created by subclassing `ap_gym.ActivePerceptionWrapper` or
`ap_gym.ActivePerceptionVectorWrapper`, respectively.
By default, these wrappers do not modify the environment in any way and pass all calls through to the wrapped
environment.
If you want to modify the behavior of the environment, you can override the respective methods (`step` or `reset`).
To change the definition of the spaces, simply set `_prediction_space`, `_prediction_target_space`,
`_inner_action_space`, `_single_prediction_space`, `_single_prediction_target_space`, `_single_inner_action_space`, and
`_loss_fn` to the desired values.

As an example, consider the following vector wrapper, which removes the action space and chooses random actions instead:

```python
import numpy as np

import gymnasium as gym
import ap_gym


class RandomActionWrapper(ap_gym.ActivePerceptionVectorWrapper):
    def __init__(self, env: ap_gym.ActivePerceptionVectorEnv):
        super().__init__(env)
        self._inner_action_space = gym.spaces.Tuple(())

    def step(self, action):
        return super().step(
            {
                "action": self.inner_action_space.sample(),
                "prediction": action["prediction"],
            }
        )


env = ap_gym.make_vec("MNIST-v0", num_envs=4)
env = RandomActionWrapper(env)
env.reset()
env.step({"action": np.zeros((4, 0)), "prediction": np.zeros((4, 10))})
```
