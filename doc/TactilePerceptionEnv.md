# Tactile Perception Environments

**This guide assumes you are familiar with ap_gym.
If you are not, please refer to the [ap_gym](https://github.com/TimSchneider42/active-perception-gym) documentation.**

In tactile perception environments, the agent has to identify properties of 3D objects by exploring them with a [GelSight Mini](https://www.gelsight.com/gelsightmini/) tactile sensor.
The agent does not have access to the location of the objects and also receives no visual input.
Instead, it must actively control the sensor to find and explore them.

Currently implemented are the following tasks, which are described in more detail in their respective documentations:

<table style="border-collapse: collapse; border: none;">
    <tr style="border: none;">
        <td align="center" style="border: none; padding: 10px;">
            <img src="img/env/TactileMNIST-v0.gif" alt="TactileMNIST-v0" width="200px"/><br/>
            <a href="TactileMNIST.md">
                TactileMNIST-v0
            </a>
        </td>
        <td align="center" style="border: none; padding: 10px;">
            <img src="img/env/Starstruck-v0.gif" alt="Starstruck-v0" width="200px"/><br/>
            <a href="Starstruck.md">
                Starstruck-v0
            </a>
        </td>
        <td align="center" style="border: none; padding: 10px;">
            <img src="img/env/Toolbox-v0.gif" alt="Toolbox-v0" width="200px"/><br/>
            <a href="Toolbox.md">
                Toolbox-v0
            </a>
        </td>
        <td align="center" style="border: none; padding: 10px;">
            <img src="img/env/TactileMNISTVolume-v0.gif" alt="TactileMNISTVolume-v0" width="200px"/><br/>
            <a href="TactileMNISTVolume.md">
                TactileMNISTVolume-v0
            </a>
        </td>
    </tr>
</table>

For an example usage of tactile perception environments, see `example/tactile_mnist_env.py`.

All tactile perception environments are instantiations of the `tactile_mnist.TactilePerceptionVectorEnv` class and share the following properties:

## Properties

<table>
    <tr>
        <td><strong>Action Space</strong></td>
        <td>
            <code>Dict({</code><br>
            <code>&nbsp;&nbsp;"sensor_target_pos_rel":Box(-1.0, 1.0, shape=(3,), dtype=np.float32),</code><br>
            <code>&nbsp;&nbsp;"sensor_target_rot_rel": Box(-1.0, 1.0, shape=(6,), dtype=np.float32)</code><br>
            <code>})</code><br/>
            <code>"sensor_target_rot_rel"</code> is only present if the environment allows sensor orientation.
        </td>
    </tr>
    <tr>
        <td><strong>Observation Space</strong></td>
        <td>
            <code>Dict({</code><br>
            <code>&nbsp;&nbsp;"sensor_img": ap_gym.ImageSpace(width=W, height=H, channels=3, dtype=np.float32),</code><br>
            <code>&nbsp;&nbsp;"sensor_pos": Box(-1.0, 1.0, shape=(3,), dtype=np.float32)</code><br>
            <code>&nbsp;&nbsp;"sensor_rot": Box(-1.0, 1.0, shape=(6,), dtype=np.float32)</code><br>
            <code>&nbsp;&nbsp;"time_step": Box(-1.0, 1.0, shape=(), dtype=np.float32)</code><br>
            <code>})</code><br/>
            <code>"sensor_rot"</code> is only present if the environment allows sensor orientation.
        </td>
    </tr>
</table>


$W \in \mathbb{N}$ is the tactile image width (default 64), and $H \in \mathbb{N}$ is the tactile image height (default 64).

## Action Space

The action is a dictionary with the following keys:

| Key                       | Type         | Description                                                                                                                                                                                                             |
|---------------------------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `"sensor_target_pos_rel"` | `np.ndarray` | 3-element `float32` numpy vector containing the normalized relative linear target movement of the sensor in the range $[-1, 1]$.                                                                                        |
| `"sensor_target_rot_rel"` | `np.ndarray` | 3-element `float32` numpy vector containing the normalized relative rotational target movement of the sensor as rotation vector in the range $[-1, 1]$. This key only exists if the environment allows sensor rotation. |

### Translation

To compute the next sensor position, the `"sensor_target_pos_rel"` value is first projected into the unit sphere and multiplied with the maximum distance the sensor can move in one step.
The maximum sensor movement is computed from the `transfer_timedelta_s`, `linear_acceleration`, and `linear_velocity` parameters.
The resulting vector is then added to the current sensor position.
To ensure that the sensor is in contact with the object while not penetrating it, the sensor is moved towards or away from the target position until it touches the object or the cell surface.
This movement is always perpendicular to the sensor's sensing surface.

### Orientation

To compute the next sensor orientation, the `"sensor_target_rot_rel"` value is first projected into the unit sphere and multiplied with the maximum rotation the sensor can perform in one step.
The resulting rotation is then multiplied with the current sensor orientation to get the new target sensor orientation.
Since we only want to allow to point downwards, we ensure that the angle between the sensor's z-axis and the world's z-axis is at maximum `max_tilt_angle` (default: 90 degrees).
This constraint is enforced by projecting the target sensor orientation back into the valid orientation space if it exceeds the maximum tilt angle, which yields the final sensor orientation.

## Observation Space

The observation is a dictionary with the following keys:

| Key            | Type         | Description                                                                                                                                                       |
|----------------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `"sensor_img"` | `np.ndarray` | $H \times W \times 3$ numpy array of type `float32` representing a tactile reading where each pixel is in the range $[-1, 1]$.                                    |
| `"sensor_pos"` | `np.ndarray` | 3-element `float32` numpy vector containing the normalized position of the sensor in the range $[-1, 1]$.                                                         |
| `"sensor_rot"` | `np.ndarray` | 6-element `float32` numpy vector containing the orientation of the sensor in the range $[-1, 1]$. This key only exists if the environment allows sensor rotation. |
| `"time_step"`  | `float`      | The current time step between 0 and `step_limit` normalized to the range $[-1, 1]$. Only present if timeout behavior is set to `"terminate"` (default).           |

We model 3D orientations as 6D vectors as suggested by [Zhou et al. (2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html).
Unlike this work though, we include the second and third column of the rotation matrix instead of the first and second, as it helps us to ensure that the sensor only receives downwards pointing target orientations.

## Rewards

The reward at each timestep is a sum of:

- A small action regularization equal to $10^{-3} \cdot{} \lVert action\rVert$.
- The loss of the current prediction of the agent

## Starting State

The tactile sensor starts at a randomly sampled pose in the workspace.
Specifically, the position is uniformly randomly samples from $[-1, 1]^2 \times [0]$.
For the orientation, if enabled, we sample a polar angle $\theta$ uniformly from $[0, \texttt{max\\_tilt\\_angle}]$, an azimuthal angle $\phi$ from $[-\pi, \pi]$, and a $z$-axis rotation angle $\psi$ from $[-\pi, \pi]$.
Then, we construct a rotation from the resulting euler angles.

## Episode End

The episode ends with the terminate flag set when the maximum number of steps (`step_limit`) is reached.

## Usage Example

Here is an example of how to use the environments:

```python
import ap_gym
import numpy as np

env = ap_gym.make("tactile_mnist:TactileMNIST-v0")

# Alternatively:
# env = ap_gym.make("tactile_mnist:Starstruck-v0")

env.reset()
for i in range(10):
    action = {
        "action": {
            "sensor_target_pos_rel": np.random.uniform(-1, 1, size=3),
            # Uncomment the following line if the environment uses sensor orientation
            # "sensor_target_rot_rel": np.random.uniform(-1, 1, size=6),
        },
        "prediction": np.ones(10)
    }

    obs, _, _, _, info = env.step(action)
    sensor_img = obs["sensor_img"]  # 64 x 64 x 3 tactile image
    sensor_pos = obs["sensor_pos"]  # Normalized 3D sensor position
    time_step = info["time_step"]  # Normalized current time step
    # Only if the environment uses sensor orientation
    # sensor_rot = obs["sensor_rot"]  # 6D sensor orientation
    ground_truth_label = info["prediction"]["target"]  # integer ground truth label of the object
```

A full example can be found in [example/tactile_mnist_env.py](example/tactile_mnist_env.py) or [example/tactile_mnist_env_vec.py](example/tactile_mnist_env_vec.py) for the vectorized version.

## Arguments

| Parameter                        | Type                            | Default       | Description                                                                                                                                   |
|----------------------------------|---------------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `config`                         | `TactilePerceptionConfig`       |               | Configuration of the tactile perception environment. See the [TactilePerceptionConfig documentation](TactilePerceptionConfig.md) for details. |
| `num_envs`                       | `int`                           |               | Number of parallel environments to create.                                                                                                    |
| `single_prediction_space`        | `gym.Space[PredType]`           |               | The prediction space of the environment.                                                                                                      |
| `single_prediction_target_space` | `gym.Space[PredTargetType]`     |               | The prediction target space of the environment.                                                                                               |
| `loss_fn`                        | `ap_gym.LossFn`                 |               | The loss function of the environment.                                                                                                         |
| `render_mode`                    | `Literal["rgb_array", "human"]` | `"rgb_array"` | Which render mode to use.                                                                                                                     |

## Types of Tactile Perception Environments

There are currently two types of tactile perception environments: [Tactile Classification Environments](TactileClassificationEnv.md) and [Tactile Regression Environments](TactileRegressionEnvs.md).
Check out their respective documentations for more details.
