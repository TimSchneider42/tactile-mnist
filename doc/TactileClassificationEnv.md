# Tactile Classification Environments

In this environment, the agent controls the position and orientation of a tactile sensor, which it can move around above the sensor.
Once `env.step` is called, the sensor will be moved by the position delta specified in `action["action"]["sensor_target_pos_rel"]` and the orientation delta specified in `action["action"]["sensor_target_rot_rel"]` and moved towards the target position until it gets in contact with the object or the cell surface.
Then, in the next observation, the agent will receive a tactile image of the object along with the actual final 3D position and orientation of the sensor.

The objective of the agent is to predict the label of the object, which it does by setting `action["prediction"]` to the respective logits.
In every step, it receives the negative cross entropy loss between its prediction and the ground truth label plus an action regularization as a reward.
Hence, the agent has to use the tactile images to both predict the label of the object and to decide where to conduct the next touch.

For an example usage of `TactileClassificationEnv`, see `example/tactile_mnist_env_example.py`.

# Tactile Classification Environments

**This guide assumes you are familiar with ap_gym.
If you are not, please refer to the [ap_gym](https://github.com/TimSchneider42/active-perception-gym) documentation.**

In tactile classification environments, the agent has to classify a 3D object by exploring it with a [GelSight Mini](https://www.gelsight.com/gelsightmini/) tactile sensor.
The agent does not have access to the object's location or orientation and also receives no visual input.
Instead, it must actively control the sensor to find and classify the object.

Currently implemented are the following two tasks, which are described in more detail in their respective documentations:

<div align="center">
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
        </tr>
    </table>
</div>

All tactile classification environments are instantiations of the `tactile_mnist.TactileClassificationVectorEnv` class and share the following properties:

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
        <td><strong>Prediction Space</strong></td>
        <td><code>Box(-inf, inf, shape=(K,), dtype=np.float32)</code></td>
    </tr>
    <tr>
        <td><strong>Prediction Target Space</strong></td>
        <td><code>Discrete(K)</code></td>
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
    <tr>
        <td><strong>Loss Function</strong></td>
        <td>
            <code>ap_gym.CrossEntropyLossFn()</code>
        </td>
    </tr>
</table>


where $K \in \mathbb{N}$ is the number of classes in the environment, $W \in \mathbb{N}$ is the tactile image width, and $H \in \mathbb{N}$ is thetactile image height.

## Action Space

The action is a dictionary with the following keys:

| Key                       | Type         | Description                                                                                                                                                                                                             |
|---------------------------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `"sensor_target_pos_rel"` | `np.ndarray` | 3D numpy array of type `float32` containing the normalized relative linear target movement of the sensor in the range $[-1, 1]$.                                                                                        |
| `"sensor_target_rot_rel"` | `np.ndarray` | 3D numpy array of type `float32` containing the normalized relative rotational target movement of the sensor as rotation vector in the range $[-1, 1]$. This key only exists if the environment allows sensor rotation. |

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

## Prediction Space

The prediction is a $K$-dimensional `np.ndarray` containing the logits of the agent's prediction w.r.t. the class label.

## Prediction Target Space

The prediction target is a scalar integer in the range $[0, K - 1]$, representing the true class.

## Observation Space

The observation is a dictionary with the following keys:

| Key            | Type         | Description                                                                                                                                                       |
|----------------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `"sensor_img"` | `np.ndarray` | $H \times W \times 3$ numpy array of type `float32` representing a tactile reading where each pixel is in the range $[-1, 1]$.                                    |
| `"sensor_pos"` | `np.ndarray` | 3D numpy array of type `float32` containing the normalized position of the sensor in the range $[-1, 1]$.                                                         |
| `"sensor_rot"` | `np.ndarray` | 6D numpy array of type `float32` containing the orientation of the sensor in the range $[-1, 1]$. This key only exists if the environment allows sensor rotation. |
| `"time_step"`  | `float`      | The current time step between 0 and `step_limit` normalized to the range $[-1, 1]$.                                                                               |

We model 3D orientations as 6D vectors as suggested by [Zhou et al. (2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html).
Unlike this work though, we include the second and third column of the rotation matrix instead of the first and second, as it helps us to ensure that the sensor only receives downwards pointing target orientations.

## Rewards

The reward at each timestep is a sum of:

- A small action regularization equal to $10^{-3} \cdot{} \lVert action\rVert$.
- The negative cross-entropy loss between the agent's prediction and the true class.

## Starting State

The tactile sensor starts at a randomly sampled pose in the workspace.
Specifically, the position is uniformly randomly samples from $[-1, 1]^2 \times [0]$.
For the orientation, we sample a polar angle $\theta$ uniformly from $[0, \texttt{max_tilt_angle}]$, an azimuthal angle $\phi$ from $[-\pi, \pi]$, and a $z$-axis rotation angle $\psi$ from $[-\pi, \pi]$.
Then, we construct a rotation from the resulting euler angles.

## Episode End

The episode ends with the terminate flag set when the maximum number of steps (`step_limit`, default: 16) is reached.

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

| Parameter                       | Type                                   | Default       | Description                                                                                                                                                         |
|---------------------------------|----------------------------------------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dataset`                       | `MeshDataset \| Sequence[MeshDataset]` |               | The dataset(s) containing mesh data points. If a single dataset is provided, it is duplicated for all environments.                                                 |
| `num_envs`                      | `int`                                  |               | The number of environments running in parallel.                                                                                                                     |
| `step_limit`                    | `int`                                  | `16`          | The maximum number of steps per episode.                                                                                                                            |
| `render_mode`                   | `Literal["rgb_array", "human"]`        | `"rgb_array"` | Rendering mode. Supports `"rgb_array"` and `"human"`.                                                                                                               |
| `taxim_device`                  | `str \| None`                          | `None`        | Device identifier for the Taxim sensor.                                                                                                                             |
| `convert_image_to_numpy`        | `bool`                                 | `True`        | Whether to convert images to NumPy arrays. Otherwise they will be returned as either JAX array or torch tensors, depending on the selected backend.                 |
| `show_sensor_target_pos`        | `bool`                                 | `False`       | Whether to visually indicate the target sensor position in the rendering.                                                                                           |
| `perturb_object_pose`           | `bool`                                 | `True`        | Whether to apply random perturbations to object poses in every step. These perturbations simulate the object slightly shifting around when touched with the sensor. |
| `randomize_initial_object_pose` | `bool`                                 | `True`        | Whether to randomize the initial pose of objects.                                                                                                                   |
| `sensor_output_size`            | `Sequence[int] \| None`                | `None`        | The output size of the sensor in pixels. Defaults to `GELSIGHT_IMAGE_SIZE_PX` if not provided.                                                                      |
| `randomize_initial_sensor_pose` | `bool`                                 | `True`        | Whether to randomize the initial sensor pose.                                                                                                                       |
| `depth_only`                    | `bool`                                 | `False`       | Whether to use depth images instead of rendered tactile images.                                                                                                     |
| `allow_sensor_rotation`         | `bool`                                 | `True`        | Whether the sensor can rotate. Otherwise it is just pointing straight downwards.                                                                                    |
| `sensor_backend`                | `Literal["torch", "jax", "auto"]`      | `"auto"`      | The backend for sensor processing. `"auto"` selects the best available backend.                                                                                     |
| `linear_velocity`               | `float`                                | `0.2`         | Maximum linear velocity of the sensor (in m/s).                                                                                                                     |
| `angular_velocity`              | `float`                                | `np.pi / 2`   | Maximum angular velocity of the sensor (in rad/s).                                                                                                                  |
| `linear_acceleration`           | `float`                                | `4.0`         | Maximum linear acceleration of the sensor (in m/s²).                                                                                                                |
| `angular_acceleration`          | `float`                                | `10 * np.pi`  | Maximum angular acceleration of the sensor (in rad/s²).                                                                                                             |
| `transfer_timedelta_s`          | `float`                                | `0.2`         | The time step between two steps.                                                                                                                                    |
| `action_regularization`         | `float`                                | `1e-3`        | Regularization coefficient for actions.                                                                                                                             |
| `max_tilt_angle`                | `float`                                | `np.pi / 4`   | Maximum allowed tilt angle for the sensor.                                                                                                                          |
| `render_transparent_background` | `bool`                                 | `False`       | Whether to render the background transparent.                                                                                                                       |

## Overview of Implemented Environments

| Environment ID                     | Dataset                                 | # classes | Step Limit | Sensor Rotation | Tactile Image Size | Object Pose Perturbation | Description                                   |
|------------------------------------|-----------------------------------------|-----------|------------|-----------------|--------------------|--------------------------|-----------------------------------------------|
| [TactileMNIST-v0](TactileMNIST.md) | [MNIST 3D](datasets.md#mnist3d-v0)      | 10        | 16         | disabled        | 64x64              | enabled                  | Classify objects from the _MNIST 3D_ dataset. |
| [Starstruck-v0](Starstruck.md)     | [Starstruck](datasets.md#starstruck-v0) | 3         | 32         | disabled        | 64x64              | disabled                 | Count the number of stars in the scene.       |
