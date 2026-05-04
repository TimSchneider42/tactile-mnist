# LightDark

<p align="center"><img src="img/LightDark-v0.gif" alt="LightDark-v0" width="200px"/></p>

## Description

In the LightDark Environment, the agent must estimate its position based on noisy observations, where the noise
level depends on the brightness of the surrounding area. The environment simulates an active regression task
where the agent can move to areas with better visibility to improve its position estimation.

This environment is useful for testing active regression models, where the agent must strategically explore its
environment to obtain more reliable observations before making predictions.

The visualization shown above has to be interpreted as follows:

- **Blue dot**: Agent's current position.
- **Green transparent circle**: Observation uncertainty (higher in dark regions).
- **Purple dot**: Agent's last prediction.
- **Light blue dot**: Agent's previous position (this is what the agent's prediction tries to approximate).
- **White background**: Bright regions with low uncertainty.
- **Dark background**: Dark regions with high uncertainty.


## Properties

<table>
  <tr>
    <td><strong>Action Space</strong></td>
    <td><code>Box(-1.0, 1.0, (2,), float32)</code></td>
  </tr>
  <tr>
    <td><strong>Prediction Space</strong></td>
    <td><code>Box(-inf, inf, (2,), float32)</code></td>
  </tr>
  <tr>
    <td><strong>Prediction Target Space</strong></td>
    <td><code>Box(-inf, inf, (2,), float32)</code></td>
  </tr>
  <tr>
    <td><strong>Observation Space</strong></td>
    <td><code>Dict({</code><br/><code>&nbsp;&nbsp;"noisy_position": Box(-2.0, 2.0, (2,), float32)</code><br/><code>&nbsp;&nbsp;"time_step"     : Box(-1.0, 1.0, (), float32)</code><br/><code>})</code></td>
  </tr>
  <tr>
    <td><strong>Loss Function</strong></td>
    <td><code>ap_gym.MSELossFn()</code></td>
  </tr>
</table>


## Action Space

The action is a `np.ndarray[float32]` $\in [-1, 1]^{2}$ that describes the agent's relative movement. The value is first projected into the unit circle and then scaled by 0.15. If the agent moves outside the valid region ($[-1, 1]^2$), the episode is terminated.

## Prediction Space

The prediction is a `np.ndarray[float32]` $\in \mathbb{R}^{2}$ that represents the predicted position of the agent.

## Prediction Target Space

The prediction target is a `np.ndarray[float32]` $\in \mathbb{R}^{2}$ that represents the true position of the agent.

## Observation Space

The observation is a dictionary with the following keys:

| Key            | Type         | Description                                                                                                                                                                                                                                                                    |
|----------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| noisy_position | `np.ndarray` | `np.ndarray[float32]` $\in [-2, 2]^{2}$ that contains a noisy estimate of the agent's position. The level of noise depends on the brightness of the area the agent is in. A brighter area results in a lower noise level, while a darker area results in a higher noise level. |
| time_step      | `float32`    | `float32` $\in [-1, 1]$ that represents the normalized current time step between 0 and `max_episode_steps` (default 50).                                                                                                                                                       |

## Rewards

The reward at each timestep is  the sum of:
- A small action regularization equal to $10^{-3} \cdot{} \lVert \textit{action}\rVert$.
- A constant reward of $0.1$ to ensure that the reward stays positive and the agent does not learn to terminate the episode on purpose.
- The negative mean squared error between the agent's prediction and the target.

## Starting State

The agent's initial position is uniformly randomly sampled from the range $[-1, 1]^2$.

## Episode End

The episode ends with the terminate flag set if one of the following conditions is met:
 1. The agent moves out of bounds.
2. The maximum number of steps (`max_episode_steps`) is reached.

## Arguments

| Name          | Type                          | Default       | Description                                                 |
|---------------|-------------------------------|---------------|-------------------------------------------------------------|
| `render_mode` | `typing.Literal['rgb_array']` | `'rgb_array'` | Rendering mode (currently only `"rgb_array"` is supported). |

## Example Usage

```python

env = ap_gym.make("LightDark-v0")

# Or for the vectorized version with 4 environments:
envs = ap_gym.make_vec("LightDark-v0", num_envs=4)
```

## Version History

- `v0`: Initial version
