# Image Localization Environments

In image localization environments, the agent has to localize a given part of the image by moving a small
glimpse around. The glimpse is never large enough to see the entire image at once, so the agent has to move
around to gather information. Unlike the [Image Classification Environments](ImageClassificationVectorEnv.md),
this task is a regression task where the agent has to predict the coordinates of the target glimpse it is
provided.

Consider the following example from the [TinyImageNetLoc](TinyImageNetLoc.md) environment:
<p align="center"><img src="img/TinyImageNetLoc-v0.gif" alt="TinyImageNetLoc-v0" width="200px"/></p>
Marked in blue is the agent's current glimpse. The transparent purple box represents the target glimpse the
agent has to predict the coordinates of and the opaque purple box is the agent's current prediction. We further
mark the history of glimpses the agent has taken in a color scale ranging from red to green, red meaning that
the prediction it took at this step was far from the target and green meaning that the prediction was close to
the target.

All image localization environments in _ap_gym_ are instantiations of the
`ap_gym.envs.image_classification.ImageLocalizationVectorEnv` class and share the following properties:


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
    <td><code>Dict({</code><br/><code>&nbsp;&nbsp;"glimpse"       : Box(0.0, 1.0, (G, G, C), float32)</code><br/><code>&nbsp;&nbsp;"glimpse_pos"   : Box(-1.0, 1.0, (2,), float32)</code><br/><code>&nbsp;&nbsp;"target_glimpse": Box(0.0, 1.0, (G, G, C), float32)</code><br/><code>&nbsp;&nbsp;"time_step"     : Box(-1.0, 1.0, (), float32)</code><br/><code>})</code></td>
  </tr>
  <tr>
    <td><strong>Loss Function</strong></td>
    <td><code>ap_gym.MSELossFn()</code></td>
  </tr>
</table>


 where $G \in \mathbb{N}$ is the glimpse size and $C \in \mathbb{N}$ is the number of image channels (1 for grayscale, 3 for RGB).

## Action Space

The action is a `np.ndarray[float32]` $\in [-1, 1]^{2}$ that describes the relative movement of the glimpse sensor. The value is first projected into the unit circle and then scaled by `image_perception_config.max_step_length`, which is 0.20 (20% of the image) by default.

## Prediction Space

The prediction is a `np.ndarray[float32]` $\in \mathbb{R}^{2}$ that contains the coordinates of the agent's prediction w.r.t. the target glimpse.

## Prediction Target Space

The prediction target is a `np.ndarray[float32]` $\in \mathbb{R}^{2}$ that contains the true coordinates of the target glimpse.

## Observation Space

The observation is a dictionary with the following keys:

| Key            | Type         | Description                                                                                                                  |
|----------------|--------------|------------------------------------------------------------------------------------------------------------------------------|
| glimpse        | `np.ndarray` | `np.ndarray[float32]` $\in [0, 1]^{G \times G \times C}$ that represents a glimpse of the image.                             |
| glimpse_pos    | `np.ndarray` | `np.ndarray[float32]` $\in [-1, 1]^{2}$ that contains the normalized position of the glimpse within the image.               |
| target_glimpse | `np.ndarray` | `np.ndarray[float32]` $\in [0, 1]^{G \times G \times C}$ that represents the target glimpse.                                 |
| time_step      | `float32`    | `float32` $\in [-1, 1]$ that represents the normalized current time step between 0 and `image_perception_config.step_limit`. |

## Rewards

The reward at each timestep is  the sum of:
- A small action regularization equal to $10^{-3} \cdot{} \lVert \textit{action}\rVert$.
- The negative mean squared error between the agent's prediction and the target.

## Starting State

The glimpse starts at a uniformly random position within the image.

## Episode End

The episode ends with the terminate flag set if the maximum number of steps (`image_perception_config.step_limit`) is reached.

## Arguments

| Name                      | Type                    | Default       | Description                                                                                                                             |
|---------------------------|-------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `image_perception_config` | `ImagePerceptionConfig` |               | Configuration of the image perception environment. See the [ImagePerceptionConfig documentation](ImagePerceptionConfig.md) for details. |
| `render_mode`             | `Literal['rgb_array']`  | `'rgb_array'` | Rendering mode (currently only `"rgb_array"` is supported).                                                                             |

## Overview of Implemented Environments

| Environment ID                           | Image type | # data points | Image size | Glimpse size | Step limit | Image description                                                                                       |
|------------------------------------------|------------|---------------|------------|--------------|------------|---------------------------------------------------------------------------------------------------------|
| [CIFAR10Loc-v0](CIFAR10Loc.md)           | RGB        | 50000         | 32x32      | 5x5          | 16         | Natural images from the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).                 |
| [TinyImageNetLoc-v0](TinyImageNetLoc.md) | RGB        | 100000        | 64x64      | 10x10        | 16         | Natural images from the [Tiny ImageNet dataset](https://huggingface.co/datasets/zh-plus/tiny-imagenet). |
