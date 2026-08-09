# Tactile Shape Reconstruction Environments

In tactile shape reconstruction environments, the agent has to reconstruct the full 3D shape of an object by exploring it with a [GelSight Mini](https://www.gelsight.com/gelsightmini/) tactile sensor.
The shape is represented by a compact latent embedding of a [COD-VAE](https://github.com/TimSchneider42/cod-vae) shape autoencoder (Cho et al., ICCV 2025), which the agent has to regress to.
Since every touch reveals only a small patch of the object's surface, the agent has to integrate information from many touches to reconstruct the global shape.

Tactile shape reconstruction environments are [tactile regression environments](TactileRegressionEnv.md), but they deviate from the standard regression scheme: their prediction target is a dict identifying the ground-truth geometry rather than a copy of the prediction space, and their loss is the COD-VAE reconstruction loss instead of an MSE.
For more details on tactile perception environments in general, see the [Tactile Perception Environments documentation](TactilePerceptionEnv.md).

Currently implemented are the following tasks, which are described in more detail in their respective documentations:

<div align="center">
    <table style="border-collapse: collapse; border: none;">
        <tr style="border: none;">
            <td align="center" style="border: none; padding: 10px;">
                <img src="img/env/TactileMNISTShape-v0.webp" alt="TactileMNISTShape-v0" width="200px"/><br/>
                <a href="TactileMNISTShape.md">
                    TactileMNISTShape-v0
                </a>
            </td>
            <td align="center" style="border: none; padding: 10px;">
                <img src="img/env/ABCShape-v0.webp" alt="ABCShape-v0" width="200px"/><br/>
                <a href="ABCShape.md">
                    ABCShape-v0
                </a>
            </td>
        </tr>
    </table>
</div>

All tactile shape reconstruction environments share the following properties:

## Properties

<table>
    <tr>
        <td><strong>Prediction Space</strong></td>
        <td><code>Box(low, high, shape=(N,), dtype=np.float32)</code></td>
    </tr>
    <tr>
        <td><strong>Prediction Target Space</strong></td>
        <td><code>Dict(box: Box(-inf, inf, shape=(4,)), mesh_index: Box(0, num_meshes - 1, shape=(), dtype=np.int64), position: Box(-inf, inf, shape=(3,)), quaternion: Box(-1.0, 1.0, shape=(4,)))</code></td>
    </tr>
    <tr>
        <td><strong>Loss Function</strong></td>
        <td><code>tactile_mnist.CODVAEReconstructionLossFn</code></td>
    </tr>
</table>

where $N = k \cdot d + 4$ is the size of the COD-VAE full latent (132 for the default model) and the bounds `low` and `high` are derived from per-dataset target statistics (see below).

## Prediction Space

The prediction is a [COD-VAE](https://github.com/TimSchneider42/cod-vae) *full latent* (see `cod_vae.CODVAEBase.pack_full_latent`): $k$ latent vectors of dimension $d$ describing the object in the model's $[-1, 1]$ cube, followed by the bounding box center (3 entries) and size (1 entry) of the posed object.
This yields a $k \cdot d + 4 = 132$-element `np.ndarray` for the default model ($k = 4$, $d = 32$).
Since the cube normalization removes the object's position and scale, the latent jointly encodes the object's shape and its current (randomized and perturbed) orientation on the platform.
The object's position and scale are instead captured by the remaining four entries: the center and maximum half-extent of the posed mesh's axis-aligned bounding box, both divided by half the cell size, i.e. expressed in a platform frame whose vertex positions are normalized to $[-1, 1]$ by half the cell size.
A single size scalar suffices because the cube normalization is isotropic; the aspect ratios of the bounding box are part of the shape encoded in the latent.
Unlike a factored orientation-plus-shape representation, this joint representation remains well-defined even for (rotationally) symmetric objects, for which the orientation alone would be ambiguous.
A mesh can be reconstructed from a prediction with the same COD-VAE model via `cod_vae.CODVAEBase.decode_mesh_full` (passing half the cell size as `frame_half_size`), and occupancy can be queried at platform-frame points via `decode_full`.
If the `renderer_show_shadow_objects` option is enabled, the environment decodes the agent's current prediction every step and renders it as a translucent shadow object; this is disabled by default, as it requires a COD-VAE decoder pass per step.

COD-VAE latents are KL-regularized towards a standard normal, so they are approximately unit scale.
The prediction space bounds are nonetheless not fixed; they are derived from per-dataset target statistics computed once on first use and cached on disk: the initial pose distribution of every object is covered by a linspace over the rotation perturbation, each sampled pose's target (the COD-VAE embedding of the posed mesh and its normalized bounding box) is computed, and per-dimension mean, standard deviation, minimum, and maximum are accumulated (exposed as `prediction_target_stats` for agent-side output normalization).
The prediction space is bounded by the observed per-dimension extremes, widened by a small margin plus the expected pose perturbation drift in the bounding box center xy.

## Prediction Target

The prediction target does not prescribe a particular latent; it identifies the ground-truth geometry instead: a dict with the index of the object's mesh in the dataset (`mesh_index`), the pose mapping the raw dataset mesh into the platform frame (`position` and scalar-last `quaternion`; any mesh pre-processing rotation, e.g. `smallest_dimension_up`, is composed into the quaternion), and the posed object's ground-truth bounding box (`box`: normalized center and maximum half-extent, computed with the exact functions COD-VAE's encoding uses and thus directly comparable to the prediction's last four entries).

## Loss

The loss is the loss COD-VAE is trained with: binary cross entropy of the occupancy decoded from the prediction against the ground-truth mesh occupancy, evaluated at uniform volume query points (weight 1.0) re-sampled on every evaluation from the mesh's slice of a shared point database, and at a fixed per-mesh set of near-surface points (weight 0.1), following the sdf_gen recipe COD-VAE is trained on.
Query points are re-sampled on every loss evaluation and mapped through the object's pose and COD-VAE's full-latent decoding, which places them in the model's cube via the *predicted* bounding box center and size, so errors in these entries directly manifest as occupancy errors (see `tactile_mnist.CODVAEReconstructionLossFn`).
Additionally, the mean squared error between the predicted and the ground-truth bounding box parameters (averaged over the four normalized components, weight 1.0) is added to the loss.
This term is the bounding box's only gradient source — the decoder's own box gradient is always stopped, as it is noisy and vanishes when the predicted box does not overlap the object — and provides a smooth localization gradient at any distance.

The reported loss is normalized by the expected loss of blind guessing, so an uninformed prediction scores around 1 and a perfect one 0: the bounding box MSE term's blind-guessing expectation follows analytically from the variance of the box targets, while the occupancy terms' expectation is estimated empirically (as part of the cached target statistics) by scoring the best uninformed static prediction — the mean latent and mean bounding box — against every sampled pose of every object.

The occupancy pools (volume occupancy labels and near-surface points) of the entire dataset are computed once on first use (in parallel) and cached on disk; they are loaded onto the VAE's device if they fit within a configurable fraction of its memory (`max_pool_vram_fraction`, default 25%), otherwise per-batch query data is streamed to the device on demand.
The dataset meshes are assumed to be watertight.
The COD-VAE model is loaded in float16 by default (`half_precision`), which roughly halves the memory footprint of a loss evaluation and doubles its throughput on a GPU; as the precision is a property of the model, it applies uniformly to the loss and to the shadow-object reconstruction.
The query/pose arithmetic and the BCE are always float32.

## Metrics

All tactile shape reconstruction environments log the following metrics in the `info` dictionary:

- `center_error`: the Euclidean distance between the predicted and the true bounding box center in meters.
- `size_error`: the absolute error of the predicted bounding box size in meters.

## Changing the COD-VAE Model

The COD-VAE model can be changed via the `model` argument, which accepts a Hugging Face Hub repository id or a local npz checkpoint:

```python
import ap_gym

env = ap_gym.make("TactileMNISTShape-v0", model="TimSchneider42/cod-vae-4x32")
```

The prediction dimensionality follows the chosen model ($k \cdot d + 4$).

## Overview of Implemented Environments

| Environment ID                                     | Dataset                                | Step Limit | Sensor Rotation | Object Pose Perturbation | Description                                                 |
|----------------------------------------------------|----------------------------------------|------------|-----------------|--------------------------|-------------------------------------------------------------|
| [TactileMNISTShape-v0](TactileMNISTShape.md)       | [MNIST 3D](datasets.md#mnist-3d)       | 16         | disabled        | enabled                  | Reconstruct the shape of objects from the _MNIST 3D_ dataset. |
| [TactileMNISTShapeStatic-v0](TactileMNISTShape.md) | [MNIST 3D](datasets.md#mnist-3d)       | 16         | disabled        | disabled                 | Same as TactileMNISTShape-v0 but the object pose stays fixed while it is being touched. |
| [ABCShape-v0](ABCShape.md)                         | [ABC Dataset](datasets.md#abc-dataset) | 32         | enabled         | enabled                  | Reconstruct the shape of objects from the _ABC_ dataset.    |
| [ABCShapeStatic-v0](ABCShape.md)                   | [ABC Dataset](datasets.md#abc-dataset) | 32         | enabled         | disabled                 | Same as ABCShape-v0 but the object pose stays fixed while it is being touched. |
