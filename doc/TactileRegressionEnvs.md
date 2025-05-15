# Tactile Regression Environments

In tactile regression environments, the agent has to infer a continuous property from a 3D object by exploring it with a [GelSight Mini](https://www.gelsight.com/gelsightmini/) tactile sensor.
The agent does not have access to the object's location or orientation and also receives no visual input.
Instead, it must actively control the sensor to find and explore the object.

For more details on tactile perception environments in general, see the [Tactile Perception Environments documentation](TactilePerceptionEnv.md).

Currently implemented are the following two tasks, which are described in more detail in their respective documentations:

<div align="center">
    <table style="border-collapse: collapse; border: none;">
        <tr style="border: none;">
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
</div>

All tactile regression environments share the following properties:

## Properties

<table>
    <tr>
        <td><strong>Prediction Space</strong></td>
        <td><code>Box(-inf, inf, shape=(N,), dtype=np.float32)</code></td>
    </tr>
    <tr>
        <td><strong>Prediction Target Space</strong></td>
        <td><code>Box(-inf, inf, shape=(N,), dtype=np.float32)</code></td>
    </tr>
    <tr>
        <td><strong>Loss Function</strong></td>
        <td>
            <code>ap_gym.MSELossFn()</code>
        </td>
    </tr>
</table>


where $N \in \mathbb{N}$ is the number of dimensions of the predicted value.

## Prediction Space

The prediction is an $N$-element `np.ndarray` containing the current prediction of the agent.
The agent's objective is to approximate the prediction target as closely as possible.

## Overview of Implemented Environments

| Environment ID                                 | Dataset                          | N | Step Limit | Sensor Rotation | Object Pose Perturbation | Description                                                 |
|------------------------------------------------|----------------------------------|---|------------|-----------------|--------------------------|-------------------------------------------------------------|
| [Toolbox-v0](Toolbox.md)                       |                                  | 4 | 64         | disabled        | enabled                  | Estimate the pose of a tool.                                |
| [TactileMNISTVolume-v0](TactileMNISTVolume.md) | [MNIST 3D](datasets.md#mnist-3d) | 1 | 32         | disabled        | enabled                  | Estimate the volume of objects from the _MNIST 3D_ dataset. |
