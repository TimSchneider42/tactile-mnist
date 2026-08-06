#!/usr/bin/env python3
"""
Oracle demo for the shape reconstruction environments: instead of learning from touch,
the "agent" encodes the ground-truth posed mesh with the environment's COD-VAE model
and always predicts the resulting full latent. This is the best prediction the
representation admits, so the reported loss shows the noise floor of the task, and
with the shadow objects enabled the rendering shows the reference reconstruction.
"""
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

import ap_gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="TactileMNISTShape-v0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    env = ap_gym.make(
        f"tactile_mnist:{args.env}", renderer_show_shadow_objects=True
    )
    inner_env = env.unwrapped
    obs, _ = env.reset(seed=0)

    latent_cache: dict[tuple, np.ndarray] = {}

    def oracle_prediction() -> np.ndarray:
        # The prediction target identifies the ground-truth geometry (mesh index and
        # pose); the oracle encodes exactly that geometry into a full latent. Since
        # the object only moves slightly between steps (pose perturbation), latents
        # are cached per (object, pose).
        dp = inner_env.current_data_points[0]
        pose = inner_env.current_object_poses_platform_frame[0]
        key = (dp.id, np.asarray(pose.matrix).tobytes())
        if key not in latent_cache:
            mesh = dp.mesh.copy()
            mesh.apply_transform(pose.matrix)
            latent_cache[key] = inner_env.vae.encode_mesh_full(
                mesh, seed=0, frame_half_size=inner_env.frame_half_size
            ).astype(np.float32)
        return latent_cache[key]

    fig, ax = plt.subplots(1, 2)
    camera_plot = ax[0].imshow(env.render())
    img_plot = ax[1].imshow(np.zeros(env.observation_space["sensor_img"].shape))
    plt.show(block=False)

    for _ in range(100):
        # Generate a circle trajectory
        angles = (
            np.arange(env.spec.max_episode_steps)
            / env.spec.max_episode_steps
            * 2
            * np.pi
        )
        target_trajectory = np.stack(
            [np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=-1
        )

        terminated = False
        for p in target_trajectory:
            action = {
                "action": {
                    **env.inner_action_space.sample(),
                    "sensor_target_pos_rel": p - obs["sensor_pos"],
                },
                "prediction": oracle_prediction(),
            }

            obs, _, terminated, truncated, info = env.step(action)
            logging.info(f"Oracle prediction loss: {info['prediction']['loss']:.4f}")
            camera_img = env.render()
            img_plot.set_data(obs["sensor_img"])
            camera_plot.set_data(camera_img)
            plt.pause(1 / env.metadata["render_fps"])
            if truncated:
                break
        assert terminated or truncated
        obs, _ = env.reset()
