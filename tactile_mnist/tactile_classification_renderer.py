from __future__ import annotations

import time
from contextlib import nullcontext
from functools import partial
from importlib.resources import files
from typing import Sequence, Iterable

import numpy as np
import trimesh
import trimesh.creation
from PIL import Image
from pyrender import (
    OffscreenRenderer,
    Node,
    Mesh,
    Scene,
    PerspectiveCamera,
    OrthographicCamera,
    RenderFlags,
    Viewer,
    DirectionalLight,
)
from scipy.spatial.transform import Rotation
from transformation import Transformation
from trimesh.primitives import Box
from trimesh.visual.material import PBRMaterial

from tactile_mnist import CELL_SIZE, MeshDataPoint
from .util import transformation_where


class MultiNode:
    def __init__(
        self,
        batch_size: int,
        *args,
        single_instance: bool = False,
        individual_args: bool = False,
        **kwargs,
    ):
        if single_instance:
            assert not individual_args
            node = Node(*args, **kwargs)
            self._nodes = (node,) * batch_size
        else:
            if individual_args:
                self._nodes = tuple(
                    Node(*(a[i] for a in args), **{k: v[i] for k, v in kwargs.items()})
                    for i in range(batch_size)
                )
            else:
                self._nodes = tuple(Node(*args, **kwargs) for _ in range(batch_size))

    def __getattr__(self, item):
        return tuple(getattr(n, item) for n in self._nodes)

    def __setattr__(self, key, value):
        if key == "_nodes":
            super().__setattr__(key, value)
        else:
            for n, v in zip(self._nodes, value):
                setattr(n, key, v)

    @property
    def nodes(self):
        return self._nodes


class BatchScene:
    def __init__(
        self, batch_size: int, nodes: Iterable[MultiNode] = (), *args, **kwargs
    ):
        assert batch_size > 0
        self._scenes = tuple(
            Scene([n.nodes[i] for n in nodes], *args, **kwargs)
            for i in range(batch_size)
        )
        if batch_size == 1:
            # See render method for explanation
            self._dummy_scene = Scene([Node(camera=PerspectiveCamera(yfov=np.pi / 4))])
        else:
            self._dummy_scene = None

    def render(
        self, renderer: OffscreenRenderer, flags=RenderFlags.NONE, seg_node_map=None
    ):
        if self._dummy_scene is not None:
            # This is a workaround for a bug, where the renderer does not consider the updates to mesh vertices
            # if it does not render another scene in between (probably some caching issue). In case where there is
            # only one scene, we render a dummy scene.
            renderer.render(self._dummy_scene, flags=RenderFlags.DEPTH_ONLY)
        res = [
            renderer.render(s, flags=flags, seg_node_map=seg_node_map)
            for s in self._scenes
        ]
        if isinstance(res[0], tuple):
            return tuple(np.stack(t, axis=0) for t in zip(*res))
        return np.stack(res, axis=0)

    def add_node(self, node: MultiNode):
        for s, n in zip(self._scenes, node.nodes):
            s.add_node(n)

    def remove_node(self, node: MultiNode):
        for s, n in zip(self._scenes, node.nodes):
            s.remove_node(n)

    def add(self, obj, name=None, pose=None, parent_node=None, parent_name=None):
        for s in self._scenes:
            s.add(
                obj,
                name=name,
                pose=pose,
                parent_node=parent_node,
                parent_name=parent_name,
            )

    def set_pose(self, node: MultiNode, pose: Transformation):
        if pose.single:
            pose = [pose] * len(node.nodes)
        for s, n, p in zip(self._scenes, node.nodes, pose):
            s.set_pose(n, p.matrix)


def image_to_world_scale(
    camera: PerspectiveCamera, image_dist: np.ndarray, dist: float
) -> np.ndarray:
    fy = 0.5 / np.tan(camera.yfov / 2)
    fx = fy / camera.aspectRatio
    return image_dist * dist / np.array([fx, fy])


def image_to_camera_frame(
    camera: PerspectiveCamera, pos: np.ndarray, dist: float
) -> np.ndarray:
    return image_to_world_scale(camera, pos - 0.5, dist)


def camera_frame_to_image(camera: PerspectiveCamera, pos: np.ndarray) -> np.ndarray:
    fy = 0.5 / np.tan(camera.yfov / 2)
    fx = fy / camera.aspectRatio
    return pos[..., :2] * np.array([fx, fy]) / -pos[..., 2:3] + 0.5


class TactileClassificationRenderer:
    def __init__(
        self,
        num_envs: int,
        depth_map_resolution: tuple[int, int],
        depth_map_pixmm: float,
        external_camera_resolution: tuple[int, int] = (640, 480),
        show_viewer: bool = False,
        show_sensor_target_pos: bool = False,
        object_color: tuple[float | int, ...] = (51, 0, 4),
        tactile_screen_zoom_color: tuple[float | int, ...] = (255, 100, 100, 160),
        platform_color: tuple[float | int, ...] = (0, 11, 51),
        show_tactile_image: bool = True,
        transparent_background: bool = False,
    ):
        self._transparent_background = transparent_background

        self.sensor_shadow_poses: Transformation = Transformation.batch_concatenate(
            [Transformation()] * num_envs
        )
        self.sensor_poses: Transformation = Transformation.batch_concatenate(
            [Transformation()] * num_envs
        )

        self._objects: tuple[MeshDataPoint] | None = None
        self._object_poses: Transformation = Transformation.batch_concatenate(
            [Transformation()] * num_envs
        )
        self._show_sensor_target_pos = show_sensor_target_pos
        self._num_envs = num_envs

        self._platform_pose = Transformation()
        platform_extents = np.concatenate([CELL_SIZE, [0.002]])
        platform_mesh = Box(
            platform_extents, Transformation([0, 0, -platform_extents[-1] / 2]).matrix
        )
        platform_mesh.visual = trimesh.visual.TextureVisuals(
            material=PBRMaterial(
                baseColorFactor=platform_color, metallicFactor=0.2, roughnessFactor=1.0
            )
        )

        # Variables needed for depth map rendering
        self._sensor_object_node = None

        # Set the camera really far away from the gel. That way we can use it to find the first contact point of the
        # sensor when it is approaching the target.
        self._camera_dist_to_gel = np.max(platform_extents) * 2
        self._camera_pose_sensor_frame = Transformation.from_pos_euler(
            [0, 0, self._camera_dist_to_gel]
        )

        self._sensor_renderer = OffscreenRenderer(*depth_map_resolution)

        m_per_px = depth_map_pixmm / 1000
        mag = np.array(depth_map_resolution) / 2 * m_per_px
        self._sensor_surface_size = np.array(depth_map_resolution) * m_per_px

        Node = partial(MultiNode, batch_size=num_envs)
        self._sensor_camera = OrthographicCamera(
            xmag=mag[0], ymag=mag[1], znear=0.001, zfar=2 * self._camera_dist_to_gel
        )
        self._sensor_camera_node = Node(
            camera=self._sensor_camera, matrix=Transformation().matrix
        )

        self._sensor_scene = BatchScene(
            num_envs,
            [
                Node(
                    mesh=Mesh.from_trimesh(platform_mesh),
                    matrix=self._platform_pose.matrix,
                    single_instance=True,
                ),
                self._sensor_camera_node,
            ],
        )

        render_camera_target = self._platform_pose.translation + np.array(
            [-0.02, -0.02, 0.0]
        )
        render_camera_pos = np.array([-0.1, -0.1, 0.1])
        render_camera_z = render_camera_pos - render_camera_target
        render_camera_z /= np.linalg.norm(render_camera_z)
        render_camera_x = np.cross(render_camera_z, np.array([0.0, 0.0, 1.0]))
        render_camera_x /= np.linalg.norm(render_camera_x)
        render_camera_y = np.cross(render_camera_z, render_camera_x)
        if render_camera_y[2] < 0:
            render_camera_y = -render_camera_y
            render_camera_x = -render_camera_x
        render_camera_rot = Rotation.from_matrix(
            np.stack([render_camera_x, render_camera_y, render_camera_z], axis=-1)
        )

        self._render_camera_pose = Transformation(render_camera_pos, render_camera_rot)

        if show_tactile_image:
            self._render_camera_pose = self._render_camera_pose * Transformation(
                [0.02, 0.02, 0.04]
            )

        self._render_camera = PerspectiveCamera(
            yfov=np.pi / 4,
            aspectRatio=external_camera_resolution[0] / external_camera_resolution[1],
            znear=0.005,
        )
        render_camera_node = Node(
            camera=self._render_camera,
            matrix=self._render_camera_pose.matrix,
            single_instance=True,
        )

        sensor_scene = trimesh.load(
            files("tactile_mnist.resources").joinpath("gelsight_mini.obj")
        )
        sensor_meshes = [m for m in sensor_scene.geometry.values()]
        for g in sensor_meshes:
            g.visual.uv = None
        self._sensor_mesh = trimesh.util.concatenate(sensor_meshes)
        self._sensor_mesh.apply_transform(
            (
                Transformation.from_pos_euler(euler_angles=[0, 0, np.pi / 2])
                * Transformation.from_pos_euler(euler_angles=[np.pi, 0, 0])
            ).matrix
        )

        self._sensor_node = Node(mesh=Mesh.from_trimesh(self._sensor_mesh))
        self._sensor_node.matrix = self.sensor_poses.matrix
        sensor_mesh_transparent = self._sensor_mesh.copy()
        mesh_colors = np.array(sensor_mesh_transparent.visual.material.image)
        mesh_colors[..., 3] = 128
        sensor_mesh_transparent.visual.material.image = Image.fromarray(mesh_colors)
        self._transparent_sensor_node = Node(
            mesh=Mesh.from_trimesh(sensor_mesh_transparent)
        )
        self._transparent_sensor_node.matrix = self.sensor_shadow_poses.matrix

        self._camera_object_node: Node | None = None
        platform_node = Node(
            mesh=Mesh.from_trimesh(platform_mesh),
            matrix=self._platform_pose.matrix,
            single_instance=True,
        )
        self._camera_scene = BatchScene(
            num_envs,
            [platform_node, self._sensor_node, render_camera_node],
            ambient_light=np.array([0.4, 0.4, 0.4, 1.0]),
            bg_color=np.array([1.0, 1.0, 1.0, 0.0]),
        )

        tactile_screen_width_rel = 0.3
        sensor_width_by_height = depth_map_resolution[0] / depth_map_resolution[1]
        tactile_screen_height_rel = tactile_screen_width_rel / (
            sensor_width_by_height / self._render_camera.aspectRatio
        )
        self._tactile_screen_size_rel = np.array(
            [tactile_screen_width_rel, tactile_screen_height_rel]
        )
        self._tactile_screen_pos_rel = (
            np.array([0.98, 0.98]) - self._tactile_screen_size_rel / 2
        )
        self._show_tactile_image = show_tactile_image

        screen_dist = 0.01

        screen_pos_camera_frame = np.concatenate(
            [
                image_to_camera_frame(
                    self._render_camera, self._tactile_screen_pos_rel, screen_dist
                ),
                [-screen_dist],
            ]
        )
        real_world_screen_size = image_to_world_scale(
            self._render_camera, self._tactile_screen_size_rel, screen_dist
        )

        screen_rot_camera_frame = Rotation.from_euler("xyz", [0, 0, 0])
        screen_pose = self._render_camera_pose * Transformation(
            screen_pos_camera_frame, screen_rot_camera_frame
        )
        screen_corners_screen_frame = np.array(
            [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
        ) * np.concatenate([real_world_screen_size / 2, [0]])
        self._screen_corners = screen_pose * screen_corners_screen_frame
        self._projected_screen_corners = camera_frame_to_image(
            self._render_camera, self._render_camera_pose.inv * self._screen_corners
        )
        self._tactile_zoom_origin_sensor_frame = np.array(
            [0, 0, self._sensor_mesh.vertices[:, 2].max()]
        )

        if show_tactile_image:
            default_vertices = (
                np.array(
                    [
                        [1, 0, 0],
                        [1, 1, 0],
                        [0, 1, 0],
                    ]
                )
                * 0.001
            )
            plane = trimesh.Trimesh(
                vertices=default_vertices, faces=np.array([[0, 1, 2]])
            )
            plane.visual = trimesh.visual.TextureVisuals(
                material=PBRMaterial(baseColorFactor=tactile_screen_zoom_color)
            )
            self._tactile_screen_zoom = Node(
                mesh=[Mesh.from_trimesh(plane) for _ in range(self._num_envs)],
                individual_args=True,
            )
            self._camera_scene.add_node(self._tactile_screen_zoom)
        else:
            self._tactile_screen_zoom = None

        if self._show_sensor_target_pos:
            self._camera_scene.add_node(self._transparent_sensor_node)

        self._camera_scene.add(
            DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0),
            pose=self._render_camera_pose.matrix,
        )

        if show_viewer:
            self._viewer = Viewer(self._camera_scene, run_in_thread=True)
            self._camera_renderer: OffscreenRenderer | None = None
            # For some reason it is necessary to wait here
            time.sleep(0.5)
        else:
            # Variables needed for camera rendering
            self._camera_renderer = OffscreenRenderer(*external_camera_resolution)
            self._viewer: Viewer | None = None
        self._object_color = object_color

    def render_external_cameras(
        self, tactile_img: np.ndarray | None = None
    ) -> np.ndarray | None:
        if self._camera_renderer is None:
            return None
        show_tactile_img = tactile_img is not None and self._show_tactile_image
        with self._get_render_lock():
            self._camera_scene.set_pose(self._sensor_node, self.sensor_poses)
            if self._show_sensor_target_pos:
                self._camera_scene.set_pose(
                    self._transparent_sensor_node, self.sensor_shadow_poses
                )
            if show_tactile_img:
                tactile_zoom_origins = (
                    self.sensor_poses * self._tactile_zoom_origin_sensor_frame
                )
                tactile_zoom_origins_camera_frame = (
                    self._render_camera_pose.inv * tactile_zoom_origins
                )
                projected_tactile_zoom_origins = camera_frame_to_image(
                    self._render_camera, tactile_zoom_origins_camera_frame
                )
                projection_diffs = (
                    self._projected_screen_corners[None]
                    - projected_tactile_zoom_origins[:, None]
                )
                angles = np.arctan2(
                    projection_diffs[:, :, 1], projection_diffs[:, :, 0]
                )
                corner_1_idx = np.argmin(angles, axis=1)
                corner_2_idx = np.argmax(angles, axis=1)
                for mesh, c1, c2, o in zip(
                    self._tactile_screen_zoom.mesh,
                    corner_1_idx,
                    corner_2_idx,
                    tactile_zoom_origins,
                ):
                    mesh.primitives[0].positions = np.stack(
                        [self._screen_corners[c1], self._screen_corners[c2], o]
                    )
            img = self._camera_scene.render(
                self._camera_renderer, flags=RenderFlags.RGBA
            )[0]
        if show_tactile_img:
            img_size = np.flip(np.array(img.shape[1:3]))
            target_pos_rel = np.array(
                [
                    self._tactile_screen_pos_rel[0],
                    1.0 - self._tactile_screen_pos_rel[1],
                ]
            )
            t_size = np.round(self._tactile_screen_size_rel * img_size).astype(np.int_)
            t_pos = np.round(target_pos_rel * img_size - t_size / 2).astype(np.int_)
            for i, t_img in enumerate(tactile_img):
                tactile_img_scaled = np.array(
                    Image.fromarray((t_img * 255).astype(np.uint8))
                    .resize(t_size)
                    .convert("RGBA")
                )
                img[
                    i,
                    t_pos[1] : t_pos[1] + t_size[1],
                    t_pos[0] : t_pos[0] + t_size[0],
                ] = tactile_img_scaled

        if not self._transparent_background:
            alpha = img[..., 3:4] / 255
            img = (img[..., :3] * alpha + (1 - alpha) * 255).astype(np.uint8)

        return img

    def render_sensor_depths(
        self, virtual_sensor_poses: Transformation | None = None
    ) -> np.ndarray:
        if virtual_sensor_poses is None:
            virtual_sensor_poses = self.sensor_poses
        sensor_camera_pose = (
            self._platform_pose * virtual_sensor_poses * self._camera_pose_sensor_frame
        )
        with self._get_render_lock():
            self._sensor_scene.set_pose(self._sensor_camera_node, sensor_camera_pose)
            depth_orig = self._sensor_scene.render(
                self._sensor_renderer, flags=RenderFlags.DEPTH_ONLY
            )
        depth = self._recover_depth_workaround(depth_orig)
        depth_clipped = np.clip(depth, 0, self._sensor_camera.zfar)
        return depth_clipped - self._camera_dist_to_gel

    def set_object_poses(
        self, new_object_poses: Transformation, mask: Sequence[bool] | None = None
    ):
        if mask is None:
            mask = np.ones(self._num_envs, dtype=np.bool_)
        self._object_poses = transformation_where(
            mask, new_object_poses, self._object_poses
        )
        object_poses_world = self._platform_pose * self._object_poses
        with self._get_render_lock():
            self._camera_scene.set_pose(self._camera_object_node, object_poses_world)
            self._sensor_scene.set_pose(self._sensor_object_node, object_poses_world)

    def close(self):
        if self._viewer is not None:
            with self._get_render_lock():
                self._viewer.close()

    def _get_render_lock(self):
        return nullcontext() if self._viewer is None else self._viewer.render_lock

    def _process_object_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        mesh = mesh.copy()
        mesh.visual = trimesh.visual.TextureVisuals(
            material=PBRMaterial(
                baseColorFactor=self._object_color,
                metallicFactor=0.1,
                roughnessFactor=0.7,
            )
        )
        return mesh

    def _recover_depth_workaround(self, depth_orig: np.ndarray) -> np.ndarray:
        """
        Workaround to fix broken depth recovery (https://github.com/mmatl/pyrender/issues/254)
        :param depth_orig: Original depth image as generated by pyrenderer.
        :return: Fixed depth image.
        """
        f = self._sensor_camera.zfar
        n = self._sensor_camera.znear
        non_zero = depth_orig != 0
        depth_raw_non_zero = (f + n) / (f - n) - 2 * n * f / (
            (f - n) * depth_orig[non_zero]
        )
        depth = np.full_like(depth_orig, np.inf)
        depth[non_zero] = (depth_raw_non_zero * (f - n) + f + n) / 2.0
        return depth

    @property
    def objects(self) -> tuple[MeshDataPoint, ...]:
        return self._objects

    @objects.setter
    def objects(self, objects: Iterable[MeshDataPoint]):
        objects = tuple(objects)
        assert len(objects) == self._num_envs
        self._objects = objects
        current_meshes = [dp.mesh.copy() for dp in self._objects]
        for mesh in current_meshes:
            mesh.visual.vertex_colors = [50, 50, 50]
        with self._get_render_lock():
            if self._camera_object_node is not None:
                self._camera_scene.remove_node(self._camera_object_node)
            self._camera_object_node = MultiNode(
                self._num_envs,
                mesh=[
                    Mesh.from_trimesh(self._process_object_mesh(mesh))
                    for mesh in current_meshes
                ],
                individual_args=True,
            )
            self._camera_scene.add_node(self._camera_object_node)
            if self._sensor_object_node is not None:
                self._sensor_scene.remove_node(self._sensor_object_node)
            self._sensor_object_node = MultiNode(
                self._num_envs,
                mesh=[Mesh.from_trimesh(mesh) for mesh in current_meshes],
                individual_args=True,
            )
            self._sensor_scene.add_node(self._sensor_object_node)
