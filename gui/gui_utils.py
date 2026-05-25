import queue
from multiprocessing.queues import Queue

import cv2
import numpy as np
import open3d as o3d
import torch
from munch import Munch

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.general_utils import (
    build_scaling_rotation,
    strip_symmetric,
)
from utils.camera_utils import Camera

cv_gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


class Frustum:
    def __init__(self, line_set, view_dir=None, view_dir_behind=None, size=None):
        self.line_set = line_set
        self.view_dir = view_dir
        self.view_dir_behind = view_dir_behind
        self.size = size
        self.pose = None

    def update_pose(self, pose):
        self.pose = pose
        
        points = np.asarray(self.line_set.points)
        points_hmg = np.hstack([points, np.ones((points.shape[0], 1))])
        points = (pose @ points_hmg.transpose())[0:3, :].transpose()

        base = np.array([[0.0, 0.0, 0.0]]) * self.size
        base_hmg = np.hstack([base, np.ones((base.shape[0], 1))])
        cameraeye = pose @ base_hmg.transpose()
        cameraeye = cameraeye[0:3, :].transpose()
        eye = cameraeye[0, :]

        base_behind = np.array([[0.0, -2.5, -30.0]]) * self.size
        base_behind_hmg = np.hstack([base_behind, np.ones((base_behind.shape[0], 1))])
        cameraeye_behind = pose @ base_behind_hmg.transpose()
        cameraeye_behind = cameraeye_behind[0:3, :].transpose()
        eye_behind = cameraeye_behind[0, :]

        center = np.mean(points[1:, :], axis=0)
        up = points[2] - points[4]

        self.view_dir = (center, eye, up, pose)
        self.view_dir_behind = (center, eye_behind, up, pose)

        self.center = center
        self.eye = eye
        self.up = up


def create_frustum(pose, frusutum_color=[0, 1, 0], size=0.02):
    points = (
        np.array(
            [
                [0.0, 0.0, 0],
                [1.0, -0.5, 2],
                [-1.0, -0.5, 2],
                [1.0, 0.5, 2],
                [-1.0, 0.5, 2],
            ]
        )
        * size
    )

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    colors = [frusutum_color for i in range(len(lines))]

    canonical_line_set = o3d.geometry.LineSet()
    canonical_line_set.points = o3d.utility.Vector3dVector(points)
    canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
    canonical_line_set.colors = o3d.utility.Vector3dVector(colors)
    frustum = Frustum(canonical_line_set, size=size)
    frustum.update_pose(pose)
    return frustum


def create_spherical_frustum(pose, frustum_color=[0, 1, 0], radius=1.0, num_points=100):
    # Create a sphere
    phi = np.linspace(0, 2 * np.pi, num_points)
    theta = np.linspace(0, np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

    # Apply the pose transformation
    points_hmg = np.hstack([points, np.ones((points.shape[0], 1))])
    points = (pose @ points_hmg.transpose())[0:3, :].transpose()

    # Create lines connecting the points (optional, for visualization)
    lines = []
    for i in range(num_points - 1):
        for j in range(num_points - 1):
            lines.append([i * num_points + j, i * num_points + (j + 1)])
            lines.append([i * num_points + j, (i + 1) * num_points + j])

    colors = [frustum_color for _ in range(len(lines))]

    # Create Open3D LineSet for visualization
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    frustum = Frustum(line_set, size=radius)
    frustum.update_pose(pose)
    return frustum

class GaussianRenderModel:
    def __init__(self, gaussians: GaussianModel):
        self.get_xyz = gaussians.get_xyz.detach().clone()
        self.active_sh_degree = gaussians.active_sh_degree
        self.get_opacity = gaussians.get_opacity.detach().clone()
        self.get_scaling = gaussians.get_scaling.detach().clone()
        self.get_rotation = gaussians.get_rotation.detach().clone()
        self.max_sh_degree = gaussians.max_sh_degree
        self.get_features = gaussians.get_features.detach().clone()

        self._rotation = gaussians._rotation.detach().clone()
        self.rotation_activation = torch.nn.functional.normalize
        self.unique_kfIDs = gaussians.unique_kfIDs.clone()
        self.n_obs = gaussians.n_obs.clone()

    def get_covariance(self, scaling_modifier=1.0):
        return self.build_covariance_from_scaling_rotation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm
    

class GaussianPacket:
    def __init__(
        self,
        gaussians: GaussianModel | None = None,
        current_frame: Camera | None = None,
        gtcolor: torch.Tensor | None = None,
        gtdepth: np.ndarray | None = None,
        keyframes: list[Camera] | None = None,
        finish: bool = False,
        kf_window: dict[int, list[int]] | None = None,
        active_kf_ids: list[int] | None = None,
    ):
        self.gaussians = None
        if gaussians is not None:
            self.gaussians = GaussianRenderModel(gaussians)

        self.current_frame = current_frame
        self.gtcolor = self.resize_img(gtcolor, 320)
        self.gtdepth = self.resize_img(gtdepth, 320)
        self.keyframes = keyframes
        self.finish = finish
        self.kf_window = kf_window
        self.active_kf_ids = active_kf_ids

    def resize_img(self, img, width):
        if img is None:
            return None

        # check if img is numpy
        if isinstance(img, np.ndarray):
            height = int(width * img.shape[0] / img.shape[1])
            return cv2.resize(img, (width, height))
        height = int(width * img.shape[1] / img.shape[2])
        # img is 3xHxW
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        )
        return img.squeeze(0)



def get_latest_queue(q) -> GaussianPacket | None:
    message = None
    while True:
        try:
            message_latest = q.get_nowait()
            if message is not None:
                del message
            message = message_latest
        except queue.Empty:
            if q.qsize() < 1:
                break
    return message

def clear_queue(queue_to_clear):
    # clean up the pipe
    if queue_to_clear is not None:
        while not queue_to_clear.empty():
            queue_to_clear.get()


class Packet_vis2main:
    def __init__(self, flag_gui_exit=None, flag_pause=None, flag_gui_active=None):
        self.flag_gui_exit = flag_gui_exit
        self.flag_pause = flag_pause
        self.flag_gui_active = flag_gui_active


class ParamsGUI:
    def __init__(
        self,
        pipe: Munch,
        background: torch.Tensor,
        # gaussians: GaussianModel | None,
        q_main2vis: Queue,
        q_vis2main: Queue,
        exit_gui_on_finish: bool = True,
        output_dir: str | None = None,
    ):
        self.pipe = pipe
        self.background = background
        # self.gaussians = GaussianPacket(gaussians=gaussians)
        # self.gaussians = gaussians
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main
        self.exit_gui_on_finish = exit_gui_on_finish
        self.output_dir = output_dir if output_dir is not None else '.'
