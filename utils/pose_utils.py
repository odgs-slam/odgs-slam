import numpy as np
import torch

from gaussian_splatting.utils.graphics_utils import getWorld2View2
from utils.camera_utils import Camera

def quaternion_to_matrix(q):
    """Convert a quaternion to a 3x3 rotation matrix in PyTorch (GPU-compatible)."""
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    qw, qx, qy, qz = q.unbind(dim=-1)

    R = torch.stack([
        torch.stack([1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)], dim=-1),
        torch.stack([2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)], dim=-1),
        torch.stack([2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)], dim=-1)
    ], dim=-2)

    return R


def update_pose(camera: Camera, converged_threshold=1e-5):
    R_delta = quaternion_to_matrix(camera.cam_q_w2c.clone())
    T_delta = camera.cam_t_w2c.clone()

    W2C = getWorld2View2(camera.R, camera.T)
    delta_Rt = getWorld2View2(R_delta, T_delta)

    new_Rt = delta_Rt @ W2C

    R_new = new_Rt[:3, :3]
    T_new = new_Rt[:3, 3]
    
    trans_diff = torch.norm(T_delta, p=2)
    rot_diff = torch.acos(torch.clamp((torch.trace(R_delta) - 1) / 2.0, -1.0, 1.0))
    converged = (trans_diff < converged_threshold) and (rot_diff < converged_threshold)

    camera.update_RT(R_new, T_new)
    camera.reset_qt()
    return converged
