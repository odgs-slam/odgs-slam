import json

import numpy as np

from utils.logging_utils import Log


def load_trajectories(file_path: str) -> tuple[list, list, list]:
    """Load trajectory data from ODGS-SLAM/MonoGS JSON format."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        trj_id = data.get("trj_id", [])
        trj_est = data.get("trj_est", [])
        trj_gt = data.get("trj_gt", [])

        return trj_id, trj_est, trj_gt
    except Exception as e:
        Log(f"Error loading {file_path}: {e}", tag="Error")
        return [], [], []


def convert_matrix_to_pose(matrix: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert 4x4 transformation matrix to position and quaternion."""
    # Extract position (translation)
    T = np.array([matrix[0][3], matrix[1][3], matrix[2][3]])

    # Extract rotation matrix
    R = np.array(
        [
            [matrix[0][0], matrix[0][1], matrix[0][2]],
            [matrix[1][0], matrix[1][1], matrix[1][2]],
            [matrix[2][0], matrix[2][1], matrix[2][2]],
        ]
    )
    return T, R


def get_poses_from_trajectory(
    trajectories: list,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract poses from trajectory estimates."""
    poses = []
    for i, pose_mat in enumerate(trajectories):
        if pose_mat is None or len(pose_mat) != 4 or len(pose_mat[0]) != 4:
            Log(f"Invalid pose matrix at index {i}: {pose_mat}", tag="Warn")
            continue
        pose_mat = np.linalg.inv(np.array(pose_mat))
        T, R = convert_matrix_to_pose(pose_mat)
        poses.append((T, R))
    return poses
