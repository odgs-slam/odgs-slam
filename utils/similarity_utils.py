import torch
import torch.nn.functional as F
import math
# import kornia.feature as KF

def equirectangular_similarity(img1, img2, threshold=0.2, method='l1', R1=None, R2=None):
    """
    Compare two equirectangular images for visual similarity.
    
    Args:
        img1, img2: torch.Tensor, (3, H, W), values in [0, 1] or [0, 255]
        threshold: float, similarity distance threshold (lower = more similar)

    Returns:
        bool: True if images are visually similar (distance < threshold)
    """
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0

    if method == 'l1':
        loss = compute_pixelwise_loss(img1, img2, R1=R1, R2=R2, loss_type='l1').item()
        return loss < threshold # Lower = more similar
    elif method == 'l2':
        loss = compute_pixelwise_loss(img1, img2, R1=R1, R2=R2, loss_type='l2').item()
        return loss < threshold
    else:
        return False  # Unsupported method


def equirectangular_grid(h, w, device):
    # Longitude (theta): [-pi, pi], Latitude (phi): [-pi/2, pi/2]
    theta = torch.linspace(-math.pi, math.pi, w, device=device)  # horizontal (W)
    phi = torch.linspace(-math.pi / 2, math.pi / 2, h, device=device)  # vertical (H)
    phi, theta = torch.meshgrid(phi, theta, indexing='ij')  # shape: (H, W)

    # Convert spherical coordinates to Cartesian (x, y, z)
    x = torch.cos(phi) * torch.sin(theta)
    y = torch.sin(phi)
    z = torch.cos(phi) * torch.cos(theta)
    dirs = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
    return dirs

def rotate_dirs(dirs, R):
    # dirs: (h, w, 3), R: (3, 3)
    dirs_flat = dirs.view(-1, 3) @ R.T # (h * w, 3)
    return dirs_flat.view(*dirs.shape)

def dirs_to_equirectangular_coords(dirs, h, w):
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    theta = torch.atan2(x, z)  # longitude ∈ [-pi, pi]
    phi = torch.asin(y.clamp(-1, 1))  # latitude ∈ [-pi/2, pi/2]

    u = (theta / math.pi + 1) / 2  # [0, 1]
    v = (phi / (math.pi / 2) + 1) / 2  # [0, 1]

    # Convert to [-1, 1] for grid_sample
    u = u * 2 - 1
    v = v * 2 - 1
    grid = torch.stack([u, v], dim=-1)
    return grid

def warp_equirectangular(img1, R1, R2):
    C, H, W = img1.shape
    device = img1.device

    dirs = equirectangular_grid(H, W, device)  # (H, W, 3)
    dirs_rot = rotate_dirs(dirs, R1 @ R2.T)  # R1 * R2^-1
    grid = dirs_to_equirectangular_coords(dirs_rot, H, W)  # (H, W, 2)

    grid = grid.unsqueeze(0)  # (1, H, W, 2)
    img1 = img1.unsqueeze(0)  # (1, C, H, W)

    warped = F.grid_sample(img1, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped.squeeze(0)

def compute_pixelwise_loss(img1, img2, R1, R2, loss_type='l1'):
    warped_img1 = warp_equirectangular(img1, R1, R2)
    if loss_type == 'l1':
        loss = torch.abs(warped_img1 - img2)
        return loss.mean()
    elif loss_type == 'l2':
        return ((warped_img1 - img2) ** 2).mean()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")