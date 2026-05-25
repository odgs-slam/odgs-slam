import numpy as np
import torch

# from utils.camera_utils import Camera


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    conv = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    img_grad = torch.nn.functional.conv2d(
        p_img.float(), conv.repeat(c, 1, 1, 1), groups=c
    )
    mask = img_grad[0] == torch.sum(conv)
    return mask


# def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
#     mask_v, mask_h = image_gradient_mask(depth)
#     gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
#     depth_grad_v, depth_grad_h = image_gradient(depth)
#     gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
#     depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

#     w_h = torch.exp(-10 * gray_grad_h**2)
#     w_v = torch.exp(-10 * gray_grad_v**2)
#     err = (w_h * torch.abs(depth_grad_h)).mean() + (
#         w_v * torch.abs(depth_grad_v)
#     ).mean()
#     return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)

def latitude_weight(height, gamma=2.0):
    y = torch.arange(height, dtype=torch.float32, device="cuda")
    latitude = (y / height - 0.5) * np.pi
    weight = torch.cos(latitude).clamp(min=1e-6) ** gamma
    return weight.unsqueeze(0).unsqueeze(-1).expand(1, height, -1)

def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"].get("rgb_boundary_threshold", 0.01)

    # filter out pixels that are close to black (small intensity) in the gt_image
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)

    weights = latitude_weight(h)
    pixel_weights = rgb_pixel_mask * viewpoint.grad_mask * opacity * weights

    l1 = pixel_weights * torch.abs(image - gt_image)

    return l1.sum() / pixel_weights.sum().clamp(min=1e-8)


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"].get("alpha", 0.95)

    gt_image = viewpoint.original_image.cuda()
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_boundary_threshold = config["Training"].get("rgb_boundary_threshold", 0.01)
    depth_boundary_threshold = config["Training"].get("depth_boundary_threshold", 0.01)
    opacity_boundary_threshold = config["Training"].get("opacity_boundary_threshold", 0.95)

    # filter out pixels that are close to black (small intensity) in the gt_image
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    # TODO - make dependent on depth scale, might even not be required due to check in rasterizer (auxiliary.h:152)
    depth_pixel_mask = (gt_depth > depth_boundary_threshold).view(*depth.shape)
    opacity_mask = (opacity > opacity_boundary_threshold).view(*depth.shape)

    _, h, _ = image.shape
    weights = latitude_weight(h)
    # use opacity to weight the rgb loss (as in monogsslam paper)
    pixel_weights = rgb_pixel_mask * viewpoint.grad_mask * opacity * weights
    # use opacity mask to weight the depth loss (as in monogsslam paper)
    depth_weights = depth_pixel_mask * opacity_mask * weights

    l1_rgb = pixel_weights * torch.abs(image - gt_image)
    l1_depth = depth_weights * torch.abs(depth - gt_depth)

    l1_rgb_mean = l1_rgb.sum() / pixel_weights.sum().clamp(min=1e-8)
    l1_depth_mean = l1_depth.sum() / depth_weights.sum().clamp(min=1e-8)
    return alpha * l1_rgb_mean + (1 - alpha) * l1_depth_mean


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"].get("rgb_boundary_threshold", 0.01)

    weights = latitude_weight(h)
    # filter out pixels that are close to black (small intensity) in the gt_image
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    pixel_weights = rgb_pixel_mask * weights

    l1_rgb = pixel_weights * torch.abs(image - gt_image)
    return l1_rgb.sum() / pixel_weights.sum().clamp(min=1e-8)


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"].get("alpha", 0.95)
    rgb_boundary_threshold = config["Training"].get("rgb_boundary_threshold", 0.01)
    depth_boundary_threshold = config["Training"].get("depth_boundary_threshold", 0.01)

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > depth_boundary_threshold).view(*depth.shape)

    _, h, _ = image.shape
    weights = latitude_weight(h)
    pixel_weights = rgb_pixel_mask * weights
    depth_weights = depth_pixel_mask * weights
    l1_rgb = pixel_weights * torch.abs(image - gt_image)
    l1_depth = depth_weights * torch.abs(depth - gt_depth)

    l1_rgb_mean = l1_rgb.sum() / pixel_weights.sum().clamp(min=1e-8)
    l1_depth_mean = l1_depth.sum() / depth_weights.sum().clamp(min=1e-8)
    return alpha * l1_rgb_mean + (1 - alpha) * l1_depth_mean


def get_median_depth(config, depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    valid = depth > 0
    if opacity is not None:
        opacity = opacity.detach()
        opacity_boundary_threshold = config["Training"].get("opacity_boundary_threshold", 0.95)
        valid = torch.logical_and(valid, opacity > opacity_boundary_threshold)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
