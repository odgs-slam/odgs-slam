
import glob
import json
import os
import gc

import cv2
import numpy as np
import torch
from PIL import Image

from scipy.spatial.transform import Rotation as R

from gaussian_splatting.utils.graphics_utils import focal2fov

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False


class PanoramaParser:
    def __init__(self, input_folder, camera=None):
        self.input_folder = input_folder
        self.camera = camera
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/*.png"))
        self.depth_paths = sorted(
            glob.glob(f"{self.input_folder}/depth/*.exr"))
        self.n_img = len(self.color_paths)
        self.transforms = self.load_transform()
    
    def load_transform(self):
        json_path = f"{self.input_folder}/positions/PanoramaCam_positions.json"
        if not os.path.exists(json_path):
            return None
        with open(json_path, 'r') as file:
            data = json.load(file)

        transforms = []
        for i, entry in enumerate(data):
            position = entry["position"]
            euler_angles = entry["rotation"]
            if self.camera is not None:
                # Flip x and negate y for coordinate system adjustment
                # Convert position from mm to meters
                if self.camera == "X4 Stick":
                    # Subtract pi/2 from z (camera mounted sideways)
                    euler_angles = np.array([euler_angles['x'] + np.pi, euler_angles['z'] - np.pi/2, -euler_angles['y']])
                    rot_matrix = R.from_euler('xyz', euler_angles, degrees=False).as_matrix()
                elif self.camera == "X4 Low":
                    # Add pi/2 from z (camera mounted other sideways)
                    euler_angles = np.array([euler_angles['x'] + np.pi, euler_angles['z'] + np.pi/2, -euler_angles['y']])
                    rot_matrix = R.from_euler('xyz', euler_angles, degrees=False).as_matrix()
                elif self.camera == "Pro":
                    # Add 2.5pi/4 from z (camdera mounted at an angle)
                    euler_angles = np.array([euler_angles['x'] + np.pi, euler_angles['z'] - 2.5 * np.pi/4, -euler_angles['y']])
                    rot_matrix = R.from_euler('xyz', euler_angles, degrees=False).as_matrix()
                else:
                    raise ValueError("Unknown camera type")
                
                pose = np.eye(4)
                pose[:3, :3] = rot_matrix
                pose[:3, 3] = [position['x'] / 1000, position['y'] / 1000, position['z'] / 1000]

                transforms.append(np.linalg.inv(pose))
            else:
                # Convert Blender's Z-up, -Y forward system to SLAM's Z-forward frame.  
                # Adjust signs, swap Y/Z, and use 'xzy' order for correct rotation. 
                euler_angles = np.array([-euler_angles['x'], euler_angles['z'], -euler_angles['y']])
                rot_matrix = R.from_euler('xzy', euler_angles, degrees=False).as_matrix()

                pose = np.eye(4)
                pose[:3, :3] = rot_matrix
                pose[:3, 3] = [position['x'], position['y'], position['z']]

                # Extrinsics from camera to world, inverse of camera pose
                transforms.append(np.linalg.inv(pose))
        return transforms


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass


class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image = np.array(Image.open(color_path))
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path)) / self.depth_scale

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, pose



class PanoramaDataset(MonocularDataset):
    def __init__(self, args, path, config, max_num_frames=None, start_frame=0):
        super().__init__(args, path, config)
        self.dataset_path = config["Dataset"]["dataset_path"]
        camera = config["Dataset"]["camera"] if "camera" in config["Dataset"].keys() else None
        parser = PanoramaParser(self.dataset_path, camera)
        
        # Apply start_frame and max_num_frames
        total_frames = parser.n_img
        start_idx = min(start_frame, total_frames - 1) if start_frame < total_frames else 0
        
        if max_num_frames is not None:
            end_idx = min(start_idx + max_num_frames, total_frames)
        else:
            end_idx = total_frames
            
        self.num_imgs = end_idx - start_idx
        self.color_paths = parser.color_paths[start_idx:end_idx]
        self.depth_paths = parser.depth_paths[start_idx:end_idx]
        self.transforms = parser.transforms[start_idx:end_idx] if parser.transforms else None

        self.inv_depth = config["Dataset"]["inv_depth"] if "inv_depth" in config["Dataset"].keys() else False
        self.image_downsample = config["Dataset"]["image_downsample"] if "image_downsample" in config["Dataset"].keys() else 1
        self.sensor_type = config["Dataset"]["sensor_type"]

        self.width = config["Dataset"]["Calibration"]["width"] // self.image_downsample
        self.height = config["Dataset"]["Calibration"]["height"] // self.image_downsample
        self.fx = config["Dataset"]["Calibration"]["fx"]
        self.fy = config["Dataset"]["Calibration"]["fy"]
        self.cx = config["Dataset"]["Calibration"]["cx"] // self.image_downsample
        self.cy = config["Dataset"]["Calibration"]["cy"] // self.image_downsample

        torch.backends.cudnn.benchmark = True
    
    def __loadExrDepth(self, file_path):
        if not HAS_OPENEXR:
            # Fallback: use OpenCV if OpenEXR is not available
            depth = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise RuntimeError(f"Failed to load EXR file: {file_path}")
            if depth.ndim == 3:
                depth = depth[:, :, 0]
        else:
            exr_file = OpenEXR.InputFile(file_path)
            dw = exr_file.header()['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            # Read only the Z (depth) channel
            if 'Z' in exr_file.header()['channels']:
                depth_str = exr_file.channel('Z', pt)
                depth = np.frombuffer(depth_str, dtype=np.float32).reshape((height, width))
            else:
                # Fallback: read first channel
                channel_name = list(exr_file.header()['channels'].keys())[0]
                depth_str = exr_file.channel(channel_name, pt)
                depth = np.frombuffer(depth_str, dtype=np.float32).reshape((height, width))
        if self.image_downsample > 1:
            depth = depth[::self.image_downsample, ::self.image_downsample]
        if self.inv_depth:
            depth = np.where(depth != 0, 1.0 / depth, 0.0)
        depth = depth / self.depth_scale
        return depth

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        transform = self.transforms[idx] if self.transforms is not None else np.eye(4)

        image = np.array(Image.open(color_path), dtype=np.uint8)
        
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        if self.image_downsample > 1:
            image = image[::self.image_downsample, ::self.image_downsample]
        
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth and idx < len(self.depth_paths) and self.sensor_type == "depth":
            depth_path = self.depth_paths[idx]
            depth = self.__loadExrDepth(depth_path)

            valid_depth_mask = (depth > 0.0) & (depth <= 100.0)
            
            depth = np.where(valid_depth_mask, depth, 0.0)
            
            if valid_depth_mask.shape == image.shape[:2]:
                # Expand mask to match image channels
                image_mask = valid_depth_mask[..., np.newaxis]
                image = np.where(image_mask, image, 0)

        mask_path = os.path.join(self.dataset_path, "mask", "mask.png")
        if os.path.exists(mask_path):
            gt_mask = np.array(Image.open(mask_path), dtype=np.bool_)
            
            if gt_mask.ndim == 3:
                gt_mask = gt_mask[:, :, 0]
            
            if self.image_downsample > 1:
                gt_mask = gt_mask[::self.image_downsample, ::self.image_downsample]
            
            if gt_mask.shape == image.shape[:2]:
                gt_mask_3ch = gt_mask[..., np.newaxis]
                image = np.where(gt_mask_3ch, image, 0)
                
                if depth is not None:
                    depth = np.where(gt_mask, depth, 0.0)

        image = (
            torch.from_numpy(image.astype(np.float32) / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        transform = torch.from_numpy(transform).to(device=self.device, dtype=self.dtype)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return image, depth, transform


def load_dataset(args, path, config, max_num_frames=None, start_frame=0):
    if config["Dataset"]["type"] == "panorama":
        return PanoramaDataset(args, path, config, max_num_frames=max_num_frames, start_frame=start_frame)
    else:
        raise ValueError("Unknown dataset type")
