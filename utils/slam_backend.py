import gc
import os
import random
import time
from multiprocessing.queues import Queue

import torch
import torch.multiprocessing as mp
from munch import Munch
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render_spherical
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.resource_monitor import get_resource_usage
from utils.slam_utils import get_loss_mapping


class BackEnd(mp.Process):
    def __init__(self, 
                 config: dict,
                 gaussians: GaussianModel,
                 background: torch.Tensor,
                 cameras_extent: float,
                 pipeline_params: Munch,
                 opt_params: Munch,
                 frontend_queue: Queue,
                 backend_queue: Queue,
                 live_mode: bool):

        super().__init__()


        self.config: dict = config
        self.gaussians: GaussianModel = gaussians
        self.background: torch.Tensor = background
        self.cameras_extent: float = cameras_extent
        self.pipeline_params: Munch = pipeline_params
        self.opt_params: Munch = opt_params
        self.frontend_queue: Queue = frontend_queue
        self.backend_queue: Queue = backend_queue
        self.live_mode = live_mode
        self.set_hyperparams()

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.mapping_times = []
        self.ram_usages = []
        self.gpu_usages = []

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent *
            self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Training"].get("single_thread", False)
        )
        self.gc_iter_count = self.config["Training"].get("gc_iter_count", 20)

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        render_pkg = None
        n_touched = None
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render_spherical(
                viewpoint, self.gaussians, self.pipeline_params, self.background, mapping_mode=True, tracking_mode=False
            )
            if render_pkg is None:
                raise RuntimeError(f"Rendering failed during initialization in iteration: {mapping_iteration}.")
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        if n_touched is None:
            raise RuntimeError("n_touched is None during map initialization.")
        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map", tag="Backend")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx]
                           for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        gaussian_split = False
        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render_spherical(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, mapping_mode=True, tracking_mode=False
                )
                if render_pkg is None:
                    raise RuntimeError(f"Rendering failed for viewpoint in current window with id: {cam_idx}, in iteration: {self.iteration_count}.")
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render_spherical(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, mapping_mode=True, tracking_mode=False
                )
                if render_pkg is None:
                    raise RuntimeError(f"Rendering failed for random viewpoint optimization with id: {cam_idx}, in iteration: {self.iteration_count}.")
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(
                scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            # Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range(len(current_window)):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        # only prune, if we are in monocular mode (since there points are initialized randomly)
                        if self.monocular:
                            to_prune = None
                            if prune_mode == "odometry":
                                to_prune = self.gaussians.n_obs < 3
                                # make sure we don't split the gaussians, break here.
                            if prune_mode == "slam":
                                # only prune keyframes which are relatively new
                                sorted_window = sorted(
                                    current_window, reverse=True)
                                mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                                if not self.initialized:
                                    mask = self.gaussians.unique_kfIDs >= 0
                                to_prune = torch.logical_and(
                                    self.gaussians.n_obs <= prune_coviz, mask
                                )
                            if to_prune is not None:
                                self.gaussians.prune_points(to_prune.cuda())
                                for idx in range((len(current_window))):
                                    current_idx = current_window[idx]
                                    self.occ_aware_visibility[current_idx] = (
                                        self.occ_aware_visibility[current_idx][~to_prune]
                                    )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM", tag="Backend")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                # Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians", tag="Backend")
                    self.gaussians.reset_opacity_nonvisible(
                        visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        return gaussian_split

    def color_refinement(self):
        Log("Starting color refinement", tag="Backend")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render_spherical(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background, mapping_mode=True, tracking_mode=False
            )
            if render_pkg is None:
                raise RuntimeError(f"Rendering failed during color refinement in iteration: {iteration}.")
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)

            del image, gt_image, visibility_filter, radii, render_pkg, Ll1, loss
            torch.cuda.empty_cache()

        gc.collect()
        Log("Map refinement done", tag="Backend")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians),
               self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def save_mapping_times(self):
        save_dir = self.config["Results"]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "mapping_times.csv")
        with open(out_path, "w") as f:
            for t in self.mapping_times:
                f.write(f"{t}\n")

    def save_resource_usages(self):
        save_dir = self.config["Results"]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        ram_path = os.path.join(save_dir, "backend_ram_usage.csv")
        gpu_path = os.path.join(save_dir, "backend_gpu_usage.csv")
        with open(ram_path, "w") as f:
            for v in self.ram_usages:
                f.write(f"{v}\n")
        with open(gpu_path, "w") as f:
            for v in self.gpu_usages:
                f.write(f"{v}\n")

    def save_keyframe_renders(self):
        """Save rendered images from all keyframes after SLAM is complete"""
        Log("Saving keyframe renders...", tag="Backend")
        save_dir = self.config["Results"]["save_dir"]
        render_dir = os.path.join(save_dir, "keyframe_renders")
        os.makedirs(render_dir, exist_ok=True)
        
        for kf_idx, viewpoint in self.viewpoints.items():
            Log(f"Rendering keyframe {kf_idx}", tag="Backend")
            
            # Render the image
            render_pkg = render_spherical(
                viewpoint, 
                self.gaussians, 
                self.pipeline_params, 
                self.background, 
                mapping_mode=False, 
                tracking_mode=False
            )
            if render_pkg is None:
                raise RuntimeError(f"Rendering failed while saving image for keyframe {kf_idx}.")
            
            rendered_image = render_pkg["render"]
            depth_image = render_pkg.get("depth", None)
            
            # Convert to numpy and save
            import numpy as np
            from PIL import Image
            
            # Save RGB render
            rgb_np = rendered_image.detach().cpu().numpy()
            rgb_np = np.transpose(rgb_np, (1, 2, 0))  # CHW to HWC
            rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
            rgb_image = Image.fromarray(rgb_np)
            rgb_path = os.path.join(render_dir, f"keyframe_{kf_idx:04d}_render.png")
            rgb_image.save(rgb_path)
            
            # Save depth if available
            if depth_image is not None:
                depth_np = depth_image.detach().cpu().numpy().squeeze()
                depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
                depth_normalized = (depth_normalized * 255).astype(np.uint8)
                depth_pil = Image.fromarray(depth_normalized, mode='L')
                depth_path = os.path.join(render_dir, f"keyframe_{kf_idx:04d}_depth.png")
                depth_pil.save(depth_path)
        
        Log(f"Saved renders for {len(self.viewpoints)} keyframes to {render_dir}", tag="Backend")

    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.save_keyframe_renders()
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system", tag="Backend")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    Log(f"Adding frame '{cur_frame_idx}' as new keyframe", tag="Backend")

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint,
                                     depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization", tag="Backend")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_q_w2c],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": f"rot_{viewpoint.uid}",
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_t_w2c],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": f"trans_{viewpoint.uid}",
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": f"exposure_a_{viewpoint.uid}",
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": f"exposure_b_{viewpoint.uid}",
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    mapping_start_time = time.time()
                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    mapping_end_time = time.time()
                    mapping_duration = mapping_end_time - mapping_start_time
                    self.mapping_times.append(mapping_duration)

                    ram, gpu = get_resource_usage()
                    self.ram_usages.append(ram)
                    self.gpu_usages.append(gpu)

                    self.push_to_frontend("keyframe")
                elif data[0] == "remove_keyframes":
                    indices_to_remove = data[1]
                    for idx in indices_to_remove:
                        if idx in self.viewpoints:
                            del self.viewpoints[idx]
                            torch.cuda.empty_cache()
                            gc.collect()
                    self.current_window = [idx for idx in self.current_window if idx not in indices_to_remove]
                else:
                    raise Exception("Unprocessed data", data)
            
            # clear GPU memory after all 20 loops
            if self.iteration_count % self.gc_iter_count == 0:
                torch.cuda.empty_cache()
                gc.collect()
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        if self.save_results:
            self.save_mapping_times()
            self.save_resource_usages()
        return
