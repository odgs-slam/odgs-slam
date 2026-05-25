import gc
import os
import time
from multiprocessing.queues import Queue

import numpy as np
import torch
import torch.multiprocessing as mp
from munch import Munch
from scipy.spatial import cKDTree

from gaussian_splatting.gaussian_renderer import render_spherical
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.dataset import PanoramaDataset
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.pose_utils import update_pose
from utils.resource_monitor import get_resource_usage
from utils.similarity_utils import equirectangular_similarity
from utils.slam_utils import get_loss_tracking, get_median_depth


class FrontEnd(mp.Process):
    def __init__(self,
                 config: dict,
                 gaussians: GaussianModel,
                 dataset: PanoramaDataset,
                 background: torch.Tensor,
                 pipeline_params: Munch,
                 frontend_queue: Queue,
                 backend_queue: Queue,
                 q_main2vis: Queue | FakeQueue,
                 q_vis2main: Queue | FakeQueue):
        super().__init__()

        self.config: dict = config
        self.dataset: PanoramaDataset = dataset
        self.background: torch.Tensor = background
        self.pipeline_params: Munch = pipeline_params
        self.frontend_queue: Queue = frontend_queue
        self.backend_queue: Queue = backend_queue
        self.q_main2vis: Queue | FakeQueue = q_main2vis
        self.q_vis2main: Queue | FakeQueue = q_vis2main
        self.set_hyperparams()

        self.initialized: bool = False
        self.kf_indices: list[int] = []
        self.monocular: bool = config["Training"]["monocular"]
        self.iteration_count: int = 0
        self.occ_aware_visibility: dict[int, torch.Tensor] = {}
        self.current_window: list[int] = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1
        self.kf_remove_interval = config["Training"].get("kf_remove_interval", 100)
        self.kf_counter = 0

        self.gaussians: GaussianModel = gaussians
        self.cameras: dict[int, Camera] = dict()
        self.device = "cuda:0"
        self.pause = False

        self.tracking_times = []
        self.kf_removal_times = []  # Store keyframe removal times
        self.ram_usages = []
        self.gpu_usages = []
        
        self.poses = []

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.use_gui = self.config["Results"].get("use_gui", False)
        self.gui_active = self.use_gui
        self.exit_on_gui_close = self.config["Results"].get("exit_on_gui_close", False)
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"].get("rgb_boundary_threshold", 0.01)
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                if opacity is not None:
                    opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(self.config,
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        self.config, depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False
        
    def init_new_frame_pose_constant_motion(self, cur_frame_idx: int, viewpoint: Camera) -> tuple[torch.Tensor, torch.Tensor]:
        # Constant motion assumption based on weighted average of last 3 motions
        if len(self.cameras) > 4:
            weights = [0.5, 0.3, 0.2] # sum must be 1

            rotations = []
            translations = []

            for i in range(3):
                cam_i = self.cameras[cur_frame_idx - 2 - i]
                cam_i1 = self.cameras[cur_frame_idx - 1 - i]

                W2C_i = getWorld2View2(cam_i.R, cam_i.T)
                W2C_i1 = getWorld2View2(cam_i1.R, cam_i1.T)

                delta = W2C_i1 @ torch.linalg.inv(W2C_i)
                R = delta[:3, :3]
                T = delta[:3, 3]

                rotations.append(R)
                translations.append(T)

            # Weighted average of translations
            T_avg = sum(w * t for w, t in zip(weights, translations, strict=True))

            # Weighted average of rotation matrices (not ideal, but usable)
            R_avg_raw = sum(w * R for w, R in zip(weights, rotations, strict=True))

            # Re-orthogonalize R_avg using SVD
            U, _, Vt = torch.linalg.svd(R_avg_raw)
            R_avg = U @ Vt

            delta_avg = torch.eye(4, device=R_avg.device)
            delta_avg[:3, :3] = R_avg
            delta_avg[:3, 3] = T_avg

            prev = self.cameras[cur_frame_idx - 1]
            W2C_prev = getWorld2View2(prev.R, prev.T)

            W2C_curr = delta_avg @ W2C_prev
            R_new = W2C_curr[:3, :3]
            T_new = W2C_curr[:3, 3]
            return R_new, T_new
        elif len(self.cameras) > 2:
            # If not enough frames are available, use simple constant motion
            prev = self.cameras[cur_frame_idx - 1]
            prev_prev = self.cameras[cur_frame_idx - 2]
            W2C_prev_prev = getWorld2View2(prev_prev.R, prev_prev.T)
            W2C_prev = getWorld2View2(prev.R, prev.T)
            delta_Rt = W2C_prev @ torch.linalg.inv(W2C_prev_prev)
            
            W2C_curr = delta_Rt @ W2C_prev
            R_new = W2C_curr[:3, :3]
            T_new = W2C_curr[:3, 3]
            return R_new, T_new
        else:
            # first frame cannot be initialized with constant motion
            prev = self.cameras[cur_frame_idx - 1]
            return prev.R, prev.T


    def tracking(self, cur_frame_idx: int, viewpoint: Camera):
        Log(f"Tracking frame: {cur_frame_idx}", tag="Frontend")

        R_new, T_new = self.init_new_frame_pose_constant_motion(cur_frame_idx, viewpoint)
        viewpoint.update_RT(R_new, T_new)

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_q_w2c],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": f"rot_{viewpoint.uid}",
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_t_w2c],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
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

        pose_optimizer = torch.optim.Adam(opt_params)
        tracking_start_time = time.time()  # Start timing

        render_pkg = None
        image, depth, opacity = None, None, None
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render_spherical(
                viewpoint, self.gaussians, self.pipeline_params, self.background, mapping_mode=False, tracking_mode=True
            )
            if render_pkg is None:
                raise RuntimeError(f"Rendering failed during tracking in iteration: {tracking_itr}.")

            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if self.use_gui and self.gui_active and tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

        tracking_end_time = time.time()
        tracking_duration = tracking_end_time - tracking_start_time
        self.tracking_times.append(tracking_duration)

        if depth is not None: 
            self.median_depth = get_median_depth(self.config, depth, opacity)
        else:
            Log("Warning: Depth is None after tracking; median depth not updated.", tag="Warn")
        
        self.poses.append(
            (cur_frame_idx, viewpoint.R.detach().clone().cpu(), viewpoint.T.detach().clone().cpu(), viewpoint.R_gt.detach().clone().cpu(), viewpoint.T_gt.detach().clone().cpu())
            )
        return render_pkg


    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
        check_pose_delta = True
    ):
        kf_overlap = self.config["Training"]["kf_overlap"]
        
        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio = intersection / union

        if not check_pose_delta:
            return point_ratio < kf_overlap

        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        # check if the distance to previous is larger than a threshold parameter scaled by the scene median depth
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        return (point_ratio < kf_overlap and dist_check2) or dist_check


    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None

        cut_off = (
            self.config["Training"].get("kf_cutoff", 0.4)
        )
        if not self.initialized:
            cut_off = 0.4

        # go from oldest to newest and if one kf should be removed, break
        for i in range(len(window)-1, N_dont_touch-1, -1):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)
                break

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(
                        1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def remove_close_keyframes(self, min_dist=0.1, similarity_threshold=0.02):
        start_time = time.time()
        positions = np.stack([(-self.cameras[idx].R.cpu().numpy().T @ self.cameras[idx].T.cpu().numpy()) for idx in self.kf_indices])
        tree = cKDTree(positions)
        pairs = tree.query_pairs(r=min_dist)
        
        # redundancy graph
        redundancy_graph = {idx: set() for idx in self.kf_indices}
        for i, j in pairs:
            idx_i = self.kf_indices[i]
            idx_j = self.kf_indices[j]
            if idx_i in self.current_window or idx_j in self.current_window:
                continue
            img_i = self.cameras[idx_i].original_image
            img_j = self.cameras[idx_j].original_image

            if equirectangular_similarity(img_i, img_j, threshold=similarity_threshold, method="l1", 
                                        R1=self.cameras[idx_i].R, R2=self.cameras[idx_j].R):
                redundancy_graph[idx_i].add(idx_j)
                redundancy_graph[idx_j].add(idx_i)

        # determine which keyframes to remove based on redundancy
        def redundancy_score(idx):
            age_score = self.kf_indices.index(idx)
            connectivity_score = len(redundancy_graph[idx])
            return 0.5 * age_score + 1.0 * connectivity_score

        # greedy removal
        to_remove = set()
        visited = set()
        sorted_kfs = sorted(self.kf_indices, key=redundancy_score)
        for idx in sorted_kfs:
            if idx in visited:
                continue
            # mark all neighbors as visited and removed
            for neighbor in redundancy_graph[idx]:
                if neighbor not in visited:
                    to_remove.add(neighbor)
                    visited.add(neighbor)
            visited.add(idx)

        for idx in to_remove:
            self.cameras[idx].clean()
            del self.cameras[idx]
            torch.cuda.empty_cache()
            self.kf_indices.remove(idx)
            gc.collect()

        elapsed = time.time() - start_time
        self.kf_removal_times.append(elapsed)

        return list(to_remove)

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        # # Extract data first
        # new_gaussians = data[1]
        # new_occ_aware_visibility = data[2]
        # keyframes = data[3]
        
        # # Clean up old gaussians before replacing with new one from backend
        # if hasattr(self, 'gaussians') and self.gaussians is not None:
        #     del self.gaussians
        #     torch.cuda.empty_cache()
        #     gc.collect()
        
        # # Now assign the new objects
        # self.gaussians = new_gaussians
        # self.occ_aware_visibility = new_occ_aware_visibility

        self.gaussians = data[1]
        self.occ_aware_visibility = data[2]
        keyframes = data[3]
        
        del data

        for kf_id, kf_R, kf_T in keyframes:
            if kf_id not in self.cameras:
                continue
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()
        if cur_frame_idx % 20 == 0:
            gc.collect()
    
    def save_tracking_times(self):
        os.makedirs(self.save_dir, exist_ok=True)
        out_path = os.path.join(self.save_dir, "tracking_times.csv")
        with open(out_path, "w") as f:
            for t in self.tracking_times:
                f.write(f"{t}\n")

    def save_kf_removal_times(self):
        os.makedirs(self.save_dir, exist_ok=True)
        out_path = os.path.join(self.save_dir, "kf_removal_times.csv")
        with open(out_path, "w") as f:
            for t in self.kf_removal_times:
                f.write(f"{t}\n")

    def save_resource_usages(self):
        os.makedirs(self.save_dir, exist_ok=True)
        ram_path = os.path.join(self.save_dir, "frontend_ram_usage.csv")
        gpu_path = os.path.join(self.save_dir, "frontend_gpu_usage.csv")
        with open(ram_path, "w") as f:
            for v in self.ram_usages:
                f.write(f"{v}\n")
        with open(gpu_path, "w") as f:
            for v in self.gpu_usages:
                f.write(f"{v}\n")

    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            # Log(f"Frontend processing frame {cur_frame_idx}.", tag="Frontend")
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                if data_vis2main.flag_gui_exit:
                    self.q_vis2main = FakeQueue()
                    self.q_main2vis = FakeQueue()
                    self.use_gui = False
                    self.gui_active = False
                    if self.exit_on_gui_close:
                        Log("GUI closed, exiting SLAM frontend.", tag="Frontend")
                        break
                    else:
                        Log("GUI closed, but not exiting SLAM frontend.", tag="Frontend")

                pause = data_vis2main.flag_pause
                changed = (pause != self.pause)
                self.pause = pause
                if self.pause and changed:
                    Log("Pausing SLAM frontend as per GUI request.", tag="Frontend")
                    self.backend_queue.put(["pause"])
                    continue
                elif not self.pause and changed:
                    Log("Unpausing SLAM frontend as per GUI request.", tag="Frontend")
                    self.backend_queue.put(["unpause"])
                
                if self.gui_active != data_vis2main.flag_gui_active:
                    self.gui_active = data_vis2main.flag_gui_active
                    # if GUI just activated, send current state
                    if self.gui_active:
                        Log("GUI activated, sending current state to GUI thread.", tag="Frontend")
                        self.q_main2vis.put(
                            gui_utils.GaussianPacket(
                                gaussians= self.gaussians if self.gaussians.get_xyz.shape[0] > 0 else None,
                                current_frame=self.cameras.get(
                                    cur_frame_idx - 1, None),
                                keyframes=[self.cameras[kf_idx]
                                           for kf_idx in self.kf_indices],
                                kf_window={self.current_window[0]: self.current_window[1:]},
                                active_kf_ids=self.kf_indices.copy(),
                            )
                        )

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                
                if render_pkg is None:
                    Log("Rendering failed during tracking; skipping frame.", tag="Warn")
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                if self.use_gui and self.gui_active:
                    current_window_dict = {}
                    current_window_dict[self.current_window[0]
                                        ] = self.current_window[1:]
                    keyframes = [self.cameras[kf_idx]
                                 for kf_idx in self.current_window]
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=self.gaussians if self.gaussians.get_xyz.shape[0] > 0 else None,
                            current_frame=viewpoint,
                            keyframes=keyframes,
                            kf_window=current_window_dict,
                            active_kf_ids=self.kf_indices.copy(),
                        )
                    )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx -
                              last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                
                window_not_full = len(self.current_window) < self.window_size

                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                    window_not_full,  # if there were not enough frames seen yet (< window_size) we perform distance-based keyframe selection too.
                )
                if self.single_thread or window_not_full:
                    create_kf = check_time and create_kf
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting.", tag="Frontend"
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )

                    ram, gpu = get_resource_usage()
                    self.ram_usages.append(ram)
                    self.gpu_usages.append(gpu)

                    self.kf_counter += 1
                    if self.kf_counter % self.kf_remove_interval == 0 and self.kf_counter >= self.kf_remove_interval:
                        min_dist = self.config["Training"].get("kf_remove_min_dist", 0.1)
                        similarity_threshold = self.config["Training"].get(
                            "kf_remove_similarity_threshold", 0.02
                        )
                        removed_kfs = self.remove_close_keyframes(min_dist=min_dist, similarity_threshold=similarity_threshold)
                        if removed_kfs:
                            self.backend_queue.put(["remove_keyframes", removed_kfs])
                            Log(
                                f"Removed redundant keyframe(s): {removed_kfs}", tag="Frontend"
                            )

                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and self.kf_counter % self.save_trj_kf_intv == 0
                ):
                    Log(f"Evaluating ATE at frame: {cur_frame_idx}", tag="Frontend")
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))

                to_keep = set(self.kf_indices) | set(self.current_window)
                prev_frame = cur_frame_idx - self.use_every_n_frames
                if prev_frame >= 0:
                    to_keep.add(prev_frame)
                # keep the last 4 frames for constant motion
                last_cam_keys = list(self.cameras.keys())[-4:]
                to_keep.update(last_cam_keys)
                to_delete = [idx for idx in self.cameras if idx not in to_keep]
                for idx in to_delete:
                    self.cameras[idx].clean()
                    del self.cameras[idx]
                    torch.cuda.empty_cache()
                    gc.collect()

            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Stopped.", tag="Frontend")
                    break

        if self.use_gui:
            # send final state to GUI
            self.q_main2vis.put(
                gui_utils.GaussianPacket(
                    gaussians=self.gaussians if self.gaussians.get_xyz.shape[0] > 0 else None,
                    current_frame=self.cameras.get(
                        cur_frame_idx - 1, None),
                    keyframes=[self.cameras[kf_idx]
                               for kf_idx in self.kf_indices],
                    kf_window={self.current_window[0]: self.current_window[1:]},
                    active_kf_ids=self.kf_indices.copy(),
                )
            )

        if self.save_results:
            if len(self.cameras) < 2:
                Log("No cameras to evaluate ATE on.", tag="Frontend")
                return
            eval_ate(
                self.cameras,
                self.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )
            save_gaussians(
                self.gaussians, self.save_dir, "final", final=True
            )
            self.save_tracking_times()
            self.save_kf_removal_times()
            self.save_resource_usages()
        return