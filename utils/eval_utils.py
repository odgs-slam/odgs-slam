import json
import os

import cv2
import evo
import evo.tools.plot
import matplotlib
import numpy as np
import torch
from evo.core import metrics
from evo.core.trajectory import PosePath3D
from matplotlib import pyplot as plt
from tqdm import tqdm

matplotlib.use('Agg')
import wandb
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gaussian_splatting.gaussian_renderer import render_spherical
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import masked_lpips, masked_ssim, ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.camera_utils import Camera
from utils.logging_utils import Log


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    # Plot
    if poses_est is None or len(poses_est) < 3:
        return None
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est.align(traj_ref, correct_scale=monocular)

    # RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE [m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, f"stats_{str(label)}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax=ax,
        traj=traj_est,
        array=ape_metric.error,
        plot_mode=plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(
        plot_dir, f"evo_2dplot_{str(label)}.png"), dpi=90)
    plt.close(fig)

    return ape_stat


def eval_ate(frames_or_poses, kf_ids=None, save_dir=None, iterations=None, final=False, monocular=False):
    trj_data = dict()
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    if isinstance(frames_or_poses, list):
        # New format: poses array
        latest_frame_idx = frames_or_poses[-1][0] + 2 if final else frames_or_poses[-1][0] + 1
        
        for frame_idx, R, T, R_gt, T_gt in frames_or_poses:
            pose_est = np.linalg.inv(gen_pose_matrix(R, T))
            pose_gt = np.linalg.inv(gen_pose_matrix(R_gt, T_gt))

            trj_id.append(frame_idx)
            trj_est.append(pose_est.tolist())
            trj_gt.append(pose_gt.tolist())

            trj_est_np.append(pose_est)
            trj_gt_np.append(pose_gt)
    else:
        if kf_ids is None:
            raise ValueError("kf_ids must be provided when using frames format")
        
        frames = frames_or_poses
        latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
        
        for kf_id in kf_ids:
            kf = frames[kf_id]
            pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
            pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

            trj_id.append(frames[kf_id].uid)
            trj_est.append(pose_est.tolist())
            trj_gt.append(pose_gt.tolist())

            trj_est_np.append(pose_est)
            trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    if save_dir is not None:
        plot_dir = os.path.join(save_dir, "plot")
        mkdir_p(plot_dir)

        label_evo = "final" if final else f"{iterations:04}"
        with open(
            os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(trj_data, f, indent=4)

        ate = evaluate_evo(
            poses_gt=trj_gt_np,
            poses_est=trj_est_np,
            plot_dir=plot_dir,
            label=label_evo,
            monocular=monocular,
        )
    else:
        traj_ref = PosePath3D(poses_se3=trj_gt_np)
        traj_est = PosePath3D(poses_se3=trj_est_np)
        traj_est.align(traj_ref, correct_scale=monocular)
        
        pose_relation = metrics.PoseRelation.translation_part
        data = (traj_ref, traj_est)
        ape_metric = metrics.APE(pose_relation)
        ape_metric.process_data(data)
        ate = ape_metric.get_statistic(metrics.StatisticsType.rmse)

    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
):
    img_pred, img_gt, saved_frame_idx = [], [], []
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    for idx in kf_indices:
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, _, _ = dataset[idx]

        rendering = render_spherical(frame, gaussians, pipe, background, mapping_mode=False, tracking_mode=False)
        if rendering is None or "render" not in rendering:
            Log(f"Rendering failed for frame {idx}, skipping PSNR/SSIM/LPIPS calculation.", tag="Warning")
            continue
        image = torch.clamp(rendering['render'], 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose(
            (1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0),
                          (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"),
             "w", encoding="utf-8"),
        indent=4,
    )
    return output

def eval_rendering_out_of_core(
    gaussians,
    dataset,
    poses,
    frame_indices,
    save_dir,
    pipe,
    background,
    iteration="final",
    write_ith_frame=0
):
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    psnr_masked_array, ssim_masked_array, lpips_masked_input_array, lpips_masked_array = [], [], [], []

    Log(f"Starting rendering evaluation for {len(frame_indices)} frames.", tag="Eval")
    Log(f"Number of frames = {len(poses)}", tag="Eval")
    
    counter = 0
    with torch.no_grad():
        for fid, (T, R) in tqdm(zip(frame_indices, poses, strict=True), total=len(frame_indices)):
            gt_image, gt_depth, gt_pose = dataset[fid]

            render_cam = Camera(
                fid,
                None,
                None,
                gt_pose,
                torch.eye(4),
                dataset.fx,
                dataset.fy,
                dataset.cx,
                dataset.cy,
                dataset.fovx,
                dataset.fovy,
                dataset.height,
                dataset.width,
                device=dataset.device,
            )
            render_cam.update_RT(torch.tensor(R), torch.tensor(T))
            render_cam.clean()

            rendering = render_spherical(render_cam, gaussians, pipe, background, mapping_mode=False, tracking_mode=False)
            if rendering is None or "render" not in rendering:
                Log(f"Rendering failed for frame {fid}, skipping PSNR/SSIM/LPIPS calculation.", tag="Warning")
                continue
            image = torch.clamp(rendering['render'], 0.0, 1.0)

            mask = gt_image > 0

            psnr_score = psnr((image).unsqueeze(0), (gt_image).unsqueeze(0))
            ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
            lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

            psnr_array.append(psnr_score.item())
            ssim_array.append(ssim_score.item())
            lpips_array.append(lpips_score.item())

            psnr_masked_score = psnr((image[mask]).unsqueeze(0), gt_image[mask].unsqueeze(0))
            ssim_masked_score = masked_ssim((image).unsqueeze(0), (gt_image).unsqueeze(0), mask.unsqueeze(0))
            masked_img = image * mask
            masked_gt = gt_image * mask
            lpips_masked_input_score = cal_lpips(masked_img.unsqueeze(0), masked_gt.unsqueeze(0))
            lpips_masked_score = masked_lpips(image.unsqueeze(0), gt_image.unsqueeze(0), mask.unsqueeze(0))
            
            psnr_masked_array.append(psnr_masked_score.item())
            ssim_masked_array.append(ssim_masked_score.item())
            lpips_masked_input_array.append(lpips_masked_input_score.item())
            lpips_masked_array.append(lpips_masked_score.item())

            if write_ith_frame > 0 and counter % write_ith_frame == 0:
                pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
                    np.uint8
                )
                gt = (gt_image.cpu().numpy().transpose(
                    (1, 2, 0)) * 255).astype(np.uint8)
                pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

                img_save_dir = os.path.join(save_dir, "rendered_images", str(iteration))
                mkdir_p(img_save_dir)
                cv2.imwrite(os.path.join(img_save_dir, f"frame_{fid}_pred.png"), pred)
                cv2.imwrite(os.path.join(img_save_dir, f"frame_{fid}_gt.png"), gt)
            counter += 1


    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["mean_psnr_masked"] = float(np.mean(psnr_masked_array))
    output["mean_ssim_masked"] = float(np.mean(ssim_masked_array))
    output["mean_lpips_masked_input"] = float(np.mean(lpips_masked_input_array))
    output["mean_lpips_masked"] = float(np.mean(lpips_masked_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    with open(os.path.join(psnr_save_dir, 'final_result.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)
    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, f"point_cloud/iteration_{str(iteration)}"
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
