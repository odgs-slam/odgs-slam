import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from munch import Munch, munchify

from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.common_utils import set_random_seed
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_rendering_out_of_core
from utils.load_eval_data_utils import get_poses_from_trajectory, load_trajectories
from utils.logging_utils import Log


def run_render_evaluation(config, results_folder_path: str | Path, write_ith_frame: int = 0):
    rand_seed = config["pipeline_params"].get("random_seed", None)

    if rand_seed is not None:
        set_random_seed(rand_seed)

    model_params: Munch = munchify(config["model_params"])  # type: ignore
    # opt_params: Munch = munchify(config["opt_params"]) # type: ignore
    pipeline_params: Munch = munchify(config["pipeline_params"])  # type: ignore

    monocular = config["Dataset"]["sensor_type"] == "monocular"
    use_spherical_harmonics = config["Training"]["spherical_harmonics"]

    model_params.sh_degree = 3 if use_spherical_harmonics else 0

    model_path = Path(results_folder_path) / "point_cloud" / "final" / "point_cloud.ply"
    if not model_path.exists() and not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    gaussians: GaussianModel = GaussianModel(model_params.sh_degree, config=config)
    gaussians.load_ply(model_path)
    gaussians.init_lr(6.0)

    dataset: torch.utils.data.Dataset = load_dataset(
        model_params, model_params.source_path, config=config
    )

    # gaussians.training_setup(opt_params)
    bg_color = [0, 0, 0]
    background: torch.Tensor = torch.tensor(
        bg_color, dtype=torch.float32, device="cuda"
    )
    config["Training"]["monocular"] = monocular

    trajectory_file = Path(results_folder_path) / "plot" / "trj_final.json"
    trj_idx, trj_est, trj_gt = load_trajectories(str(trajectory_file))
    poses_est = get_poses_from_trajectory(trj_est)

    if len(poses_est) != len(trj_idx):
        raise ValueError("Mismatch between trajectory indices and estimated poses.")

    rendering_result = eval_rendering_out_of_core(
        gaussians,
        dataset,
        poses_est,
        trj_idx,
        config["Results"]["save_dir"],
        pipeline_params,
        background,
        iteration="post_run_evaluation",
        write_ith_frame=write_ith_frame
    )
    Log(
        f"Rendering evaluation complete for frames loaded from {trajectory_file}",
        tag="Info",
    )
    Log(f"Rendering results: {rendering_result}", tag="Info")
    Log(
        f"Results written to {config['Results']['save_dir']}/psnr/post_run_evaluation/final_result.json",
        tag="Info",
    )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--results_folder", type=str)
    parser.add_argument("--write_ith_frame", type=int, default=0, help="Write every ith frame during evaluation, if 0, no frames will be written.")

    args = parser.parse_args()

    results_folder_path = Path(args.results_folder)
    write_ith_frame = args.write_ith_frame

    if not results_folder_path.exists() or not results_folder_path.is_dir():
        Log(
            f"Results folder {results_folder_path} does not exist or is not a directory.",
            tag="Error",
        )
        sys.exit(1)

    config_path = results_folder_path / "config.yml"
    config = load_config(str(config_path))

    run_render_evaluation(config, results_folder_path, write_ith_frame=write_ith_frame)

    # All done
    Log("Done.", tag="Info")
