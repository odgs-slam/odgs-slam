import sys
from argparse import ArgumentParser
from multiprocessing.queues import Queue
from time import sleep

import torch
import torch.multiprocessing as mp
import yaml
from munch import Munch, munchify

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from gui import gui_utils, slam_gui
from utils.camera_utils import Camera as CameraUtils
from gui.gl_render.util import Camera
from utils.common_utils import set_random_seed
from utils.config_utils import load_config
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue, clone_obj
from utils.dataset import load_dataset


def run_viewer(config, model_path):
    rand_seed = config["pipeline_params"].get("random_seed", None)

    if rand_seed is not None:
        set_random_seed(rand_seed)

    model_params: Munch = munchify(config["model_params"]) # type: ignore
    # opt_params: Munch = munchify(config["opt_params"]) # type: ignore
    pipeline_params: Munch = munchify(config["pipeline_params"]) # type: ignore

    monocular = config["Dataset"]["sensor_type"] == "monocular"
    use_spherical_harmonics = config["Training"]["spherical_harmonics"]
    config["Results"]["use_gui"] = True
    config["Results"]["exit_gui_on_finish"] = False
    use_gui = config["Results"]["use_gui"]
    
    model_params.sh_degree = 3 if use_spherical_harmonics else 0

    gaussians: GaussianModel = GaussianModel(model_params.sh_degree, config=config)
    gaussians.load_ply(model_path)
    gaussians.init_lr(6.0)
    dataset: torch.utils.data.Dataset = load_dataset(
        model_params, model_params.source_path, config=config
    )

    # gaussians.training_setup(opt_params)
    bg_color = [0, 0, 0]
    background: torch.Tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    q_main2vis: Queue | FakeQueue = mp.Queue() if use_gui else FakeQueue()
    q_vis2main: FakeQueue = FakeQueue()

    config["Training"]["monocular"] = monocular

    params_gui = gui_utils.ParamsGUI(
        pipe=pipeline_params,
        background=background,
        # gaussians=gaussians,
        q_main2vis=q_main2vis,
        q_vis2main=q_vis2main,
        exit_gui_on_finish=config["Results"]["exit_gui_on_finish"]
    )

    projection_matrix = getProjectionMatrix2(
                znear=0.01,
                zfar=100.0,
                fx=dataset.fx,
                fy=dataset.fy,
                cx=dataset.cx,
                cy=dataset.cy,
                W=dataset.width,
                H=dataset.height,
            ).transpose(0, 1)
    projection_matrix = projection_matrix.to(device=torch.device("cuda"))

    viewpoint = CameraUtils.init_from_dataset(dataset, 10, projection_matrix)
    viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

    q_main2vis.put(
        gui_utils.GaussianPacket(
            gaussians= clone_obj(gaussians),
            finish=True,
            current_frame=viewpoint,
        )
    )

    gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
    gui_process.start()
    sleep(1)

    q_main2vis.put(
        gui_utils.GaussianPacket(
            gaussians= clone_obj(gaussians),
            finish=True,
            current_frame=viewpoint,
        )
    )

    gui_process.join()


        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--model_path", type=str, help="Path to the 3DGS model file (.ply).")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config) as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)

    run_viewer(config, model_path=args.model_path)

    # All done
    Log("Done.", tag="Viewer")