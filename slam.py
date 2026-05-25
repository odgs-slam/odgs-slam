import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from multiprocessing import Queue

import torch
import torch.multiprocessing as mp
import wandb
import yaml
from munch import Munch, munchify

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.common_utils import set_random_seed
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None):
        rand_seed = config["pipeline_params"].get("random_seed", None)

        if rand_seed is not None:
            set_random_seed(rand_seed)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config: dict = config
        self.save_dir = save_dir
        model_params: Munch = munchify(config["model_params"]) # type: ignore
        opt_params: Munch = munchify(config["opt_params"]) # type: ignore
        pipeline_params: Munch = munchify(config["pipeline_params"]) # type: ignore
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians: GaussianModel = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset: torch.utils.data.Dataset = load_dataset(
            model_params,
            model_params.source_path,
            config=config,
            max_num_frames=self.config['Dataset'].get('max_num_frames', None),
            start_frame=self.config['Dataset'].get('start_frame', 0),
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background: torch.Tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue: Queue = mp.Queue()
        backend_queue: Queue = mp.Queue()

        q_main2vis: Queue | FakeQueue = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main: Queue | FakeQueue = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config, self.gaussians, self.dataset, self.background, self.pipeline_params, frontend_queue, backend_queue, q_main2vis, q_vis2main)
        self.backend = BackEnd(self.config, self.gaussians, self.background, 6.0, self.pipeline_params, self.opt_params, frontend_queue, backend_queue, self.live_mode)

        # self.frontend.dataset = self.dataset
        # self.frontend.background = self.background
        # self.frontend.pipeline_params = self.pipeline_params
        # self.frontend.frontend_queue = frontend_queue
        # self.frontend.backend_queue = backend_queue
        # self.frontend.q_main2vis = q_main2vis
        # self.frontend.q_vis2main = q_vis2main
        # self.frontend.set_hyperparams()

        # self.backend.gaussians = self.gaussians
        # self.backend.background = self.background
        # self.backend.cameras_extent = 6.0
        # self.backend.pipeline_params = self.pipeline_params
        # self.backend.opt_params = self.opt_params
        # self.backend.frontend_queue = frontend_queue
        # self.backend.backend_queue = backend_queue
        # self.backend.live_mode = self.live_mode
        # self.backend.set_hyperparams()

        self.params_gui = None
        if self.use_gui:
            if isinstance(q_main2vis, FakeQueue) or isinstance(q_vis2main, FakeQueue):
                raise ValueError("GUI queues cannot be of type FakeQueue when GUI is enabled.")
            self.params_gui = gui_utils.ParamsGUI(
                pipe=self.pipeline_params,
                background=self.background,
                # gaussians=self.gaussians,
                q_main2vis=q_main2vis,
                q_vis2main=q_vis2main,
                exit_gui_on_finish=self.config["Results"]["exit_gui_on_finish"],
                output_dir=self.config["Results"]["save_dir"]
            )

        backend_process = mp.Process(target=self.backend.run)
        gui_process = None
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(2)

        backend_process.start()
        self.frontend.run()
        backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        if self.eval_rendering:
            if N_frames >= 2:
                self.gaussians = self.frontend.gaussians
                kf_indices = self.frontend.kf_indices
                ATE = eval_ate(
                    self.frontend.poses,
                    save_dir=self.save_dir,
                    iterations=0,
                    final=True,
                    monocular=self.monocular,
                )

                rendering_result = eval_rendering(
                    self.frontend.cameras,
                    self.gaussians,
                    self.dataset,
                    self.save_dir,
                    self.pipeline_params,
                    self.background,
                    kf_indices=kf_indices,
                    iteration="before_opt",
                )
                columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
                metrics_table = wandb.Table(columns=columns)
                metrics_table.add_data(
                    "Before",
                    rendering_result["mean_psnr"],
                    rendering_result["mean_ssim"],
                    rendering_result["mean_lpips"],
                    ATE,
                    FPS,
                )

                # re-used the frontend queue to retrive the gaussians from the backend.
                while not frontend_queue.empty():
                    frontend_queue.get()
                if self.config["Results"]["color_refine"]:
                    backend_queue.put(["color_refinement"])
                    while True:
                        if frontend_queue.empty():
                            time.sleep(0.01)
                            continue
                        data = frontend_queue.get()
                        if data[0] == "sync_backend" and frontend_queue.empty():
                            gaussians = data[1]
                            self.gaussians = gaussians
                            break
                    if gui_process and gui_process.is_alive():
                        q_main2vis.put(
                            gui_utils.GaussianPacket(
                                gaussians=self.gaussians
                            )
                        )

                    rendering_result = eval_rendering(
                        self.frontend.cameras,
                        self.gaussians,
                        self.dataset,
                        self.save_dir,
                        self.pipeline_params,
                        self.background,
                        kf_indices=kf_indices,
                        iteration="after_opt",
                    )
                    metrics_table.add_data(
                        "After",
                        rendering_result["mean_psnr"],
                        rendering_result["mean_ssim"],
                        rendering_result["mean_lpips"],
                        ATE,
                        FPS,
                    )
                    save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)
                wandb.log({"Metrics": metrics_table})
            else:
                Log("Not enough frames to evaluate ATE and rendering.", tag="ODGS-SLAM")

        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread", tag="ODGS-SLAM")
        if gui_process and gui_process.is_alive():
            #TODO here the final parameters need to be provided to UI to update the final visualization
            q_main2vis.put(
                gui_utils.GaussianPacket(
                    gaussians=self.gaussians,
                    current_frame=self.frontend.cameras.get(len(self.frontend.cameras) - 1, None),
                    keyframes=[self.frontend.cameras[kf_idx]
                               for kf_idx in self.frontend.kf_indices],
                    kf_window={self.frontend.current_window[0]: self.frontend.current_window[1:]},
                    active_kf_ids=self.frontend.kf_indices,
                    finish=True
                )
            )
            if gui_process:
                gui_process.join()
            Log("GUI Stopped and joined the main thread", tag="ODGS-SLAM")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config) as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running ODGS-SLAM in Evaluation Mode", tag="Eval")
        Log("Following config will be overriden", tag="Eval")
        Log("\tsave_results=True", tag="Eval")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False", tag="Eval")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True", tag="Eval")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True", tag="Eval")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir, tag="Eval")
        run = wandb.init(
            project="ODGS-SLAM",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.", tag="ODGS-SLAM")
