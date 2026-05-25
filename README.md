<h1 align="center">ODGS-SLAM: Omnidirectional Gaussian Splatting SLAM</h1>

<p align="center">
<strong>Stefan Spiss, Joey Hieronimy, Marcel Ritter, Matthias Harders</strong>
</p>

<p align="center">
<a href="https://odgs-slam.github.io/">Project Page</a>
</p>

---

<p align="center">
<strong>ODGS-SLAM</strong> is a dense visual SLAM system that brings 3D Gaussian Splatting to full 360° panoramic image sequences. It takes equirectangular RGB or RGBD frames as input and uses a Gaussian map as the sole scene representation for both camera tracking and scene mapping. The SLAM logic lives in this repository; the differential omnidirectional Gaussian rasteriser is developed separately and included as a submodule (<a href="https://github.com/odgs-slam/omni-gaussian-rasterization-w-pose">source</a>).
</p>

---

## Installation

```bash
git clone https://github.com/odgs-slam/odgs-slam.git --recursive
cd odgs-slam
```

Set up the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/omni-gaussian-rasterization
```

`requirements.txt` pins the exact package versions used for the paper's evaluation runs on the system specified in the Reproducibility section. For a looser installation without pinned versions, use `requirements_general.txt` instead.

Depending on the system, it might be required to apply some small patches to the submodules (see files `submodules/simple-knn_glog-and-flt-limits.patch`, `submodules/diff-gaussian-rasterization_glog-and-cstdint.patch`, `submodules/omni-gaussian-rasterization_glog-and-cstdint.patch`)

---

## Running the System

Configuration files for all sequences are provided in `./configs`, organized into `./configs/rgb/` and `./configs/rgbd/` for RGB and RGBD modes respectively. Detailed descriptions of the parameters can be found in `./configs/base_config.yml`.

Download the dataset from [here](https://researchdata.uibk.ac.at/records/z6f6r-sjc65) and place the unzipped folder in `./data/`.

Examples using the synthetic room exploration sequence:

### RGB
```bash
python slam.py --config configs/rgb/rgb_render_ex_r1.yml
```

### RGBD
```bash
python slam.py --config configs/rgbd/rgbd_render_ex_r1.yml
```

All other sequences can be run by substituting the corresponding config files.

To enable/disable the GUI, set the `use_gui` flag under the `Results` section in `./configs/base_config.yml` or in the sequence-specific config file.

### Evaluation

Add the `--eval` flag to run in headless mode, save results, and log rendering and tracking metrics:

```bash
python slam.py --config configs/rgb/rgb_render_ex_r1.yml --eval
```

Logged metrics include tracking accuracy (ATE RMSE) and rendering quality (PSNR, SSIM, LPIPS).

---

## Evaluation Setup

Results may differ slightly from those reported in the paper due to multi-process non-determinism from GPU utilisation. All reported experiments were run on workstations with Intel Core i7-9700K CPUs, with 32 GiB RAM and Nvidia RTX 4090 GPUs.

The main evaluation was run on a workstation with the following detailed specifications and the package version, as specified in `requirements.txt`:

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 |
| Python | 3.10.12 |
| CUDA | 12.6.85 |
| PyTorch | 2.5.1 |
| CPU | Intel Core i7-9700K @ 3.60GHz |
| GPU | NVIDIA GeForce RTX 4090 |
| RAM | 32 GiB |


---

## Acknowledgements

This work builds upon and was inspired by several open-source projects. We would like to thank the authors and contributors of these repositories for making their work available.

- [Gaussian Splatting SLAM (MonoGS)](https://github.com/muskie82/MonoGS)
- [ODGS](https://github.com/esw0116/ODGS)
- [GS-SLAM](https://github.com/yanchi-3dv/diff-gaussian-rasterization-for-gsslam)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Differential Gaussian Rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [Differential Gaussian Rasterization w/ Pose](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git)
- [Tiny Gaussian Splatting Viewer](https://github.com/limacv/GaussianSplattingViewer)
- [Open3D](https://github.com/isl-org/Open3D)

See `Dependencies.md` for more details.

---

## License

This repository builds on [Gaussian Splatting SLAM (MonoGS)](https://github.com/muskie82/MonoGS) (commit 6c9254c) as its base. Use of this software is subject to the [LICENSE.md](LICENSE.md) issued by Imperial College London, which permits non-commercial academic research use only. See also the original MonoGS license [here](https://github.com/muskie82/MonoGS/blob/main/LICENSE.md)

Licenses of other dependencies used in the project can be found in `Dependencies.md`.

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@InProceedings{Spiss_2026_CVPR,
    author    = {Spiss, Stefan and Hieronimy, Joey and Ritter, Marcel and Harders, Matthias},
    title     = {{ODGS-SLAM}: {O}mnidirectional {G}aussian {S}platting {SLAM}},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {26114-26123}
}
```
