"""
This is the implementation of Gaussian Splatting using GSplat implementation
"""

import json
import math
import os
import os.path as osp
import time
from argparse import Namespace
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

import cv2
import einops

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import viser
import wandb

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from plyfile import PlyData, PlyElement

from scene.cameras import Camera
from scene.dataset_readers import BasicPointCloud, SceneInfo

from sklearn.neighbors import NearestNeighbors
from typing_extensions import Literal
from utils.eval import run_eval
from utils.image_utils import convert_image_tensor2array, rgb_to_luminance, save_image
from utils.render_utils import apply_turbo_colormap, depth_to_normal

from viewer import TrainerViewer3DGS

from .bilateral_grid import BilateralGrid, color_affine_transform, total_variation_loss

from .loss import calculcate_inverse_depth_loss, ImageLoss


class SceneSplats:
    """
    This is a simplified Gaussian Splats models for gsplat Gaussian training.
    """

    def __init__(self):
        self._splats = None
        self._optimizers = None

    def create_splats_with_optimizers(
        self,
        point_cloud: BasicPointCloud,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
        scene_scale: float = 1.0,
        sh_degree: int = 3,
        sparse_grad: bool = False,
        batch_size: int = 1,
        color_format: Literal["rgbm", "rgb", "m"] = "rgb",
        device: str = "cuda",
    ):
        """
        The initialized parameters currently are hard-coded.
        """
        ############## Initialize the geometry ##################
        points = torch.from_numpy(point_cloud.points).float()
        N = points.shape[0]

        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
        quats = torch.rand((N, 4))  # [N, 4]
        opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

        print(
            f"Scene Scale: {scene_scale}. Increase the position learning rate to {1.6e-4 * scene_scale}"
        )
        params = [
            # name, value, lr
            ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
            ("scales", torch.nn.Parameter(scales), 5e-3),
            ("quats", torch.nn.Parameter(quats), 1e-3),
            ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ]

        ############### Initialie color #########################
        self.sh_degree = sh_degree
        self.color_format = color_format
        if point_cloud.colors is not None:
            color_init = torch.from_numpy(
                point_cloud.colors
            ).float()  # point cloud in float [0,1]
        else:
            color_init = None

        if (
            color_format == "rgbm"
        ):  # represent each Gaussian as a four channel RGB-Monochrome Gaussian
            if color_init is None:
                # todo: we will init the color to 0.5 for each channel in their linear space
                color_init = 0.5 * torch.ones((N, 4))
            else:
                assert (
                    color_init.shape[-1] == 4
                ), "rgbm requires initialized color to have 4-channel!"

            colors = torch.zeros((N, (sh_degree + 1) ** 2, 4))  # [N, K, C]
            colors[:, 0, :] = color_to_sh(color_init)
            params.append(("rgb_sh0", torch.nn.Parameter(colors[:, :1, :3]), 2.5e-3))
            params.append(
                ("rgb_shN", torch.nn.Parameter(colors[:, 1:, :3]), 2.5e-3 / 20)
            )
            params.append(("mono_sh0", torch.nn.Parameter(colors[:, :1, 3:]), 2.5e-3))
            params.append(
                ("mono_shN", torch.nn.Parameter(colors[:, 1:, 3:]), 2.5e-3 / 20)
            )

        elif color_format == "rgb":
            # todo: we will init the color to 0.5 when in its linear space
            if color_init is None:
                color_init = 0.5 * torch.ones((N, 3))
                # color_init = 0.21 * torch.ones((N, 3))
            else:
                assert (
                    color_init.shape[-1] == 3
                ), "rgb requires initialized color to have 4-channel!"

            colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, C]
            colors[:, 0, :] = color_to_sh(color_init)
            params.append(("rgb_sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
            params.append(
                ("rgb_shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20)
            )

        elif color_format == "m":
            if color_init is None:
                color_init = 0.5 * torch.ones((N, 1))
            else:
                assert (
                    color_init.shape[-1] == 1
                ), "monochrome requires initialized color to have 1-channel!"

            colors = torch.zeros((N, (sh_degree + 1) ** 2, 1))  # [N, K, C]
            colors[:, 0, :] = color_to_sh(color_init)
            params.append(("mono_sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
            params.append(
                ("mono_shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20)
            )
        else:
            raise RuntimeError(f"Cannot recognize color format {color_format}")

        self._splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

        ################ Create optimizer ##########################

        self._optimizers = {
            name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
                [{"params": self._splats[name], "lr": lr * math.sqrt(batch_size)}],
                eps=1e-15 / math.sqrt(batch_size),
                betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
            )
            for name, _, lr in params
        }

    @property
    def params(self) -> torch.nn.ParameterDict:
        return self._splats

    @property
    def means(self) -> torch.nn.Parameter:
        return self._splats["means"]

    @property
    def scales(self) -> torch.nn.Parameter:
        return self._splats["scales"]

    @property
    def activated_scales(self) -> torch.nn.Parameter:
        return torch.exp(self.scales)

    @property
    def quats(self) -> torch.nn.Parameter:
        return self._splats["quats"]

    @property
    def opacities(self) -> torch.nn.Parameter:
        return self._splats["opacities"]

    @property
    def activated_opacities(self) -> torch.nn.Parameter:
        return torch.sigmoid(self.opacities)
        # return self.opacities.clamp(0, 1)

    @property
    def rgb_feature(self) -> torch.nn.Parameter:
        assert "rgb_sh0" in self._splats, "Splats do not contain RGB feature!"
        return torch.cat([self._splats["rgb_sh0"], self._splats["rgb_shN"]], 1)

    @property
    def monochrome_feature(self) -> torch.nn.Parameter:
        assert "mono_sh0" in self._splats, "Splats do not contain monochrome feature!"
        return torch.cat([self._splats["mono_sh0"], self._splats["mono_shN"]], 1)

    @property
    def has_monochrome_channel(self) -> bool:
        return "mono_sh0" in self._splats

    @property
    def optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return self._optimizers

    def construct_list_of_attributes(
        self, shN_dim: int = 15, color_channel_dim: int = 3
    ):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(1 * color_channel_dim):  # sh0
            l.append("f_dc_{}".format(i))
        for i in range(shN_dim * color_channel_dim):  # shN
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.scales.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.quats.shape[1]):
            l.append("rot_{}".format(i))

        # if not fuse_3D_filter:
        #     l.append('filter_3D')
        return l

    def save_ply(self, file_path: str):
        """
        save the model to a ply file. This is compatible to a general ply file
        """

        xyz = self.means.detach().cpu().numpy()  # (Nx3)
        normals = np.zeros_like(xyz)  # (Nx3)

        # we will save the appearance feature of the Gaussians aggregated together
        if self.color_format == "rgb":
            splats_sh0 = self._splats["rgb_sh0"]
            splats_shN = self._splats["rgb_shN"]
            color_channel_dim = 3
        elif self.color_format == "m":
            splats_sh0 = self._splats["mono_sh0"]
            splats_shN = self._splats["mono_shN"]
            color_channel_dim = 1
        else:
            splats_sh0 = torch.cat(
                (self._splats["rgb_sh0"], self._splats["mono_sh0"]), dim=-1
            )
            splats_shN = torch.cat(
                (self._splats["rgb_shN"], self._splats["mono_shN"]), dim=-1
            )
            color_channel_dim = 4

        color_sh0 = (
            einops.rearrange(splats_sh0, "N 1 C -> N C").detach().cpu().numpy()
        )  # (N C)
        color_shN = (
            einops.rearrange(splats_shN, "N M C -> N (C M)").detach().cpu().numpy()
        )  # (N CxM)
        shN_dim = splats_shN.shape[1]

        rotation = self.quats.detach().cpu().numpy()  # (Nx4)
        # the saved opacities and scale are before activation
        opacities = self.opacities.detach().cpu().numpy()[..., None]
        scale = self.scales.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4")
            for attribute in self.construct_list_of_attributes(
                shN_dim=shN_dim, color_channel_dim=color_channel_dim
            )
        ]

        attributes = np.concatenate(
            (xyz, normals, color_sh0, color_shN, opacities, scale, rotation), axis=1
        )

        vertex_data = np.empty(xyz.shape[0], dtype=dtype_full)
        vertex_data[:] = list(map(tuple, attributes))
        vertex_element = PlyElement.describe(vertex_data, "vertex")

        metadata_data = np.array(
            [
                (
                    list(bytearray(self.color_format, "utf-8")),
                    color_channel_dim,
                    self.sh_degree,
                )
            ],
            dtype=[("color_format", "O"), ("color_channel", "i4"), ("sh_degree", "i4")],
        )
        metadata_element = PlyElement.describe(metadata_data, "metadata")

        plydata = PlyData([vertex_element, metadata_element], text=False)
        # plydata.comments.append(f'color_channels={self.color_format}')
        # plydata.comments.append(f"sh_degree={self.sh_degree}")

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        plydata.write(file_path)

        print(f"Saved Gaussian Splats to a PLY file: {file_path}")

        self.load_ply(file_path)

    def load_ply(self, ply_path: str, device: str = "cuda"):
        """ """
        plydata = PlyData.read(ply_path)

        if "metadata" in plydata:
            metadata = plydata["metadata"]
            self.color_format = "".join(chr(c) for c in metadata[0][0])
            color_channel_dim = int(metadata[0][1])
            print(
                f"Read in color format: {self.color_format}, with {color_channel_dim} channels"
            )

            self.sh_degree = int(metadata[0][2])
            print(f"Read number of SH degrees: {self.sh_degree}")
        else:
            # if not specified, all previous code follow this format
            color_channel_dim = 3
            self.color_format = "rgb"
            self.sh_degree = 3

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )

        opacities = np.asarray(plydata.elements[0]["opacity"])

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        quats = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            quats[:, idx] = np.asarray(plydata.elements[0][attr_name])

        params = [
            # name, value, lr
            ("means", torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float)), 1.6e-4),
            (
                "scales",
                torch.nn.Parameter(torch.tensor(scales, dtype=torch.float)),
                5e-3,
            ),
            ("quats", torch.nn.Parameter(torch.tensor(quats, dtype=torch.float)), 1e-3),
            (
                "opacities",
                torch.nn.Parameter(torch.tensor(opacities, dtype=torch.float)),
                5e-2,
            ),
        ]

        # Load SH coefficients
        sh0 = np.zeros((xyz.shape[0], 1, color_channel_dim))
        for idx in range(color_channel_dim):
            sh0[:, 0, idx] = np.asarray(plydata.elements[0][f"f_dc_{idx}"])

        rest_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        rest_f_names = sorted(rest_f_names, key=lambda x: int(x.split("_")[-1]))
        shN = np.zeros((xyz.shape[0], len(rest_f_names)))
        for idx, attr_name in enumerate(rest_f_names):
            shN[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs)
        shN = einops.rearrange(shN, "N (C M) -> N M C", C=color_channel_dim)

        if self.color_format == "rgbm":
            params.append(
                (
                    "rgb_sh0",
                    torch.nn.Parameter(torch.tensor(sh0[..., :3], dtype=torch.float)),
                    2.5e-3,
                )
            )
            params.append(
                (
                    "rgb_shN",
                    torch.nn.Parameter(torch.tensor(shN[..., :3], dtype=torch.float)),
                    2.5e-3 / 20,
                )
            )
            params.append(
                (
                    "mono_sh0",
                    torch.nn.Parameter(torch.tensor(sh0[..., 3:], dtype=torch.float)),
                    2.5e-3,
                )
            )
            params.append(
                (
                    "mono_shN",
                    torch.nn.Parameter(torch.tensor(shN[..., 3:], dtype=torch.float)),
                    2.5e-3 / 20,
                )
            )
        elif self.color_format == "rgb":
            params.append(
                (
                    "rgb_sh0",
                    torch.nn.Parameter(torch.tensor(sh0, dtype=torch.float)),
                    2.5e-3,
                )
            )
            params.append(
                (
                    "rgb_shN",
                    torch.nn.Parameter(torch.tensor(shN, dtype=torch.float)),
                    2.5e-3 / 20,
                )
            )
        else:
            params.append(
                (
                    "mono_sh0",
                    torch.nn.Parameter(torch.tensor(sh0, dtype=torch.float)),
                    2.5e-3,
                )
            )
            params.append(
                (
                    "mono_shN",
                    torch.nn.Parameter(torch.tensor(shN, dtype=torch.float)),
                    2.5e-3 / 20,
                )
            )

        self._splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)


def color_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def knn(x: torch.Tensor, K: int = 4) -> torch.Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


class VanillaGSplat(L.LightningModule):
    """
    The GS splat implementation using gsplat as the backend.
    """

    _total_time: float = 0

    def __init__(
        self,
        cfg: DictConfig,
        scene_info: SceneInfo,
        train_mode: bool = True,
    ):
        super().__init__()
        self.cfg = cfg

        self.wandb_logging = cfg.wandb.use_wandb

        # Note: Follow the ratio of 1.1. defined in Gsplat.
        # This will only affect the learning rate of position related variables, but not the actual scene scale
        self.scene_scale = scene_info.scene_scale * 1.1 * cfg.opt.global_scale

        self.scene_info = scene_info
        self.train_cameras = scene_info.train_cameras
        self.valid_cameras = scene_info.valid_cameras
        self.test_cameras = scene_info.test_cameras

        bg_color = [1, 1, 1] if cfg.model.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32)

        # in default we will use multiple optimizer for each group
        self.automatic_optimization = False
        self.bilateral_grids = None
        self.bilateral_grid_opt = None

        self.render_output_path = None

        self.model_path = ""

        self.splats = SceneSplats()

    @property
    def total_points(self) -> int:
        return self.splats.means.shape[0]

    @property
    def total_time_s(self) -> float:
        return self._total_time

    def configure_optimizers(self):
        optimizers = list(self.manual_opt.values())

        if self.bilateral_grid_opt is not None:
            optimizers.append(self.bilateral_grid_opt)

        return optimizers

    def save_point_cloud(self, save_dir: str, save_all: bool = False):
        """
        save the point cloud and additional data
        When save_all is enabled, it will save a ply file
        """
        fused_ply_file = f"{save_dir}/point_cloud.ply"
        self.splats.save_ply(fused_ply_file)

    def load_ply(self, ply_path: str):
        """
        load the point cloud from a ply file
        """
        self.splats.load_ply(ply_path)

    def load_bilateral_grid(self, path: str):
        """
        initialize the bilateral grid from a .pth file
        """
        model = torch.load(path)
        n_channel, _, grid_W, grid_X, grid_Y = model["grids"].shape

        if self.bilateral_grids is None:
            self.bilateral_grids = BilateralGrid(
                n_channel,
                grid_X=grid_X,
                grid_Y=grid_Y,
                grid_W=grid_W,
            )

        print(
            f"Initialize the bilateral grid of shape {n_channel}x12x{grid_W}x{grid_X}x{grid_Y}"
        )
        self.bilateral_grids.load_state_dict(model)

    def _adjust_training_iterations(self):

        factor = self.cfg.opt.steps_scaler

        print(f"Adjust all training iterations linearly with a factor of {factor}")

        strategy = self.densification_strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            raise RuntimeError(
                "cannot recognized the strategy or the strategy is not defined yet!"
            )

        if self.cfg.opt.normal_loss:
            self.cfg.opt.normal_start_iter = int(
                self.cfg.opt.normal_start_iter * factor
            )
            print(
                f"Update normal loss starting iterations to: {self.cfg.opt.normal_start_iter}"
            )

        if self.cfg.opt.dist_loss:
            self.cfg.opt.dist_start_iter = int(self.cfg.opt.dist_start_iter * factor)
            print(
                f"Update dist loss starting iterations to: {self.cfg.opt.dist_start_iter}"
            )

        if self.cfg.opt.handle_rolling_shutter:
            self.cfg.opt.handle_rolling_shutter_start_iter = int(
                self.cfg.opt.handle_rolling_shutter_start_iter * factor
            )
            print(
                f"Update dist loss starting iterations to: {self.cfg.opt.dist_start_iter}"
            )

    def _create_strategy(self):
        # Densification Strategy
        if self.cfg.opt.densification_strategy == "default":
            print("Use default GS training strategy")
            opt = self.cfg.opt.gs_default_strategy
            self.densification_strategy = DefaultStrategy(
                prune_opa=opt.prune_opa,
                grow_grad2d=opt.grow_grad2d,
                grow_scale3d=opt.grow_scale3d,
                prune_scale3d=opt.prune_scale3d,
                refine_start_iter=opt.refine_start_iter,
                refine_stop_iter=opt.refine_stop_iter,
                reset_every=opt.reset_every,
                refine_every=opt.refine_every,
            )

            self.densification_strategy.check_sanity(
                self.splats.params, self.manual_opt
            )

            self.strategy_state = self.densification_strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif self.cfg.opt.densification_strategy == "MCMC":
            print("Use MCMC training strategy")
            opt = self.cfg.opt.mcmc_strategy
            self.densification_strategy = MCMCStrategy(
                cap_max=opt.cap_max,
                noise_lr=opt.noise_lr,
                refine_start_iter=opt.refine_start_iter,
                refine_stop_iter=opt.refine_stop_iter,
                min_opacity=opt.min_opacity,
            )
            self.strategy_state = self.densification_strategy.initialize_state()
        else:
            raise NotImplementedError(
                f"Cannot recognize the densification strategy {self.cfg.opt.densification_strategy}"
            )

    def setup(self, stage: str):
        """
        Set up the trainable parameters and optimizer for the GaussianModel
        setup() ensures this is called before configure_optimizers()
        """
        # initilaize the Gaussian splats in scene model
        # assert self.cfg.scene.load_ply == "", "do not support directly loading ply file yet"
        if self.cfg.scene.load_ply != "" and os.path.exists(self.cfg.scene.load_ply):
            print(f"Load Gaussian models from {self.cfg.scene.load_ply}")
            self.load_ply(self.cfg.scene.load_ply)

        cfg = self.cfg
        assert cfg.scene.model_path, "output model path has not been set!"
        self.model_path = cfg.scene.model_path

        use_rgb, use_monochrome = False, False
        for label in self.scene_info.camera_labels:
            if "rgb" in label:
                use_rgb = True
            if "slam" in label:
                use_monochrome = True

        if use_rgb and not use_monochrome:
            self.color_format = "rgb"
            print("Represent Gaussians as RGB channel only")
            self.rgb_image_loss = ImageLoss(color_space="rgb", device=self.device)
            self.mono_image_loss = None
        elif use_rgb and use_monochrome:
            if self.cfg.camera.share_luminance_space:
                self.color_format = "rgb"
                print(
                    "Represent Gaussians as RGB channel only and share lumiance space with monochrome cameras"
                )
                self.cfg.camera.use_bilateral_grid = True
                print("Force to use bilateral grid in this case.")
            else:
                self.color_format = "rgbm"
                print("Represent Gaussians as RGB & monochrome channel")
            self.rgb_image_loss = ImageLoss(color_space="rgb", device=self.device)
            self.mono_image_loss = ImageLoss(
                color_space="luminance", device=self.device
            )
        elif not use_rgb and use_monochrome:
            self.color_format = "m"
            self.rgb_image_loss = None
            self.mono_image_loss = ImageLoss(
                color_space="luminance", device=self.device
            )
            print("Represent Gaussians as monochrome channel only")
        else:
            raise RuntimeError(
                f"Color channel needs to contain either RGB or monochrom channel! Dit not find from {self.scene_info.camera_labels}"
            )

        if stage == "fit":

            print("Output folder: {}".format(self.model_path))
            os.makedirs(self.model_path, exist_ok=True)

            # TODO: Change this to a yaml file
            with open(osp.join(self.model_path, "cfg_args"), "w") as cfg_log_f:
                cfg_log_f.write(str(Namespace(**vars(self.cfg))))

            # create the Gaussians with the optimizer
            self.splats.create_splats_with_optimizers(
                point_cloud=self.scene_info.point_cloud,
                init_opacity=cfg.opt.init_opa,
                init_scale=cfg.opt.init_scale,
                scene_scale=self.scene_scale,
                sh_degree=cfg.opt.sh_degree,
                sparse_grad=cfg.opt.sparse_grad,
                batch_size=cfg.opt.batch_size,
                color_format=self.color_format,
                device=self.device,
            )
            self.manual_opt = self.splats.optimizers

            # create the gsplat training strategy
            self._create_strategy()

            # adjust all the optimization stepping parameters according to the steps scale factor
            self._adjust_training_iterations()

            self.schedulers = [
                # means has a learning rate schedule, that end at 0.01 of the initial value
                torch.optim.lr_scheduler.ExponentialLR(
                    self.manual_opt["means"],
                    gamma=0.01 ** (1.0 / self.cfg.opt.iterations),
                ),
            ]

            if self.cfg.camera.use_bilateral_grid:
                print("Use bilateral grids in reconstructions")
                if self.cfg.camera.view_dependent_transform:
                    print("=>Create a bilateral grid for each individual camera")

                    raise RuntimeError(
                        "We currently don't support per-view dependent bilateral grid yet!"
                    )

                    bilateral_grid_channel = len(self.train_cameras)
                else:
                    print("=>Create a bilateral grid for each RGB camera type")
                    bilateral_grid_channel = 0

                    for camera_label in self.scene_info.camera_labels:
                        if "rgb" in camera_label:
                            bilateral_grid_channel += 1

                    if bilateral_grid_channel > 0:
                        print(
                            f"Create bilateral grids for {bilateral_grid_channel} RGB camera"
                        )
                        self.bilateral_grids = BilateralGrid(
                            bilateral_grid_channel,
                            grid_X=self.cfg.camera.bilateral_grid_shape[0],
                            grid_Y=self.cfg.camera.bilateral_grid_shape[1],
                            grid_W=self.cfg.camera.bilateral_grid_shape[2],
                        ).to(self.device)
                        self.bilateral_grid_opt = torch.optim.Adam(
                            self.bilateral_grids.parameters(),
                            lr=self.cfg.opt.bilateral_grid.lr
                            * math.sqrt(cfg.opt.batch_size),
                            eps=1e-15,
                        )
                        self.schedulers.append(
                            torch.optim.lr_scheduler.ChainedScheduler(
                                [
                                    torch.optim.lr_scheduler.LinearLR(
                                        self.bilateral_grid_opt,
                                        start_factor=0.01,
                                        total_iters=self.cfg.opt.bilateral_grid.linear_lr_iteration,
                                    ),
                                    torch.optim.lr_scheduler.ExponentialLR(
                                        self.bilateral_grid_opt,
                                        gamma=self.cfg.opt.bilateral_grid.exponential_decay_ratio
                                        ** (1.0 / self.cfg.opt.iterations),
                                    ),
                                ]
                            )
                        )
                    else:
                        print(
                            "Did not find RGB cameras. Will not create bilateral grids."
                        )

        else:
            if cfg.render.render_output != "":
                self.render_output_path = cfg.render.render_output
                os.makedirs(self.render_output_path, exist_ok=True)
                print(f"Created the rendering path to {self.render_output_path}")

                if self.cfg.render.render_image:
                    self.image_output_path = f"{self.cfg.render.render_output}/images"
                    os.makedirs(self.image_output_path, exist_ok=True)

                if self.cfg.render.render_depth:
                    self.depth_output_path = f"{self.cfg.render.render_output}/depth"
                    os.makedirs(self.depth_output_path, exist_ok=True)

                    self.depth_viz_output_path = (
                        f"{self.cfg.render.render_output}/depth_viz"
                    )
                    os.makedirs(self.depth_viz_output_path, exist_ok=True)

                if self.cfg.render.render_normal:
                    self.normal_output_path = f"{self.cfg.render.render_output}/normal"
                    os.makedirs(self.normal_output_path, exist_ok=True)

                if self.cfg.render.render_gt:
                    self.gt_output_path = f"{self.cfg.render.render_output}/gt"
                    os.makedirs(self.gt_output_path, exist_ok=True)

        #####
        # todo: Further look into pose optimizer, appearance optimizer,

    def on_train_start(self) -> None:
        """
        Log scene information, the initial point cloud and camera positions
        """
        json_cams = {
            "train": [cam.to_json() for cam in self.train_cameras],
            "valid": [cam.to_json() for cam in self.valid_cameras],
            "test": [cam.to_json() for cam in self.test_cameras],
        }
        with open(osp.join(self.model_path, "cameras.json"), "w") as file:
            json.dump(json_cams, file, indent=4)

        if self.cfg.viewer.use_trainer_viewer:
            self.server = viser.ViserServer(port=self.cfg.viewer.port, verbose=False)
            self.viewer = TrainerViewer3DGS(
                server=self.server,
                model=self,
                train_cameras=self.scene_info.train_cameras,
                valid_cameras=self.scene_info.valid_cameras,
                color_format=self.color_format,
            )
            print("Created training viewer.")
        else:
            self.viewer = None
            print(
                "Viewer was disabled in the configuration. Set viewer.use_trainer_viewer to True if the viewer is needed."
            )

        self.train_iter = 0

    def on_train_end(self) -> None:
        pass

    def _render_motion_array(self, camera: Camera, is_training: bool = False) -> bool:
        """
        Return yes render the camera viewpoint from a set of motion array
        """

        handle_moving_camera = (
            camera.is_moving_camera and self.cfg.opt.handle_rolling_shutter
        )

        if is_training:
            return (
                handle_moving_camera
                and self.train_iter > self.cfg.opt.handle_rolling_shutter_start_iter
            )
        else:
            return handle_moving_camera

    def radiance_finishing(self, linear_image: torch.Tensor) -> torch.Tensor:
        """
        1. transfer the linear radiance image to a (potential) nonlinear image using bilateral grids (if applied)

        2. convert the image space to RGB image if needed.
        """
        if self.bilateral_grids is not None:

            if linear_image.dim() == 3:
                input_image = linear_image[None]
            else:
                input_image = linear_image

            luminance_image = rgb_to_luminance(input_image)
            affine_mats = self.bilateral_grids(luminance_image)  # (b h w 3 4)
            transformed_image = color_affine_transform(
                affine_mats, einops.rearrange(input_image, "b c h w -> b h w c")
            )

            if linear_image.dim() == 3:
                image = einops.rearrange(transformed_image[0], "h w c -> c h w")
            elif linear_image.dim() == 4:
                image = einops.rearrange(transformed_image, "b h w c -> b c h w")
        else:
            image = linear_image

        return image

    def render(
        self,
        camera: Camera,
        scaling_modifier: float = 1.0,
        min_depth: float = 0.0,
        sh_degree: int = 3,
        is_training: bool = False,
        rasterize_mode: Literal["classic", "antialiased"] = "classic",
    ):
        """
        Render the scene from a specified viewpoint_cam
        """
        # calculate the closest rendering distance.
        # If there is no depth information available, set it to zero and will skip pre-filtering.
        min_depth = 0
        if self.cfg.model.use_3d_smooth_filter and is_training:
            if camera.sparse_depth is not None:
                min_depth = camera.sparse_depth.to(self.device).min()
            elif camera.render_depth is not None:
                min_depth = camera.render_depth_min

        if self._render_motion_array(camera, is_training=is_training):
            # will render a batch of image poses and merge
            w2c_array_tensor = camera.motion_w2c_array_tensor
            # number of rolling shutter sample, and exposure sample respectively
            N_rolling_shutter_samples, N_exposure_samples = w2c_array_tensor.shape[:2]
            viewmat_w2c = einops.rearrange(
                w2c_array_tensor, "n_rs n_exp a b -> (n_rs n_exp) a b"
            ).to(self.device)
            intrinsics = (
                camera.intrinsic[None]
                .repeat(viewmat_w2c.shape[0], 1, 1)
                .to(self.device)
            )  # (C, 3, 3)
        else:
            viewmat_w2c = camera.w2c_44[None].to(self.device)  # (C, 4, 4)
            intrinsics = camera.intrinsic[None].to(self.device)  # (C, 3, 3)

        render_pkg = self(
            intrinsics=intrinsics,
            viewmat_w2c=viewmat_w2c,
            render_height=int(camera.image_height),
            render_width=int(camera.image_width),
            scaling_modifier=scaling_modifier,
            min_depth=min_depth,
            is_rgb=camera.is_rgb,
            sh_degree_to_use=sh_degree,
            camera_model=camera.camera_projection_model_gsplat,
            rasterize_mode=rasterize_mode,
        )

        if self._render_motion_array(camera, is_training=is_training):

            rs_render = einops.rearrange(
                render_pkg["render"],
                "(n_rs n_exp) c h w -> n_rs n_exp c h w",
                n_rs=N_rolling_shutter_samples,
                n_exp=N_exposure_samples,
            )
            rs_depth = einops.rearrange(
                render_pkg["depth"],
                "(n_rs n_exp) h w -> n_rs n_exp h w",
                n_rs=N_rolling_shutter_samples,
                n_exp=N_exposure_samples,
            )
            rs_alphas = einops.rearrange(
                render_pkg["alphas"],
                "(n_rs n_exp) c h w -> n_rs n_exp c h w",
                n_rs=N_rolling_shutter_samples,
                n_exp=N_exposure_samples,
            )

            # average exposure samples
            rs_render = torch.mean(rs_render, dim=1, keepdim=False)
            rs_depth = torch.mean(rs_depth, dim=1, keepdim=False)
            rs_alphas = torch.mean(rs_alphas, dim=1, keepdim=False)

            if (
                N_rolling_shutter_samples > 1
            ):  # gather rolling shutter samples according to the mask
                rolling_shutter_mask = camera.rolling_shutter_index_image.to(
                    self.device
                )

                rs_render = torch.gather(
                    rs_render,
                    0,
                    einops.repeat(rolling_shutter_mask, "h w -> b c h w", b=1, c=3),
                )
                rs_depth = torch.gather(
                    rs_depth,
                    0,
                    einops.repeat(rolling_shutter_mask, "h w -> b h w", b=1),
                )
                rs_alphas = torch.gather(
                    rs_alphas,
                    0,
                    einops.repeat(rolling_shutter_mask, "h w -> b c h w", b=1, c=1),
                )

            render_pkg["render"] = rs_render
            render_pkg["depth"] = rs_depth
            render_pkg["alphas"] = rs_alphas

        return render_pkg

    def _apply_3d_depth_based_filter(
        self, min_depth: float, focals: torch.Tensor, kernel_size_3d=0.2
    ):
        # This is a view-dependent 3D Gaussian filter that differs from the 3D smooth filter in Mip-splatting.
        # We use a single filter based on the minimum depth observed from the corresponding camera, instead of using a point dependent 3D filter.
        # And we don't perform such update when the whole Gaussians are saved into ply files, so the original opacities are preserved
        # instead of being scaled in undesired way.
        # We believe it makes more sense for free-view trajectory, where the 3D point-based filter could simple degenerate close to 0 for each.
        # See reported as issue of opacity scaling in Mip-splatting in: https://github.com/autonomousvision/mip-splatting/issues/48)

        scales = self.splats.activated_scales  # [N, 3]
        opacities = self.splats.activated_opacities  # [N,]

        scales_square = torch.square(scales)  # [N, 3]
        det1 = scales_square.prod(dim=1)
        # choose the min filter (max sampling rate) among all
        filter3d = torch.max(min_depth / focals * (kernel_size_3d**0.5), dim=0)[
            0
        ]  # (1)
        filter3d = torch.broadcast_to(filter3d, (1, 3))  # (1, 3)

        scales_after_square = scales_square + torch.square(filter3d)  # [N, 3]
        det2 = scales_after_square.prod(dim=1)  # [N,]
        coef = torch.sqrt(det1 / det2 + 1e-7)  # [N,]

        # This is the coef in eq. (7) of Mip-splatting, of the combined Gaussian filter, which is absorbed into the opacity filter.
        opacities = opacities * coef

        scales = torch.square(scales) + torch.square(filter3d)  # [N, 3]
        scales = torch.sqrt(scales)
        return scales, opacities

    def forward(
        self,
        intrinsics: torch.Tensor,
        viewmat_w2c: torch.Tensor,
        render_height: int,
        render_width: int,
        scaling_modifier: float = 1.0,
        min_depth: float = 0.0,
        is_rgb: bool = True,
        sh_degree_to_use: int = 3,
        camera_model: Literal["pinhole", "fisheye"] = "pinhole",
        rasterize_mode: Literal["classic", "antialiased"] = "classic",
    ):
        means = self.splats.means  # [N, 3]
        # rasterization does normalization internally
        quats = self.splats.quats  # [N, 4]

        if is_rgb or (not self.splats.has_monochrome_channel):
            colors = self.splats.rgb_feature
        else:
            colors = einops.repeat(
                self.splats.monochrome_feature, "b c (h 1) -> b c (h 3)", h=1
            )

        # vanilla Gaussian splatting activation

        if min_depth > 0 and self.cfg.model.use_3d_smooth_filter:
            kernel_size_2d = 0.1  # the screen dilation kernel size in mip-splatting
            scales, opacities = self._apply_3d_depth_based_filter(
                min_depth, focals=intrinsics[:, 0, 0]
            )
        else:
            scales = self.splats.activated_scales  # [N, 3]
            opacities = self.splats.activated_opacities  # [N,]
            kernel_size_2d = 0.3  # the default screen dilation kernel size in GS

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales * scaling_modifier,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat_w2c.to(self.device),  # [C, 4, 4]
            Ks=intrinsics.to(self.device),  # [C, 3, 3]
            eps2d=kernel_size_2d,
            width=render_width,
            height=render_height,
            packed=self.cfg.opt.packed,
            absgrad=self.cfg.opt.absgrad,
            sparse_grad=self.cfg.opt.sparse_grad,
            rasterize_mode=rasterize_mode,
            sh_degree=sh_degree_to_use,
            camera_model=camera_model,
            render_mode="RGB+ED",
        )

        image = einops.rearrange(render_colors[..., :3], "b h w c -> b c h w")
        alphas = einops.rearrange(render_alphas, "b h w c -> b c h w")
        depth = render_colors[..., 3]  # (B, H, W)

        if not is_rgb:
            if self.splats.has_monochrome_channel:
                image = image[:, 0:1]  # choose only one channel for monochrome images
            else:
                image = rgb_to_luminance(
                    image
                )  # render the monochrome (luminance) from RGB image

        return {
            "render": image,
            "depth": depth,
            "alphas": alphas,
            "info": info,
        }

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        """
        Update Gaussians learning rate and SH degree
        """
        self.global_tic = time.time()

        self.train_iter += 1

        if self.viewer:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()

    @property
    def sh_degree_to_use(self) -> int:
        return min(
            self.train_iter // self.cfg.opt.sh_degree_interval, self.cfg.model.sh_degree
        )

    @property
    def rasterization_mode(self) -> str:
        return "antialiased" if self.cfg.opt.antialiased else "classic"

    def training_step(self, batch, batch_idx):

        gt_image = batch["image"].to(self.device)  # (B, C, H, W)
        batch_size, color_channel, image_height, image_width = gt_image.shape

        # We supported multi-batch training initially but has roll-back to single-image training.
        # There was no obvious benefit to use multi-batch training and it complicates the pipeline by quite a bit.
        assert batch_size == 1, "Disabled multi-batch training. Use batch=1 instead."
        image_id = batch["idx"].item()
        train_cam = self.train_cameras[image_id]
        camera_name = train_cam.camera_name

        renders = self.render(
            train_cam,
            sh_degree=self.sh_degree_to_use,
            rasterize_mode=self.rasterization_mode,
            is_training=True,
        )
        train_cam.render_depth_min = renders["depth"].min().item()

        # image = renders["render"]
        # add irradiance compensation (exposure or so)
        # image = image * batch['irradiance_multiplier'].float()
        image = train_cam.expose_image(irradiance=renders["render"], clamp=False)

        # if train_cam.is_rgb:
        # image = apply_aria_rgb_gamma(image)
        # image = self.radiance_finishing(image)

        # valid_mask = (batch['irradiance_multiplier'] > 0).float()
        valid_mask = train_cam.valid_mask

        use_mask_loss = False
        if self.cfg.scene.mask_rendering:
            # enforce mask rendering. This will set the background to completely black.
            # The Gaussians in the background will thus be eliminated.
            assert (
                "mask" in batch.keys()
            ), "masked rendering require to have mask as input"
            mask = batch["mask"].to(self.device)  # (B, 1, H, W)
            # currently set to a random color does not work as good as setting it to black
            bmask = (mask < 0.5).expand_as(gt_image)
            gt_image[bmask] = 0
            image[bmask] = 0

            use_mask_loss = True
        else:
            use_mask_loss = False

        self.num_train_rays_per_step = 3 * image_height * image_width

        for i, opt in enumerate(self.optimizers()):
            opt.zero_grad(set_to_none=True)
            # the following is to fix multiple opt steps added to global_step:
            # ref: https://github.com/Lightning-AI/pytorch-lightning/issues/17958#issuecomment-1979571983
            if i + 1 < len(self.optimizers()):
                opt._on_before_step = lambda: self.trainer.profiler.start(
                    "optimizer_step"
                )
                opt._on_after_step = lambda: self.trainer.profiler.stop(
                    "optimizer_step"
                )

        pixel_loss_type = self.cfg.opt.pixel_loss

        train_losses_categories = [pixel_loss_type, "dssim", "psnr"]

        if self.cfg.opt.l1_grad:
            train_losses_categories.append("l1_grad")

        if train_cam.is_rgb:
            image_loss_func = self.rgb_image_loss
            assert (
                image_loss_func is not None
            ), "color image loss function cannot be None for RGB camera!"
        else:
            image_loss_func = self.mono_image_loss
            assert (
                image_loss_func is not None
            ), "mono image loss function cannot be None for Monochrome camera!"

        # clamp radiance value to the observed dynamic range
        image = image.clamp(0, 1)
        image_losses = image_loss_func(
            image, gt_image, losses=train_losses_categories, valid_mask=valid_mask
        )

        loss = (
            self.cfg.opt.pixel_lambda * image_losses[pixel_loss_type]
            + 0.2 * image_losses["dssim"]
        )
        if "l1_grad" in train_losses_categories:
            loss += self.cfg.opt.l1_grad_lamda * image_losses["l1_grad"]

        logs = {f"train/{camera_name}/image_loss_total": loss}
        for loss_type in train_losses_categories:
            logs[f"train/{camera_name}/{loss_type}"] = image_losses[loss_type]

        if self.cfg.opt.depth_loss:
            depthloss = calculcate_inverse_depth_loss(
                render_depth=renders["depth"] / self.scene_scale,
                sparse_point2d=batch["sparse_point2d"].to(self.device),
                sparse_inv_depth=batch["sparse_inv_depth"].to(self.device),
                sparse_inv_distance_std=batch["sparse_inv_distance_std"].to(
                    self.device
                ),
                losses=["huber"],
                huber_delta=0.5,
            )
            depth_lambda = self.cfg.opt.depth_lambda
            loss += depth_lambda * depthloss["huber"]
            logs["train/depth_loss_huber"] = depthloss

        if self.cfg.opt.opacity_reg > 0:
            loss_opacity_reg = torch.abs(self.splats.activated_opacities).mean()
            loss += self.cfg.opt.opacity_reg * loss_opacity_reg
            logs["train/opacity_reg_mean"] = loss_opacity_reg

        if self.cfg.opt.scale_reg > 0:
            loss_scale_reg = torch.abs(self.splats.activated_scales).mean()
            loss += self.cfg.opt.scale_reg * loss_scale_reg
            logs["train/scale_reg_mean"] = loss_scale_reg

        if self.cfg.camera.use_bilateral_grid and self.bilateral_grids is not None:
            tvloss = total_variation_loss(self.bilateral_grids.grids)
            logs["train/tv_bilateral_grids"] = tvloss
            loss += tvloss

        if use_mask_loss:
            # enforce the background alphas to be zero
            maskloss = renders["alphas"][mask < 0.5].mean()
            logs["train/mask_loss"] = maskloss
            loss += maskloss

        self.log_dict(
            logs, on_step=True, on_epoch=True, logger=True, batch_size=batch_size
        )

        # update the training strategy
        self.densification_strategy.step_pre_backward(
            params=self.splats.params,
            optimizers=self.manual_opt,
            state=self.strategy_state,
            step=self.train_iter,
            info=renders["info"],
        )

        to_return = {
            "loss": loss,
            "render": image,
        }

        # in manual optimization mode
        self.manual_backward(loss)

        if isinstance(self.densification_strategy, DefaultStrategy):
            self.densification_strategy.step_post_backward(
                params=self.splats.params,
                optimizers=self.manual_opt,
                state=self.strategy_state,
                step=self.train_iter,
                info=renders["info"],
                packed=self.cfg.opt.packed,
            )
        elif isinstance(self.densification_strategy, MCMCStrategy):
            self.densification_strategy.step_post_backward(
                params=self.splats.params,
                optimizers=self.manual_opt,
                state=self.strategy_state,
                step=self.train_iter,
                info=renders["info"],
                lr=self.schedulers[0].get_last_lr()[0],
            )

        for i, opt in enumerate(self.optimizers()):
            opt.step()

        for scheduler in self.schedulers:
            scheduler.step()

        return to_return

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        # update the viewer
        train_time_per_batch = time.time() - self.global_tic
        self._total_time += train_time_per_batch

        if self.viewer:
            self.viewer.lock.release()
            num_train_steps_per_sec = 1.0 / train_time_per_batch
            num_train_rays_per_sec = (
                self.num_train_rays_per_step * num_train_steps_per_sec
            )
            # Update the viewer state.
            self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            # Update the scene.
            self.viewer.update(self.train_iter, self.num_train_rays_per_step)

        if self.wandb_logging:
            self.log_dict(
                {"gaussians/total_points": float(self.total_points)},
                on_step=True,
                on_epoch=False,
                logger=True,
            )

    def validation_step(self, batch, batch_idx):

        subset = "valid"
        camera = self.valid_cameras[batch["idx"].item()]
        camera_name = camera.camera_name

        gt_image = batch["image"].to(self.device)[0]  # (3, H, W)
        gt_image = torch.clamp(gt_image, 0.0, 1.0)

        image_height, image_width = gt_image.shape[-2:]

        with torch.no_grad():
            renders = self.render(camera)

        image = renders["render"][0]
        camera.render_depth_min = renders["depth"].min().item()
        image = camera.expose_image(irradiance=image, clamp=False)

        if camera.is_rgb:
            image = self.radiance_finishing(image)

        image = image.clamp(0, 1)
        valid_mask = camera.valid_mask
        if "mask" in batch.keys():  # enforce mask rendering
            # assert "mask" in batch.keys(), "masked rendering require to have mask as input"
            mask = batch["mask"].to(self.device)[0]  # (1, H, W)
            # currently set to a random color does not work as good as setting it to black
            bmask = (mask < 0.5).expand_as(gt_image)
            gt_image[bmask] = 0
            image[bmask] = 0
            # alphas = renders["alphas"].expand_as(image)
            # image = image + random_color * (1.0 - alphas)

            if valid_mask is None:
                valid_mask = mask
            else:
                valid_mask = mask * valid_mask.to(self.device)
        else:
            bmask = None

        self.num_train_rays_per_step = 3 * image_height * image_width

        log_image_indices = np.linspace(
            0, len(self.trainer.val_dataloaders) - 1, 20, dtype=int
        )

        if self.wandb_logging and batch_idx in log_image_indices:
            image = torch.clamp(image, 0.0, 1.0)
            image_wandb = convert_image_tensor2array(image)
            gt_image_wandb = convert_image_tensor2array(gt_image)

            # log depth
            depth = renders["depth"]
            min_d, max_d = self.cfg.render.depth_min, self.cfg.render.depth_max
            depth_normalized = (depth.clamp(min_d, max_d) - min_d) / (max_d - min_d)
            depth_vis = apply_turbo_colormap(depth_normalized[0])
            depth_vis_array = convert_image_tensor2array(depth_vis)

            # log normal
            if "normals" in renders:
                normal = renders["normals"][0]
            else:
                normal = depth_to_normal(camera, depth[0])
            normal_vis = (normal + 1) / 2
            normal_vis_array = convert_image_tensor2array(normal_vis)

            if bmask is not None:
                bmask_array = einops.rearrange(bmask, "c h w -> h w c").cpu().numpy()
                depth_vis_array[bmask_array] = 0
                normal_vis_array[bmask_array] = 0

            image_wandb = np.concatenate(
                (gt_image_wandb, image_wandb, depth_vis_array, normal_vis_array), axis=1
            )
            caption = "GT - Render - Depth - Normal"

            if image_wandb.shape[0] > 800:
                # resize the logged output to size with maximum 800 pixels in height. '
                h, w = image_wandb.shape[:2]
                image_wandb = cv2.resize(image_wandb, (int(w * 800 / h), 800))

            image_wandb = wandb.Image(image_wandb, caption=caption)
            log_name = subset + "_view/{}".format(camera.image_name)
            wandb.log({log_name: image_wandb}, commit=False)

        ###################################################################
        # Evaluate the photometric losses
        if camera.is_rgb:
            image_loss_func = self.rgb_image_loss
            assert (
                image_loss_func is not None
            ), "color image loss function cannot be None for RGB camera!"
        else:
            image_loss_func = self.mono_image_loss
            assert (
                image_loss_func is not None
            ), "mono image loss function cannot be None for Monochrome camera!"

        eval_loss_categories = ["l1", "ssim", "psnr", "lpips"]
        image_losses = image_loss_func(
            image[None],
            gt_image[None],
            losses=eval_loss_categories,
            valid_mask=valid_mask,
        )
        logs = {}
        for loss_type in eval_loss_categories:
            logs[f"valid/{camera_name}/{loss_type}"] = image_losses[loss_type]

        # if masked_valid_mask is not None:
        #     masked_image_losses =  self.image_loss(image[None], gt_image[None], losses=eval_loss_categories, valid_mask=masked_valid_mask)
        #     for loss_type in eval_loss_categories:
        #         logs[f"valid/{camera_name}/masked_{loss_type}"] = image_losses[masked_image_losses]

        ####################################################################
        # Evaluate the geometric losses (if depth supervision is available)

        if "sparse_point2d" in batch.keys():
            depthloss = calculcate_inverse_depth_loss(
                render_depth=renders["depth"] / self.scene_scale,
                sparse_point2d=batch["sparse_point2d"].to(self.device),
                sparse_inv_depth=batch["sparse_inv_depth"].to(self.device),
                sparse_inv_distance_std=None,  # batch["sparse_inv_distance_std"].to(self.device),
                losses=["l1"],
            )
            logs["valid/inv_depth_l1"] = depthloss["l1"]

        self.log_dict(logs, on_step=False, on_epoch=True, logger=True, batch_size=1)

    def on_test_epoch_start(self) -> None:
        self.test_logs = {
            "ssim": {},
            "psnr": {},
            "lpips": {},
            "image": {},
            "depth": {},
            "normal": {},
        }

        # from utils.tsdf_fusion import TSDFVolume
        # self.tsdf_volume = TSDFVolume()

    def on_test_epoch_end(self) -> None:
        # save the final point cloud

        if self.cfg.scene.save_ply:
            save_dir = os.path.join(
                self.model_path, "point_cloud", f"iteration_{self.cfg.opt.iterations}"
            )
            os.makedirs(save_dir, exist_ok=True)
            self.save_point_cloud(save_dir)
            print(f"save the final point cloud to {save_dir}")

        # save_dir = os.path.join(self.model_path, "mesh")
        # os.makedirs(save_dir, exist_ok=True)
        # self.tsdf_volume.visualize_mesh()
        # self.tsdf_volume.extract_mesh(f"{save_dir}/tsdf.ply")

        # save the bilateral grid
        if self.bilateral_grids is not None:
            torch.save(
                self.bilateral_grids.state_dict(),
                f"{self.model_path}/bilateral_grid.pth",
            )

        # write the final evaluation table to a json path
        with open(osp.join(self.model_path, "test_logs.json"), "w") as file:
            json.dump(self.test_logs, file, indent=4)

        # if ground truth exist, perform holistic evaluation
        gt_json = Path(self.cfg.scene.source_path) / self.cfg.scene.data_format
        test_output_json = Path(self.cfg.scene.model_path) / "test_logs.json"

        try:
            eval_output_full = run_eval(test_output_json, gt_json)

            if eval_output_full is not None:
                self.log_dict(
                    {
                        "test/camera-rgb/depth_scale_invariant_eval": eval_output_full[
                            "depth_scale_invariant_eval"
                        ]["average"],
                        "test/camera-rgb/normal_cosine_l1_distance_eval": eval_output_full[
                            "normal_cosine_l1_distance_eval"
                        ][
                            "average"
                        ],
                    },
                    logger=True,
                )

                # append the results to test_logs.json
                with open(
                    Path(self.cfg.scene.model_path) / "test_logs_with_geometry.json",
                    "w",
                ) as f:
                    json.dump(eval_output_full, f, indent=True)
            else:
                print(
                    "There is no additional ground truth files. Skip full evaluation."
                )
        except:
            print("Have issues to execute full evaluation. Skip!")

        if self.cfg.viewer.use_trainer_viewer:
            print(
                "All test finished. Trainer Viewer is still running... Ctrl+C to exit."
            )
            time.sleep(1000000)

    def test_step(self, batch, batch_idx):
        """
        The test step will collect more information during the evaluation
        """
        subset = batch["subset"][0]
        camera = self.test_cameras[batch["idx"].item()]
        camera_name = camera.camera_name

        with torch.no_grad():
            render_pkg = self.render(camera)

        # currently assumes only 1 image rendered per batch
        image = render_pkg["render"][0]

        if "image" in batch.keys():
            gt_image = batch["image"].to(self.device)[0]  # (3, H, W)
            gt_image = torch.clamp(gt_image, 0.0, 1.0)

            calculate_loss = True
        else:
            gt_image = None
            calculate_loss = False

        image = camera.expose_image(irradiance=image, clamp=False)

        if camera.is_rgb:
            image = self.radiance_finishing(image)

        image = torch.clamp(image, 0.0, 1.0)

        # self.tsdf_volume.integrate(
        #     depth=render_pkg['depth'],
        #     color=image,
        #     camera = camera
        # )

        valid_mask = camera.valid_mask
        if "mask" in batch.keys():  # enforce mask rendering
            # assert "mask" in batch.keys(), "masked rendering require to have mask as input"
            mask = batch["mask"].to(self.device)[0]  # (1, H, W)
            # currently set to a random color does not work as good as setting it to black
            bmask = (mask < 0.5).expand_as(gt_image)
            gt_image[bmask] = 0
            image[bmask] = 0
            # alphas = renders["alphas"].expand_as(image)
            # image = image + random_color * (1.0 - alphas)

            if valid_mask is None:
                valid_mask = mask
            else:
                valid_mask = mask * valid_mask.to(self.device)

        # image_name = f"{camera.time_s:04f}_{camera.image_name}"
        image_name = camera.image_name
        log_key = f"{camera_name}/{image_name}"

        # render the test image
        if self.render_output_path != "" and self.cfg.render.render_image:
            image_array = convert_image_tensor2array(image)
            output_path = f"{self.image_output_path}/{image_name}"
            save_image(image_array, output_path)
            self.test_logs["image"][log_key] = output_path

        # we save the original depth instead of the visualized depth
        if self.render_output_path != "" and self.cfg.render.render_depth:
            depth = render_pkg["depth"]

            min_d, max_d = (
                depth.min(),
                depth.max(),
            )  # self.cfg.render.depth_min, self.cfg.render.depth_max
            depth_normalized = (depth.clamp(min_d, max_d) - min_d) / (max_d - min_d)

            depth_array = (depth_normalized[0].cpu().numpy() * 255).astype(np.uint8)
            output_path = f"{self.depth_output_path}/{image_name}"
            save_image(depth_array, output_path)
            self.test_logs["depth"][log_key] = output_path

            # depth_viz_array
            if valid_mask is not None:
                depth_normalized[(valid_mask < 0.5).expand_as(depth_normalized)] = min_d
            # depth_vis = apply_turbo_colormap(depth_normalized[0])
            # depth_vis_array = convert_image_tensor2array(depth_vis.clamp(0, 1))
            # viz_output_path = f"{self.depth_viz_output_path}/{image_name}"
            # save_image(depth_vis_array, viz_output_path)

        if self.render_output_path != "" and self.cfg.render.render_normal:
            if "normals" in render_pkg:
                normal = render_pkg["normals"][0]
            else:
                assert (
                    self.cfg.render.render_depth
                ), "estimate normal will require depth be renderered as well!"
                normal = depth_to_normal(camera, depth[0])

            if self.cfg.render.viewspace_normal:
                # The encoded normal is supposed to be in the world space
                # Transform it to the image space for evaluation purpose
                R_c2w = camera.c2w_44_opengl[:3, :3].to(normal.device)
                R_w2c = torch.linalg.inv(R_c2w)
                normal = torch.einsum("ij, hwj->hwi", R_w2c, normal)

            normal_vis = (normal + 1) / 2
            normal_vis_array = (normal_vis.cpu().numpy() * 255).astype(np.uint8)
            output_path = f"{self.normal_output_path}/{image_name}"
            save_image(normal_vis_array, output_path)
            self.test_logs["normal"][log_key] = output_path

        if self.render_output_path and self.cfg.render.render_gt:
            # create a symbolic link
            output_path = Path(f"{self.gt_output_path}/{image_name}")
            if not output_path.exists():
                output_path.symlink_to(camera.image_path)

        if calculate_loss:

            if camera.is_rgb:
                image_loss_func = self.rgb_image_loss
                assert (
                    image_loss_func is not None
                ), "color image loss function cannot be None for RGB camera!"
            else:
                image_loss_func = self.mono_image_loss
                assert (
                    image_loss_func is not None
                ), "mono image loss function cannot be None for Monochrome camera!"

            eval_loss_categories = ["ssim", "psnr", "lpips"]
            image_losses = image_loss_func(
                image[None],
                gt_image[None],
                losses=eval_loss_categories,
                valid_mask=valid_mask,
            )
            logs = {}
            for loss_type in eval_loss_categories:
                logs[f"test/{camera_name}/{loss_type}"] = image_losses[loss_type]
                self.test_logs[loss_type][log_key] = image_losses[loss_type].item()
            self.log_dict(logs, on_step=False, on_epoch=True, logger=True, batch_size=1)
