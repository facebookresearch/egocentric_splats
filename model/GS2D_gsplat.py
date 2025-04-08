import einops
import torch

from gsplat.rendering import rasterization_2dgs
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from omegaconf import DictConfig

from scene.cameras import Camera
from scene.dataset_readers import SceneInfo

from typing_extensions import Literal

from .bilateral_grid import total_variation_loss

from .loss import calculcate_inverse_depth_loss

from .vanilla_gsplat import VanillaGSplat


def get_valid_mask(data, threshold=1.0):
    valid_mask = (~torch.isnan(data).any(dim=-1)) * (~torch.isinf(data).any(dim=-1))
    valid_mask *= (torch.abs(data) < threshold).any(dim=-1)
    return valid_mask.detach()


class Gaussians2D(VanillaGSplat):
    """
    2D Gaussian Splatting for Geometrically Accurate Radiance Fields,
    implemented following the gsplat 2D GS implementation.
    https://arxiv.org/pdf/2403.17888
    """

    def __init__(
        self,
        cfg: DictConfig,
        scene_info: SceneInfo,
        train_mode: bool = True,
    ):
        super().__init__(cfg, scene_info, train_mode)

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
        Render the scene from a specified camera
        """
        # calculate the closest rendering distance.
        # If there is no depth information available, set it to zero and will skip pre-filtering.
        assert (
            camera.camera_projection_model_gsplat == "pinhole"
        ), "2dgs will only support pinhole camera model"

        min_depth = 0
        if self.cfg.model.use_3d_smooth_filter:
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
        )

        if self._render_motion_array(camera, is_training=is_training):

            if N_exposure_samples > 1:
                # todo: the "normals_from_depth" does not return batch rendering. Need to chase further
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
                    "(n_rs n_exp) h w c -> n_rs n_exp h w c",
                    n_rs=N_rolling_shutter_samples,
                    n_exp=N_exposure_samples,
                )
                rs_normals = einops.rearrange(
                    render_pkg["normals"],
                    "(n_rs n_exp) h w c -> n_rs n_exp h w c",
                    n_rs=N_rolling_shutter_samples,
                    n_exp=N_exposure_samples,
                )
                rs_normals_from_depth = einops.rearrange(
                    render_pkg["normals_from_depth"],
                    "(n_rs n_exp) h w c -> n_rs n_exp h w c",
                    n_rs=N_rolling_shutter_samples,
                    n_exp=N_exposure_samples,
                )
                rs_render_distort = einops.rearrange(
                    render_pkg["render_distort"],
                    "(n_rs n_exp) h w c -> n_rs n_exp h w c",
                    n_rs=N_rolling_shutter_samples,
                    n_exp=N_exposure_samples,
                )

                # average exposure samples
                rs_render = torch.mean(rs_render, dim=1, keepdim=False)
                rs_depth = torch.mean(rs_depth, dim=1, keepdim=False)
                rs_alphas = torch.mean(rs_alphas, dim=1, keepdim=False)
                rs_normals = torch.mean(rs_normals, dim=1, keepdim=False)
                rs_normals_from_depth = torch.mean(
                    rs_normals_from_depth, dim=1, keepdim=False
                )
                rs_render_distort = torch.mean(rs_render_distort, dim=1, keepdim=False)
            else:
                rs_render = render_pkg["render"]
                rs_depth = render_pkg["depth"]
                rs_alphas = render_pkg["alphas"]
                rs_normals = render_pkg["normals"]
                rs_normals_from_depth = render_pkg["normals_from_depth"]
                rs_render_distort = render_pkg["render_distort"]

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
                    einops.repeat(rolling_shutter_mask, "h w -> b h w c", b=1, c=1),
                )
                rs_normals = torch.gather(
                    rs_normals,
                    0,
                    einops.repeat(rolling_shutter_mask, "h w -> b h w c", b=1, c=1),
                )
                rs_normals_from_depth = torch.gather(
                    rs_normals_from_depth,
                    0,
                    einops.repeat(rolling_shutter_mask, "h w -> b h w c", b=1, c=1),
                )
                rs_render_distort = torch.gather(
                    rs_render_distort,
                    0,
                    einops.repeat(rolling_shutter_mask, "h w -> b h w c", b=1, c=1),
                )

            render_pkg["render"] = rs_render
            render_pkg["depth"] = rs_depth
            render_pkg["alphas"] = rs_alphas
            render_pkg["normals"] = rs_normals
            render_pkg["normals_from_depth"] = rs_normals_from_depth
            render_pkg["render_distort"] = rs_render_distort

        return render_pkg

    def _create_strategy(self):
        """
        Create the densification Strategy. There are a few hyper parameters in 2D GS different from the vanilla 3D GS.
        """
        if self.cfg.opt.densification_strategy == "default":
            print("Use default GS training strategy")
            self.densification_strategy = DefaultStrategy(
                key_for_gradient="gradient_2dgs"
            )
            self.densification_strategy.check_sanity(
                self.splats.params, self.manual_opt
            )
            self.strategy_state = self.densification_strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif self.cfg.opt.densification_strategy == "MCMC":
            print("Use MCMC training strategy")
            self.densification_strategy = MCMCStrategy()
            self.strategy_state = self.densification_strategy.initialize_state()
        else:
            raise NotImplementedError(
                f"Cannot recognize the densification strategy {self.cfg.opt.densification_strategy}"
            )

    def forward(
        self,  # camera: Camera,
        intrinsics: torch.Tensor,
        viewmat_w2c: torch.Tensor,
        render_height: int,
        render_width: int,
        scaling_modifier: float = 1.0,
        min_depth: float = 0.0,
        is_rgb: bool = True,
        sh_degree_to_use: int = 3,
    ):
        """
        Render the scene given a batch of viewpoints and corresponding extrinsics (view matrix as world to camera)

        * scaling_modifier: adjust the scale of Gaussians. Default to be 1.0.
        * rasterize_mode: using classic rasterization or antialiased rasetrization,
            which is the 2D Mip Filter in Mip-Splatting.
            We will use antialiased filter in rendering, but classic in training.
        """
        means = self.splats.means  # [N, 3]
        quats = self.splats.quats  # [N, 4]

        scales = torch.exp(self.splats.scales) * scaling_modifier  # [N, 3]
        opacities = torch.sigmoid(self.splats.opacities).squeeze(-1)  # [N,]

        if is_rgb:
            colors = self.splats.rgb_feature
        else:
            colors = einops.repeat(
                self.splats.monochrome_feature, "b c (h 1) -> b c (h 3)", h=1
            )

        if min_depth > 0 and self.cfg.model.use_3d_smooth_filter:
            kernel_size_2d = 0.1  # the screen dilation kernel size in mip-splatting
            scales, opacities = self._apply_3d_depth_based_filter(
                min_depth, focals=intrinsics[:, 0, 0]
            )
        else:
            kernel_size_2d = 0.3  # the default screen dilation kernel size in GS

        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat_w2c.to(self.device),  # [C, 4, 4]
            eps2d=kernel_size_2d,
            Ks=intrinsics.to(self.device),  # [C, 3, 3]
            width=render_width,
            height=render_height,
            render_mode="RGB+ED",
            packed=False,
            absgrad=self.cfg.opt.absgrad,
            sh_degree=sh_degree_to_use,
        )

        image = einops.rearrange(render_colors[..., :3], "b h w c -> b c h w")
        depth = render_colors[..., 3]

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
            "alphas": render_alphas,
            "normals": render_normals,
            "normals_from_depth": normals_from_depth,
            "render_distort": render_distort,
            "info": info,
        }

    def training_step(self, batch, batch_idx):

        gt_image = batch["image"].to(self.device)  # (B, C, H, W)

        batch_size, color_channel, image_height, image_width = gt_image.shape

        subset = "train"

        # We supported multi-batch training initially but has roll-back to single-image training.
        # There was no obvious benefit to use multi-batch training and it complicates the pipeline by quite a bit.
        assert batch_size == 1, "Disabled multi-batch training. Use batch=1 instead."
        image_id = batch["idx"].item()
        train_cam = self.train_cameras[image_id]
        camera_name = train_cam.camera_name

        try:
            renders = self.render(
                train_cam, sh_degree=self.sh_degree_to_use, is_training=True
            )
        except:
            print(f"encounter issue to render training camera for {image_id}.")
            return {
                "loss": loss,
                "render": image,
            }
            # change to a different camera id

        train_cam.render_depth_min = renders["depth"].min().item()

        image = train_cam.expose_image(irradiance=renders["render"], clamp=False)
        # image = renders["render"]
        # image = image * batch['irradiance_multiplier']
        # image_valid_mask = (batch['irradiance_multiplier'] >0).float()

        valid_mask = train_cam.valid_mask
        use_mask_loss = False
        if self.cfg.scene.mask_rendering:  # enforce mask rendering
            # assert "mask" in batch.keys(), "masked rendering require to have mask as input"
            mask = batch["mask"].to(self.device)
            bmask = (mask < 0.5).expand_as(image)  # (B, 3, H, W)
            gt_image[bmask] = 0
            image[bmask] = 0

            use_mask_loss = True

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

        # clamp radiance value to the observed dynamic range
        pixel_loss_type = self.cfg.opt.pixel_loss
        train_losses_categories = [pixel_loss_type, "dssim", "psnr"]

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

        # add a spase depth loss using the spase semi-dense points observed in each image
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

        # add the normal consistentcy loss
        if (
            self.cfg.opt.normal_loss
            and self.train_iter > self.cfg.opt.normal_start_iter
        ):
            # There are a few issues from the 2d GS predicted normal. They are not normalized and may have NaN values inside.
            # As of now, the normal loss is still not working as expected.
            normal_lambda = self.cfg.opt.normal_lambda
            normals = renders["normals"]
            # normals = normals / torch.norm(normals, p=2, dim=-1, keepdim=True)
            normals_from_depth = (
                renders["normals_from_depth"] * renders["alphas"].detach()
            )
            # normals_from_depth = normals_from_depth / torch.norm(normals_from_depth, p=2, dim=-1, keepdim=True)

            normal_valid_mask = get_valid_mask(normals, threshold=1.0) * get_valid_mask(
                normals_from_depth, threshold=1.0
            )

            if normal_valid_mask.sum() > 0:
                normal_error = 1.0 - torch.einsum(
                    "nc, nc -> n",
                    normals[normal_valid_mask],
                    normals_from_depth[normal_valid_mask],
                )
                normalloss = normal_error.mean()

                loss += normal_lambda * normalloss
                logs[f"{subset}/normal_loss"] = normalloss

        # add the depth distortion loss
        # @todo: experiment with different loss function
        if self.cfg.opt.dist_loss and self.train_iter > self.cfg.opt.dist_start_iter:
            dist_lambda = self.cfg.opt.dist_lambda
            valid_mask = get_valid_mask(renders["render_distort"], threshold=5.0)

            distloss = renders["render_distort"][valid_mask].mean()

            loss += distloss * dist_lambda
            logs[f"{subset}/dist_lambda"] = distloss

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
            mask = einops.rearrange(mask, "b c h w -> b h w c")
            maskloss = renders["alphas"][mask < 0.5].mean()
            logs["train/mask_loss"] = maskloss
            loss += maskloss

        # self.log("train_iter", float(self.train_iter), on_step=True, on_epoch=False, logger=True, batch_size=1)
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
                packed=False,
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

        # self.trainer.global_step += 1
        for scheduler in self.schedulers:
            scheduler.step()

        to_return = {
            "loss": loss,
            "renders": renders,
        }

        return to_return
