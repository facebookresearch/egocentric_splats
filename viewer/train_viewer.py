from typing import List, Tuple

import einops
import numpy as np
import torch
import viser

from scene.cameras import Camera
from utils.image_utils import linear_to_sRGB
from utils.render_utils import apply_turbo_colormap, depth_to_normal

from .viewer_customized import ViewerCustomized


class TrainerViewerBase(ViewerCustomized):

    def __init__(
        self,
        server: viser.ViserServer,
        model: torch.nn.Module,
    ):
        super().__init__(server, self.viewer_render_fn, mode="training")

        self.viewer_renderer = model

    def viewer_render_fn(self):
        return None


class TrainerViewer3DGS(TrainerViewerBase):
    """
    A general purpose viewer. We will always use gsplat as the default rasterization solution.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        model: torch.nn.Module,
        train_cameras: List[Camera],
        valid_cameras: List[Camera],
        color_format: str = "rgb",
    ):
        super().__init__(server, model)

        self._train_cameras = train_cameras
        self._valid_cameras = valid_cameras
        self._color_format = color_format

        self._radiance_weight = self._train_cameras[0].radiance_weight

        server.on_client_disconnect(self._disconnect_client)
        server.on_client_connect(self._connect_client)

        self.generate_guis()

    def generate_guis(self):

        with self.server.gui.add_folder(
            "Stats", visible=self.mode == "training"
        ) as self._stats_folder:
            self._stats_text_fn = (
                lambda: f"""<sub>
                Step: {self._step}\\
                Last Update: {self._last_update_step} \\
                Total points: {self.viewer_renderer.total_points} \\
                Train time: {self.viewer_renderer.total_time_s:.2f}
                </sub>"""
            )
            self._stats_text = self.server.gui.add_markdown(self._stats_text_fn())

        with self.server.gui.add_folder(
            "Training", visible=self.mode == "training"
        ) as self._training_folder:
            self._pause_train_button = self.server.gui.add_button("Pause")
            self._pause_train_button.on_click(self._toggle_train_buttons)
            self._pause_train_button.on_click(self._toggle_train_s)
            self._resume_train_button = self.server.gui.add_button("Resume")
            self._resume_train_button.visible = False
            self._resume_train_button.on_click(self._toggle_train_buttons)
            self._resume_train_button.on_click(self._toggle_train_s)

            self._train_util_slider = self.server.gui.add_slider(
                "Train Util", min=0.0, max=1.0, step=0.05, initial_value=0.9
            )
            self._train_util_slider.on_update(self.rerender)

        with self.server.gui.add_folder("General") as self._rendering_folder:

            self.add_camera_reset_button()

            self.add_aria_gravity_align_button()

            self.add_lock_aspect_ratio_button()

            self.add_rgb_postprocessing_button(cameras=self._train_cameras)

            self.add_gaussian_model_option_button()

            self.add_render_options_button()

            self.add_frame_slider_folder(
                cameras=self.train_cameras, slider_name="train_cameras"
            )

            self.add_frame_slider_folder(
                cameras=self.valid_cameras, slider_name="valid_cameras"
            )

    def viewer_render_fn(self, camera_state, img_wh: Tuple[int, int]):

        c2w = camera_state.c2w
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)

        # fov = camera_state.fov
        fov = self.fov_height_radian

        aspect_ratio = self.render_aspect_ratio

        image_width, image_height = img_wh

        if aspect_ratio != None and image_width != int(image_height * aspect_ratio):
            render_foreground = True
            render_image = np.ones((image_height, image_width, 3))
            render_width = int(image_height * aspect_ratio)
            render_height = image_height
        else:
            render_foreground = False
            render_width = image_width
            render_height = image_height

        camera = Camera(
            uid=0,
            w2c=w2c,
            FoVx=fov,
            FoVy=fov,
            image_width=render_width,
            image_height=render_height,
            image_name="",
            image_path="",
            mask_path="",
            camera_name="render",
            camera_modality=self.radiance_channel,
            camera_projection_model=self.render_camera_model,
            exposure_duration_s=self.exposure,
            gain=self.gain,
            scene_name="",
        )

        camera.radiance_weight = self._radiance_weight

        with torch.no_grad():
            render_pkg = self.viewer_renderer.render(
                camera, self.scaling_modifier, rasterize_mode="antialiased"
            )

            image = camera.expose_image(render_pkg["render"])

            if camera.is_rgb:
                image = self.viewer_renderer.radiance_finishing(image)[0]
            else:
                image = image[0]

            image = einops.rearrange(image, "c h w -> h w c")

            if not camera.is_rgb:
                image = image.repeat(1, 1, 3)

            depth = render_pkg["depth"]

            if self.render_modality == "color":
                if self.apply_tonemapping:
                    image_vis = linear_to_sRGB(image, self.gamma)

                    image_vis = torch.clamp(image_vis, max=1.0)
                else:
                    image_vis = image
            elif self.render_modality == "depth":
                min_d, max_d = self.min_depth_range.value, self.max_depth_range.value
                depth_normalized = (depth.clamp(min_d, max_d) - min_d) / (max_d - min_d)
                image_vis = apply_turbo_colormap(depth_normalized[0])
            elif self.render_modality == "normal":
                if "normals" in render_pkg:
                    normal = render_pkg["normals"][0]
                else:
                    # estimate normal from depth
                    normal = depth_to_normal(camera, depth[0])
                image_vis = (normal + 1) / 2
            elif self.render_modality == "normal_from_depth":
                if "normals_from_depth" in render_pkg:
                    normal = render_pkg["normals_from_depth"]
                else:
                    # estimate normal from depth
                    normal = depth_to_normal(camera, depth[0])
                image_vis = (normal + 1) / 2
                # normal = einops.rearrange(normal, "c h w -> h w c")

        if render_foreground:
            pad = (image_width - render_width) // 2
            render_image[:, pad : pad + render_width] = image_vis.cpu().numpy()
        else:
            render_image = image_vis.cpu().numpy()

        return render_image

    @property
    def train_cameras(self):
        return self._train_cameras

    @property
    def valid_cameras(self):
        return self._valid_cameras

    @property
    def all_cameras(self):
        return self._train_cameras + self._valid_cameras
