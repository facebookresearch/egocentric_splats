import os
from datetime import datetime

import imageio

import numpy as np
import torch
import viser
import viser.transforms as vtf

from viewer import Viewer


def get_sizes(
    max_res: int,
    canvas_aspect_ratio: float,
    lock_aspect_one: bool,
):
    if lock_aspect_one:
        render_aspect_ratio = 1
        if canvas_aspect_ratio <= 1:  # width <= height
            canvas_height = max_res
            canvas_width = int(canvas_height * canvas_aspect_ratio)
        else:  # width > height
            canvas_width = max_res
            canvas_height = int(canvas_width / canvas_aspect_ratio)
        # Image size is always smaller than canvas size
        image_width = min(canvas_width, canvas_height)
        image_height = min(canvas_width, canvas_height)
    else:
        render_aspect_ratio = canvas_aspect_ratio
        if render_aspect_ratio <= 1:  # width <= height
            image_height = max_res
            image_width = int(image_height * render_aspect_ratio)
        else:  # width > height
            image_width = max_res
            image_height = int(image_width / render_aspect_ratio)
        canvas_width = image_width
        canvas_height = image_height

    return canvas_width, canvas_height, image_width, image_height


def find_nearest_se3(ref_pose: np.ndarray, poses: np.ndarray) -> [int, np.ndarray]:
    """
    Find the nearest 6D pose in poses to ref_pose

    Args:
        ref_pose: (4, 4) np.ndarray, the reference pose
        poses: (N, 4, 4) np.ndarray, the poses to search from

    Returns:
        (4, 4) np.ndarray, the nearest pose
    """
    # Compute the Frobenius norm (matrix norm) between ref_pose and each pose in poses
    distances = np.linalg.norm(poses - ref_pose, ord="fro", axis=(-2, -1))

    # Find the index of the pose with the smallest distance to ref_pose
    nearest_index = np.argmin(distances)

    # Return the nearest pose
    return nearest_index.item(), poses[nearest_index]


class ViewerCustomized(Viewer):
    """
    This is the viewer with a bunch of customized utility functions.

    The downstream viewer application can inherit from this and add UI components easily depending on needs.
    """

    up_direction = np.asarray([0.0, 0.0, 1.0])

    _aspect_ratio = None
    _scaling_modifier = None
    _exposure = None
    _gain = None
    _gamma = None

    _color_format: str = "rgb"

    show_cameras: bool = False
    apply_tonemapping: bool = False
    render_modality: str = (
        "color"  # render the image in color, depth, or normal modality
    )
    radiance_channel: str = (
        "rgb"  # render radiance in rgb or monochrome (only when both are available)
    )
    render_camera_model: str = (
        "linear"  # the rendering camera model, using linear or spherical
    )

    show_edit_panel = True
    show_render_panel = False
    render_gravity_aligned = True

    frame_slider_gui = {}

    def __init__(
        self,
        server: viser.ViserServer,
        render_func,
        mode: bool,
    ):
        """
        cfg: the configuration file
        """
        super().__init__(server, render_fn=render_func, mode=mode)

        self.camera_handles = []

    def _define_guis(self):
        """
        This will overwrite the skip the original default nerfview GUIs.
        All downstream applications will need to define customized viewer.
        """
        pass

    def _reorient(self, cameras, mode: str, dataset_type: str = None):
        """
        Reorient the scene according to input cameras.

        To be updated using Camera module instead of json camera input
        """
        transform = torch.eye(4, dtype=torch.float)

        if mode == "disable":
            return transform

        # skip reorient if dataset type is blender
        if dataset_type in ["blender", "nsvf"] and mode == "auto":
            print("skip reorient for {} dataset".format(dataset_type))
            return transform

        up = torch.zeros(3)
        for i in cameras:
            up += torch.tensor(i["rotation"])[:3, 1]
        up = -up / torch.linalg.norm(up)

        print("up vector = {}".format(up))
        self.up_direction = up

        return transform

    def add_cameras_to_scene(self, cameras, viser_server, stride: int = 10):
        """
        Visualiz the loaded cameras to the scene.
        """
        total_cam = len(cameras)
        if total_cam == 0:
            return

        if len(self.camera_handles) > 0:
            for i in self.camera_handles:
                i.visible = True
            return

        for idx, camera in enumerate(cameras):
            if idx % stride != 0:
                continue

            camera_id = camera.camera_id
            c2w = camera.c2w_44_np
            R = vtf.SO3.from_matrix(c2w[:3, :3])
            camera_handle = viser_server.scene.add_camera_frustum(
                name=f"cameras/{camera_id}",
                fov=camera.fov_x,
                scale=0.01,
                aspect=camera.aspect_ratio,
                wxyz=R.wxyz,
                position=c2w[:3, 3],
                color=(0, 25, 200),
            )

            @camera_handle.on_click
            def _(
                event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle],
            ) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.camera_handles.append(camera_handle)

        self.camera_visible = True

        # def toggle_camera_visibility(_):
        #     with viser_server.atomic():
        #         self.camera_visible = not self.camera_visible
        #         for i in self.camera_handles:
        #             i.visible = self.camera_visible

        # def update_camera_scale(_):
        #     with viser_server.atomic():
        #         for i in self.camera_handles:
        #             i.scale = self.camera_scale_slider.value

        # with viser_server.gui.add_folder("Visualized Cameras"):
        #     self.toggle_camera_button = viser_server.gui.add_button("Toggle Camera Visibility")
        #     self.camera_scale_slider = viser_server.add_slider(
        #         "Camera Scale",
        #         min=0.,
        #         max=0.1,
        #         step=0.01,
        #         initial_value=0.01,
        #     )
        # self.toggle_camera_button.on_click(toggle_camera_visibility)
        # self.camera_scale_slider.on_update(update_camera_scale)

    def get_current_model_path(self):
        return self.model_root + self.model_paths_dropdown.value

    def add_capture_save_folder(self, server: viser.ViserServer):
        if (
            hasattr(self, "capture_save_folder")
            and self.capture_save_folder is not None
        ):
            self.capture_save_folder.remove()

        with server.gui.add_folder("Capture") as gui_folder:
            self.capture_save_folder = gui_folder
            save_name_textbox = server.gui.add_text(
                "Save Name", initial_value="capture"
            )
            capture_button = server.gui.add_button("Capture", icon=viser.Icon.CAMERA)

            @capture_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                camera = event.client.camera
                save_name = save_name_textbox.value

                aspect_ratio = camera.aspect
                _, _, image_width, image_height = get_sizes(
                    self.max_resolution,
                    aspect_ratio,
                    self.lock_aspect_one,
                )

                render = camera.get_render(image_height, image_width, "jpeg")
                time_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("./captures", exist_ok=True)
                save_path = os.path.join("./captures", f"{save_name}_{time_suffix}.jpg")
                imageio.imwrite(save_path, render)

    def add_frame_slider_folder(self, cameras, slider_name: str):
        """
        Select a viewpoint from existing cameras
        """
        num_cameras = len(cameras)

        if slider_name in self.frame_slider_gui.keys():
            print(
                f"Duplicate {slider_name} being initialized! Will remove existing ones!"
            )
            self.frame_slider_gui[slider_name].remove()

        with self.server.gui.add_folder(slider_name) as gui_folder:
            self.frame_slider_gui[slider_name] = gui_folder
            # Add a slider to traverse through the training cameras
            # play_button = self.server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
            # pause_button = self.server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False)
            # snap_button = self.server.gui.add_button("Nearest camera", icon=viser.Icon.CAMERA, visible=True)
            prev_button = self.server.gui.add_button(
                "Previous", icon=viser.Icon.ARROW_AUTOFIT_LEFT
            )
            next_button = self.server.gui.add_button(
                "Next", icon=viser.Icon.ARROW_AUTOFIT_RIGHT
            )

            frame_step_slider = self.server.gui.add_slider(
                "Frame Step",
                min=1,
                max=20,
                step=1,
                initial_value=1,
            )

            frame_slider = self.server.gui.add_slider(
                "Frame",
                min=0,
                max=num_cameras - 1,
                step=1,
                initial_value=0,
            )
            self.frame_slider = frame_slider

            @frame_slider.on_update
            def _(_) -> None:
                # cam_pose = self.camera_poses[frame_slider.value]
                camera = cameras[frame_slider.value]
                c2w = camera.c2w_44_np
                R = vtf.SO3.from_matrix(c2w[:3, :3])

                self._exposure.value = camera.exposure_s * 1e3
                self._gain.value = camera.gain

                if self.render_gravity_aligned:  # image being rotated
                    self.max_resolution = camera.image_width
                    self.fov_height_radian = camera.fov_x
                else:
                    self.max_resolution = camera.image_height
                    self.fov_height_radian = camera.fov_y

                for client in self.server.get_clients().values():
                    with client.atomic():
                        client.camera.wxyz = R.wxyz
                        client.camera.position = c2w[:3, 3]
                        client.camera.fov = self.fov_height_radian

                        # set the train cameras to gravity aligned direction
                        if self.render_gravity_aligned:
                            client.camera.up_direction = np.asarray([0.0, 0.0, 1.0])

            @prev_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                frame_slider.value = (
                    frame_slider.value - frame_step_slider.value
                ) % num_cameras

            @next_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                frame_slider.value = (
                    frame_slider.value + frame_step_slider.value
                ) % num_cameras

            show_camera_checkbox = self.server.gui.add_checkbox(
                "Show cameras (1/10)",
                initial_value=False,
                hint="Visualize the cameras in the viewer.",
            )

            @show_camera_checkbox.on_update
            def _(event: viser.GuiEvent) -> None:
                self.show_cameras = show_camera_checkbox.value
                print(f"show cameras: {self.show_cameras}")

                if self.show_cameras:
                    self.add_cameras_to_scene(cameras, self.server)
                else:
                    for i in self.camera_handles:
                        i.visible = False
                    # self.camera_handles = []

            # To be updated
            """
            @snap_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                current_pose = np.eye(4)
                current_pose[:3, :3] = vtf.SO3(event.client.camera.wxyz).as_matrix()
                current_pose[:3, 3] = event.client.camera.position
                nearest_idx, nearest_pose = find_nearest_se3(current_pose, self.camera_poses_viewer)

                T_world_current = vtf.SE3.from_rotation_and_translation(
                    vtf.SO3(event.client.camera.wxyz), event.client.camera.position
                )
                T_world_target = vtf.SE3.from_matrix(nearest_pose)

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(10): # gradually move the camera there
                    T_world_set = T_world_current @ vtf.SE3.exp(
                        T_current_target.log() * j / 9.0
                    )

                    # We can atomically set the orientation and the position of the camera
                    # together to prevent jitter that might happen if one was set before the
                    # other.
                    with event.client.atomic():
                        event.client.camera.wxyz = T_world_set.rotation().wxyz
                        event.client.camera.position = T_world_set.translation()

                        if self.render_gravity_aligned: 
                            event.client.camera.up_direction = np.asarray([0., 0., 1.])

                    event.client.flush()  # Optional!
                    time.sleep(1.0 / 20.0)
                
                frame_slider.value = nearest_idx

            @play_button.on_click
            def _(_) -> None:
                play_button.visible = False
                pause_button.visible = True
                snap_button.visible = False
                
                def play() -> None:
                    while not play_button.visible:
                        if num_cameras > 0:
                            assert frame_slider is not None
                            frame_slider.value = (frame_slider.value + frame_step_slider.value) % num_cameras
                        time.sleep(1.0 / 2) # about 2 fps

                threading.Thread(target=play).start()
                
            @pause_button.on_click
            def _(_) -> None:
                play_button.visible = True
                pause_button.visible = False
                snap_button.visible = True
            """

    def add_camera_reset_button(self):
        # A button to set the current camera up direction to be the up direction
        reset_up_button = self.server.gui.add_button(
            "Reset up direction",
            icon=viser.Icon.ARROW_AUTOFIT_UP,
            hint="Set the current up direction as the up direction.",
        )

        @reset_up_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            event.client.camera.up_direction = vtf.SO3(
                event.client.camera.wxyz
            ) @ np.array([0.0, -1.0, 0.0])

    def add_aria_gravity_align_button(self):
        # A button to set the camera up direction to be the gravity direction
        set_top_up_button = self.server.gui.add_button(
            "Align Aria gravity",
            icon=viser.Icon.ARROW_AUTOFIT_UP,
            hint="Align the camera up direction with the gravity direction.",
        )

        @set_top_up_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            self.render_gravity_aligned = True
            event.client.camera.up_direction = np.asarray([0.0, 0.0, 1.0])

    def add_lock_aspect_ratio_button(self):
        # A checkbox to lock aspect ratio to be 1
        aspect_dict = {
            "full screen": None,
            "16:9": 16.0 / 9.0,
            "4:3": 4.0 / 3.0,
            "1:1 (RGB)": 1.0,
            "3:4 (SLAM)": 3.0 / 4.0,
        }

        aspect_ratio_dropdown = self.server.gui.add_dropdown(
            "aspect ratio",
            tuple(aspect_dict.keys()),
            initial_value="full screen",
        )
        # lock_aspect_one_checkbox = self.server.gui.add_checkbox(
        #     "Lock aspect ratio to 1",
        #     initial_value=self.lock_aspect_one,
        #     hint="Lock the aspect ratio of rendering to 1.",
        # )

        @aspect_ratio_dropdown.on_update
        def _(event: viser.GuiEvent) -> None:
            aspect_selected = aspect_ratio_dropdown.value
            self._aspect_ratio = aspect_dict[aspect_selected]
            self.rerender(_)

    def add_render_options_button(self):
        # add render options
        with self.server.gui.add_folder("Render"):
            render_modality = self.server.gui.add_dropdown(
                "Modality",
                ("color", "depth", "normal"),
                initial_value="color",
            )

            @render_modality.on_update
            def _(event: viser.GuiEvent) -> None:
                self.render_modality = render_modality.value
                print(f"Render modality: {self.render_modality}")
                self.rerender(_)

            render_camera_model = self.server.gui.add_dropdown(
                "Camera model",
                ("linear", "spherical"),
                initial_value="linear",
            )

            @render_camera_model.on_update
            def _(event: viser.GuiEvent) -> None:
                self.render_camera_model = render_camera_model.value
                print(f"Render using camera model: {self.render_camera_model}")
                self.rerender(_)

            if "rgb" in self._color_format:
                color_options = ("rgb", "monochrome")

                radiance_channel = self.server.gui.add_dropdown(
                    "Radiance channel",
                    color_options,
                    initial_value=self.radiance_channel,
                )

                @radiance_channel.on_update
                def _(event: viser.GuiEvent) -> None:
                    self.radiance_channel = radiance_channel.value
                    print(f"Render radiance channel: {self.radiance_channel}")
                    self.rerender(_)

            else:
                self.radiance_channel = "monochrome"

            self._max_img_res_slider = self.server.gui.add_slider(
                "Max Res (height)",
                min=128,
                max=3840,
                step=128,
                initial_value=1920,
            )
            self._max_img_res_slider.on_update(self.rerender)

            self._fov_y_slider = self.server.gui.add_slider(
                "FoV (height)",
                min=60,
                max=130,
                step=1,
                initial_value=100,
            )
            self._fov_y_slider.on_update(self.rerender)

            self.max_depth_range = self.server.gui.add_slider(
                "Depth Max(m)",
                min=0.1,
                max=10.0,
                step=0.1,
                initial_value=3.0,
            )
            self.max_depth_range.on_update(self.rerender)
            self.min_depth_range = self.server.gui.add_slider(
                "Depth Min(m)",
                min=0.1,
                max=10.0,
                step=0.1,
                initial_value=0.1,
            )
            self.min_depth_range.on_update(self.rerender)

    def add_rgb_postprocessing_button(self, cameras=None):
        # Add postprocessing options
        if cameras is not None:
            exp_all = [c.exposure_s for c in cameras]
            gain_all = [c.gain for c in cameras]
            exposure_init = float(np.median(exp_all) * 1e3)
            gain_init = float(np.median(gain_all))
            print(
                f"init visualizer with exposure {exposure_init} ms and gain {gain_init}"
            )
        else:
            # estimated value
            exposure_init = 5
            gain_init = 10

        with self.server.gui.add_folder("Postprocess"):
            tonemapping_checkbox = self.server.gui.add_checkbox(
                "Apply sRGB ",
                initial_value=self.apply_tonemapping,
                hint="Apply sRGB to rendering.",
            )

            @tonemapping_checkbox.on_update
            def _(event: viser.GuiEvent) -> None:
                self.apply_tonemapping = tonemapping_checkbox.value
                print(f"sRGB: {self.apply_tonemapping}")

            self._gamma = self.server.gui.add_slider(
                "gamma", min=1.0, max=2.6, step=0.1, initial_value=2.2
            )
            self._exposure = self.server.gui.add_slider(
                "exposure value (ms)",
                min=0.01,
                max=max(exposure_init * 2, 100),
                step=0.1,
                initial_value=exposure_init,
            )
            self._gain = self.server.gui.add_slider(
                "gain value",
                min=1,
                max=23,
                step=1.0,
                initial_value=gain_init,
            )
            self._exposure.on_update(self.rerender)
            self._gain.on_update(self.rerender)
            self._gamma.on_update(self.rerender)

            # self.awb4R = self.server.gui.add_slider(
            #     "AWB gain for red channel.",
            #     min = 0,
            #     max = 1,
            #     step = 0.01,
            #     initial_value = 1.0
            # )
            # self.awb4R.on_update(self.rerender)
            # self.awb4G = self.server.gui.add_slider(
            #     "AWB gain for green channel.",
            #     min = 0,
            #     max = 1,
            #     step = 0.01,
            #     initial_value = 1
            # )
            # self.awb4G.on_update(self.rerender)
            # self.awb4B = self.server.gui.add_slider(
            #     "AWB gain for blue channel.",
            #     min = 0,
            #     max = 1,
            #     step = 0.01,
            #     initial_value = 1.0
            # )
            # self.awb4B.on_update(self.rerender)

    def add_gaussian_model_option_button(self):
        with self.server.gui.add_folder("Model"):
            self._scaling_modifier = self.server.gui.add_slider(
                "Scaling Modifier",
                min=0,
                max=1.0,
                step=0.05,
                initial_value=1.0,
            )
            self._scaling_modifier.on_update(self.rerender)

            # if self.viewer_renderer.gaussians.max_sh_degree > 0:
            #     self.active_sh_degree_slider = self.server.gui.add_slider(
            #         "Active SH Degree",
            #         min=0,
            #         max=self.viewer_renderer.gaussians.max_sh_degree,
            #         step=1,
            #         initial_value=self.viewer_renderer.gaussians.max_sh_degree,
            #     )
            #     #@todo: update the SH cooefficients of the viewer
            #     self.active_sh_degree_slider.on_update(self.rerender)

            # self.time_slider = self.server.gui.add_slider(
            #     "Time",
            #     min=0.,
            #     max=1.,
            #     step=0.01,
            #     initial_value=0.,
            # )
            # self.time_slider.on_update(self.rerender)

    # def add_render_video_option_button(self, cameras):
    #     """
    #     Render an offline video
    #     """
    #     with self.server.gui.add_folder("Render Video"):
    #         render_video_button = self.server.gui.add_button(
    #             "Render video",
    #             icon=viser.Icon.ARROW_AUTOFIT_RIGHT
    #         )

    #         # sample the key frames
    #         key_frame_step_slider = self.server.gui.add_slider(
    #             "Key frame Step",
    #             min=1,
    #             max=100,
    #             step=10,
    #             initial_value=20,
    #         )

    #         @render_video_button.on_click
    #         def _(event: viser.GuiEvent) -> None:
    #             assert event.client is not None

    #         # @key_frame_step_slider.on_update
    #         # def _(_) -> None:

    @property
    def exposure(self) -> float:
        if self._exposure is None:
            return 1e-2
        else:
            return self._exposure.value * 1e-3

    @property
    def gain(self) -> float:
        if self._gain is None:
            return 1.0
        else:
            return self._gain.value

    @property
    def gamma(self) -> float:
        if self._gamma is None:
            return 2.2
        else:
            return self._gamma.value

    @property
    def scaling_modifier(self):
        if self._scaling_modifier is None:
            return 1.0
        else:
            return self._scaling_modifier.value

    @property
    def max_resolution(self):
        """
        max resolution (in height)
        """
        return self._max_img_res_slider.value

    @max_resolution.setter
    def max_resolution(self, value):
        self._max_img_res_slider.value = value

    @property
    def fov_height(self):
        return self._fov_y_slider.value

    @fov_height.setter
    def fov_height(self, value):
        self._fov_y_slider.value = value

    @property
    def fov_height_radian(self):
        return self.fov_height * np.pi / 180

    @fov_height_radian.setter
    def fov_height_radian(self, value):
        self._fov_y_slider.value = value * 180 / np.pi

    @property
    def render_aspect_ratio(self):
        return self._aspect_ratio
