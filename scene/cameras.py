import copy
import json
from dataclasses import dataclass

from pathlib import Path
from typing import Dict, List, Literal

import einops
import numpy as np
import projectaria_tools.core.mps as mps
import pyquaternion

import torch
from PIL import Image
from projectaria_tools.core.sophus import interpolate, SE3
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.aria_utils import interpolate_aria_pose

from utils.point_utils import project


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def PILtoTorch(input_image, resolution: tuple = None) -> torch.Tensor:
    if resolution is not None:
        input_image = input_image.resize(resolution)

    updated_image = np.array(input_image)
    updated_image = torch.from_numpy(updated_image) / 255.0
    updated_image = updated_image.clamp(0.0, 1.0)

    if len(updated_image.shape) == 3:
        return updated_image.permute(2, 0, 1)
    else:
        return updated_image[None]


@dataclass
class SubImageMask:
    """
    a sub-image ROI mask information
    """

    index: int  # the mask index in the original image
    mask: (
        torch.Tensor
    )  # mask with non-zero indicating the corresponding ROI in this image.
    y_min: int  # min index of y of the mask in the original image that is non-zero
    y_max: int  # max index of y of the mask in the original image that is non-zero


class Camera:
    """
    Edited based on the original implementation in 3D GS, but with following changes:

    1. The original 3D GS in default transform all camera matrices to column-major in default (for the convention used in their rasterizer).
     We kept the row-major operation, and make the conversion optional.

    2. We added a few extra parameters to camera (e.g. mask, exposure, gain).
    """

    # The following are shared across all cameras.
    # The vignette image determines the falloff ratio for each pixel.
    _vignette_image: Dict[str, torch.Tensor] = {}
    # The mask image determines an overall thresholded region for each pixel.
    _mask_image: Dict[str, torch.Tensor] = {}

    # the mapping of camera models supported in Aria camera to GSplat render camera model
    _camera_model_aria2gsplats = {
        "linear": "pinhole",
        "spherical": "fisheye",
    }

    max_exposure_s: float = 2e-3

    # map the irradiance in a nonlinear (gamma) space
    use_nonlinear_irradiance: bool = True

    def __init__(
        self,
        uid: int,
        w2c: np.ndarray,  # set world to camera transform
        FoVx: float,
        FoVy: float,
        image_width: int,
        image_height: int,
        cx: int = -1,
        cy: int = -1,
        image_name: str = None,
        image_path: str = None,
        mask_path: str = None,
        camera_name: Literal[
            "camera-rgb", "camera-slam-left", "camera-slam-right"
        ] = "camera-rgb",
        camera_projection_model: Literal["linear", "spherical"] = "linear",
        camera_modality: Literal["rgb", "monochrome"] = "rgb",
        exposure_duration_s: float = 1.0,
        gain: float = 1.0,
        radiance_weight: float = 1.0,
        scene_name: str = "",
        cache_image: bool = False,
    ):
        super(Camera, self).__init__()

        self._uid = (
            uid  # This is mostly used as the index of the camera among all cameras.
        )
        # self._R_w2c = w2c[:3, :3]
        # self._T_w2c = w2c[:3, 3]

        self._w2c = w2c

        self._fov_x = FoVx
        self._fov_y = FoVy
        self._image_width = image_width
        self._image_height = image_height

        if cx < 0:
            self._cx = self._image_width // 2
        else:
            self._cx = cx

        if cy < 0:
            self._cy = self._image_height // 2
        else:
            self._cy = cy

        # retain a copy for reset purpose
        self._fov_x_initial = FoVx
        self._fov_y_initial = FoVy
        self._image_width_initial = image_width
        self._image_height_initial = image_height

        self._image_path = image_path
        self._mask_path = mask_path
        self._exposure_duration_s = exposure_duration_s
        self._gain = gain
        self._camera_name = camera_name
        self._camera_modality = camera_modality
        self._camera_projection_model = camera_projection_model

        # for a standard camera without explicit camera model. We will use the uid.
        self._timestamp_ns = uid

        self._rs_info = []

        # self.image_name = image_name
        self.scene_name = scene_name

        self.zfar = 100.0
        self.znear = 0.01

        # self.trans = np.array([0.0, 0.0, 0.0])
        # self.scale = scale

        # We will rescale the dynamic range of the observed irradiance image.
        # It is also the denominator to compensate the weight of exposure & gain, so the observed irradiance can be roughly fit into the range [0,1]
        self._radiance_weight = radiance_weight

        # The observed image (if available)
        self._observed_image: torch.Tensor = None
        # An associated mask image (if available)
        self._alpha_mask: torch.Tensor = None
        # sparse depth input (if available)
        self._sparse_points: torch.Tensor = None  # (N, 5)

        # The estimated image (cached from most recent rendering if available)
        self._est_image: torch.Tensor = None
        # The estimated depth (cached from most recet rendering if available)
        self._est_depth: torch.Tensor = None
        # The estimated normal (cached from most recent rendering if available)
        self._est_normal: torch.Tensor = None

        # cache the loaded image (could lead to memory issue if turns out to be too many of the)
        self.cache_image: bool = cache_image

    def set_vignette_image(self, vignette_image_path: str, camera_name: str):
        """
        This will set the vignette image for a specific camera_name
        """
        if camera_name != self._camera_name:
            raise RuntimeError(
                f"Vignette image camera name {camera_name} is inconsistent with current name {self.camera_name}"
            )

        if camera_name in self._vignette_image.keys():
            raise Warning(
                "The provided vignette image has already been set! Overwrite existing vignette image path!"
            )

        assert (
            vignette_image_path.exists()
        ), f"Could not find vignette for for camera-rgb at {vignette_image_path}"
        vignette_image = np.array(Image.open(str(vignette_image_path))) / 255.0

        # only choose one channel for slam camera
        if camera_name.startswith("camera-slam"):
            vignette = torch.from_numpy(vignette_image[..., 0]).float()
        else:
            if vignette_image.ndim == 2:
                vignette = torch.from_numpy(vignette_image)[None].float()
            else:
                vignette = einops.rearrange(
                    torch.from_numpy(vignette_image), "h w c -> c h w"
                ).float()

        self._vignette_image[camera_name] = vignette

    def set_valid_mask(self, mask_image_path: str, camera_name: str):
        """
        This will set the valid mask for the camera.
        For example, in Aria RGB camera, the mask exclude periphery regions where pixels are too noisy
        """
        if camera_name != self._camera_name:
            raise RuntimeError(
                f"Mask image camera name {camera_name} is inconsistent with current name {self.camera_name}"
            )

        if camera_name in self._mask_image.keys():
            raise Warning(
                "The provided vignette image has already been set! Overwrite existing vignette image path!"
            )

        assert (
            mask_image_path.exists()
        ), f"Could not find vignette for for camera-rgb at {mask_image_path}"
        mask_image = np.array(Image.open(str(mask_image_path))) / 255

        if mask_image.ndim == 2:
            self._mask_image[camera_name] = torch.from_numpy(mask_image)[None]
        else:
            self._mask_image[camera_name] = einops.rearrange(
                torch.from_numpy(mask_image), "h w c -> c h w"
            )

    @property
    def camera_id(self):
        return self._uid

    @property
    def is_rgb(self) -> bool:
        return self.camera_modality == "rgb"

    @property
    def exposure_multiplier(self):
        """
        This aggregates the exposure and anolog gain as a single value
        """
        return self._exposure_duration_s * self._gain

    @property
    def exposure_s(self):
        return self._exposure_duration_s

    @property
    def gain(self):
        return self._gain

    def amplify_gain(self, ratio):
        self._gain *= ratio

    @property
    def vignette_image(self) -> torch.Tensor:
        if self._camera_name in self._vignette_image.keys():
            return self._vignette_image[self._camera_name]
        else:
            return None

    @property
    def valid_mask(self) -> torch.Tensor:
        if self._camera_name in self._mask_image.keys():
            return self._mask_image[self._camera_name]
        else:
            return None

    @property
    def radiance_weight(self) -> float:
        return self._radiance_weight

    @radiance_weight.setter
    def radiance_weight(self, v: float):
        self._radiance_weight = v

    @property
    def fov_x(self) -> float:
        return self._fov_x

    @property
    def fov_y(self) -> float:
        return self._fov_y

    @property
    def fx(self) -> float:
        return fov2focal(self.fov_x, self._image_width)

    @property
    def fy(self) -> float:
        return fov2focal(self.fov_y, self._image_height)

    @property
    def cx(self) -> float:
        return self._cx

    @property
    def cy(self) -> float:
        return self._cy

    @property
    def image_width(self) -> int:
        return self._image_width

    @property
    def image_height(self) -> int:
        return self._image_height

    @image_height.setter
    def image_height(self, value):
        """
        This will only reset the image height while keeping other sensor config the same
        """
        scale_ratio = value / self._image_height

        self._image_width = int(value * self.aspect_ratio)
        self._image_height = value

        self._cx *= scale_ratio
        self._cy *= scale_ratio

    def zoom(self, ratio, keep_width: bool = False, keep_height: bool = False):
        """
        simulate a digital zoom by adjusting the focal
        """
        if not keep_width:
            fx = ratio * self.fx
            fov_x = focal2fov(fx, self._image_width)
        else:
            fov_x = self._fov_x

        if not keep_height:
            fy = ratio * self.fy
            fov_y = focal2fov(fy, self._image_height)
        else:
            fov_y = self._fov_y

        self._set_fov(fov_x, fov_y)

    def reset(self):
        """
        reset the parameters to the initialized camera config
        """
        scale_ratio_x = self._image_height_initial / self._image_height
        scale_ratio_y = self._image_width_initial / self._image_width

        self._image_width = self._image_width_initial
        self._image_height = self._image_height_initial

        self._cx *= scale_ratio_x
        self._cy *= scale_ratio_y

        self._set_fov(self._fov_x_initial, self._fov_y_initial)

    @property
    def intrinsic(self) -> torch.FloatTensor:
        """ """
        return torch.FloatTensor(
            [
                [self.fx, 0, self._cx],
                [0, self.fy, self._cy],
                [0, 0, 1],
            ]
        )

    @property
    def intrinsic_np(self) -> np.ndarray:
        """ """
        return np.array(
            [
                [self.fx, 0, self._cx],
                [0, self.fy, self._cy],
                [0, 0, 1],
            ]
        )

    @property
    def c2w_44(self) -> torch.FloatTensor:
        """
        a 4x4 camera to world matrix in torch tensor.
        This is in default in the COLMAP coordinate system (Y down, Z forward)
        """
        return torch.from_numpy(self.c2w_44_np).float()

    @property
    def c2w_44_np(self) -> np.ndarray:
        """
        a 4x4 camera to world matrix in numpy array
        """
        c2w = np.linalg.inv(self.w2c_44_np)
        return c2w

    @property
    def c2w_44_opengl(self) -> torch.FloatTensor:
        """
        a 4x4 camera to world matrix in OpenGL/Blender camera axes

        change from OpenGL/Blender camera axes (Y up, Z back)
        """
        c2w = self.c2w_44
        c2w[:3, 1:3] *= -1
        return c2w

    @property
    def w2c_44(self) -> torch.FloatTensor:
        return torch.from_numpy(self.w2c_44_np).float()

    @property
    def w2c_44_np(self) -> np.ndarray:
        return self._w2c

    @property
    def R_w2c(self) -> torch.FloatTensor:  # (3,3)
        return torch.FloatTensor(self._w2c[:3, :3])

    @property
    def T_w2c(self) -> torch.FloatTensor:  # (3,1)
        return torch.FloatTensor(self._w2c[:3, 3:])

    @property
    def aspect_ratio(self) -> float:
        return float(self._image_width) / self._image_height

    @aspect_ratio.setter
    def aspect_ratio(self, value):
        # existing_ratio = self.aspect_ratio
        # reset fov horizontally
        new_width = int(self._image_height * value)
        scale_ratio_x = new_width / self._image_width

        fov_x = focal2fov(self.fx, new_width)
        self._image_width = new_width
        self._set_fov(fov_x, self._fov_y)

        self._cx *= scale_ratio_x

    @property
    def camera_name(self) -> str:
        return self._camera_name

    @property
    def camera_modality(self) -> str:
        return self._camera_modality

    @property
    def camera_projection_model(self) -> str:
        return self._camera_projection_model

    @property
    def camera_projection_model_gsplat(self) -> str:
        return self._camera_model_aria2gsplats[self._camera_projection_model]

    @property
    def is_moving_camera(self) -> bool:
        return False

    def get_exposure_viewmatrices(self, max_exposure_s: float = 2e-3) -> np.ndarray:
        """
        currently it is only simulating the pose for the center row

        This is currently assuming rolling shutter free.
        @todo: Support rolling shutter matrices
        """
        w2c_SE3_exposure_start = SE3.from_matrix(self._w2c)
        w2c_SE3_exposure_end = SE3.from_matrix(self._w2c_exposure_end)

        w2c_array = []
        exposure_step_size = int(self.exposure_s / max_exposure_s)
        for idx in range(0, exposure_step_size + 1):
            w2c_interp = interpolate(
                w2c_SE3_exposure_start, w2c_SE3_exposure_end, idx / exposure_step_size
            )
            w2c_array.append(w2c_interp.to_matrix())
        w2c_array = np.stack(w2c_array)

        return w2c_array

    def expose_image(
        self, irradiance: torch.Tensor, clamp: bool = True
    ) -> torch.Tensor:

        if self.use_nonlinear_irradiance:
            return self.expose_image_gamma(irradiance, clamp)
        else:
            return self.expose_image_linear(irradiance, clamp)

    def expose_image_gamma(
        self, irradiance: torch.Tensor, clamp: bool = True
    ) -> torch.Tensor:
        """
        Overwrite the expose image function with Aria specific properties

        We assume the scenb irradiance is in a gamma space instead of linear space.
        When adding linear operations, we will convert it to a gamma space and then convert it back.
        """
        gamma = 2.2
        mask_linear = irradiance < 0.04045
        gamma_reciprocal = 1.0 / gamma
        irradiance_linear = irradiance[mask_linear] / 12.92
        irradiance_gamma = ((irradiance[~mask_linear] + 0.055) / 1.055) ** gamma

        irradiance_linear *= self.exposure_multiplier * self._radiance_weight
        irradiance_gamma *= self.exposure_multiplier * self._radiance_weight
        # exposed_image = image * self.exposure_multiplier * self._radiance_weight

        if self.vignette_image is not None:
            v = self.vignette_image.expand_as(irradiance).to(irradiance.device)
            irradiance_linear *= v[mask_linear]
            irradiance_gamma *= v[~mask_linear]

        if self.is_rgb:
            # if self.aria_rgb_crf is not None:
            #     color_map = self.aria_rgb_crf.to(irradiance.device)
            #     raise RuntimeError("Color map conversion is problematic")
            # else:

            # apply gamma compression in the image space
            exposed_image = torch.zeros_like(irradiance)
            irradiance_linear = 12.92 * irradiance_linear
            irradiance_gamma = (
                1.055 * torch.pow(irradiance_gamma, gamma_reciprocal) - 0.055
            )
            exposed_image[mask_linear] = irradiance_linear
            exposed_image[~mask_linear] = irradiance_gamma
        else:
            exposed_image = torch.zeros_like(irradiance)
            exposed_image[mask_linear] = irradiance_linear
            exposed_image[~mask_linear] = irradiance_gamma

        if self.valid_mask is not None:
            exposed_image *= self.valid_mask.to(irradiance.device)

        if clamp:
            return exposed_image.clamp(0, 1).float()
        else:
            return exposed_image.float()

    def expose_image_linear(
        self, irradiance: torch.Tensor, clamp: bool = True
    ) -> torch.Tensor:
        """
        Generate an exposed image assuming the input and output irradiance are both linear.
        This is used as the baseline to reflect the majority of algorithms.
        """
        exposed_image = irradiance * self.exposure_multiplier * self._radiance_weight

        if self.vignette_image is not None:
            exposed_image *= self.vignette_image.to(irradiance.device)

        if self.valid_mask is not None:
            exposed_image *= self.valid_mask.to(irradiance.device)

        # if self.is_rgb:
        #     exposed_image = apply_aria_rgb_gamma(exposed_image)

        if clamp:
            return exposed_image.clamp(0, 1).float()
        else:
            return exposed_image.float()

    @property
    def image_name(self):
        if self._image_path is not None:
            return self._image_path.split("/")[-1]
        else:
            return f"{self._timestamp_ns:03f}.png"

    @property
    def image_path(self):
        return self._image_path

    def get_image(self, resolution_scale):

        if self._observed_image is not None:
            return self._observed_image, self._alpha_mask

        image = Image.open(self._image_path)
        image_array = PILtoTorch(image)

        # If the image is RGBA image, extract the alpha channel as the mask
        alpha_mask = None
        gt_image = image_array[:3, ...]
        if image_array.shape[0] == 4:
            alpha_mask = image_array[3:4, ...]

        # If there is an additional mask associated with it to further crop the ROI
        if alpha_mask is None and self._mask_path is not None:
            alpha_mask = Image.open(self._mask_path)
            alpha_mask = PILtoTorch(alpha_mask).squeeze()

        if self.cache_image:
            self._observed_image = gt_image
            self._alpha_mask = alpha_mask

        return gt_image, alpha_mask

    @property
    def image(self) -> Image:
        return Image.open(self._image_path)

    @property
    def sparse_depth(self) -> torch.Tensor:
        if self._sparse_points is not None:
            return self._sparse_points[:, 2]
        else:
            return None

    @property
    def sparse_inv_depth(self) -> torch.Tensor:
        if self._sparse_points is not None:
            depth = self.sparse_depth
            # will suppress that is too close
            invdepth = torch.where(depth > 1e-1, 1.0 / depth, torch.zeros_like(depth))
            return invdepth
        else:
            return None

    @property
    def sparse_point2d(self) -> torch.Tensor:
        if self._sparse_points is not None:
            return self._sparse_points[:, :2]
        else:
            return None

    @property
    def sparse_point_inv_distance_std(self) -> torch.Tensor:
        """
        Standard deviation of the distance estimate, in meters.
        Could be used for determining the quality of the 3D point position estimate
        """
        if self._sparse_points is not None:
            return self._sparse_points[:, 3]
        else:
            return None

    @property
    def sparse_point_distance_std(self) -> torch.Tensor:
        """
        Standard deviation of the inverse distance estimate, in meter^-1.
        Could be used for determining the quality of the 3D point position estimate
        """
        if self._sparse_points is not None:
            return self._sparse_points[:, 4]
        else:
            return None

    def read_sparse_depth(self, json_file: str):
        """
        read sparse depth from the spasre depth json file
        """
        with open(json_file, "r") as f:
            data = json.load(f)

        u = torch.FloatTensor(data["u"])
        v = torch.FloatTensor(data["v"])
        z = torch.FloatTensor(data["z"])
        invd_std = torch.FloatTensor(data["inverseDistanceStd"])
        d_std = torch.FloatTensor(data["distanceStd"])

        if len(u) > 0:
            self._sparse_points = torch.stack([u, v, z, invd_std, d_std], dim=1)

    @property
    def render_image(self) -> torch.Tensor:
        return self._est_image

    @render_image.setter
    def render_image(self, value: torch.Tensor):
        self._est_image = value

    @property
    def render_depth(self) -> torch.Tensor:
        return self._est_depth

    @render_depth.setter
    def render_depth(self, value: torch.Tensor):
        self._est_depth = value

    @property
    def render_depth_min(self) -> float:
        """
        The closest distance of the rendered depth
        """
        return self._est_depth_min

    @render_depth_min.setter
    def render_depth_min(self, value: float):
        self._est_depth_min = value

    @property
    def render_normal(self) -> torch.Tensor:
        return self._est_normal

    @property
    def sample_interval_min(self) -> float:
        """
        a sample interval depth / focal, eq(6) defined in mip-splatting
        """
        if self.render_depth is None:
            return 0.0  # we should densely sample as much as possible

        return self.render_depth.min() / max(self.fx, self.fy)

    @render_normal.setter
    def render_normal(self, value: torch.Tensor):
        self._est_normal = value

    def _set_image_size(self, new_width, new_height, keep_fovy=True):
        old_width = self._image_width
        old_height = self._image_height
        self._image_width = new_width
        self._image_height = new_height

        # Adjust the FoV according to the new image size
        if keep_fovy:
            ratio = new_width / new_height / (old_width / old_height)
            self._fov_x = np.arctan(np.tan(self._fov_x / 2) * ratio) * 2
        else:
            ratio = new_height / new_width / (old_height / old_width)
            self._fov_y = np.arctan(np.tan(self._fov_y / 2) * ratio) * 2

        self._set_fov(self._fov_x, self._fov_y)

    def _set_fov(self, FoVx, FoVy):
        self._fov_x = FoVx
        self._fov_y = FoVy

    def reset_extrinsic(self, w2c):
        self._w2c = w2c

    def copy(self):
        new_cam = copy.deepcopy(self)
        return new_cam

    def to_json(self):
        return camera_to_JSON(self._uid, self)

    @property
    def time_s(self) -> float:
        return self._timestamp_ns

    def update_time(self, value):
        self._timestamp_ns = value


class AriaCamera(Camera):
    """
    A physical moving camera (for Aria camera set)

    We use following properties for an Aria RGB image.

    For full resolution RGB image (2880x2880), the Rolling-shutter read-out time is
    """

    _rs_row_index_image: Dict[str, torch.Tensor] = {}
    _rs_row_index_masks: List[SubImageMask] = []

    # the read out time for an iamge at different resolution
    _readout_time_calib = {
        "full": int(16.26 * 1e6),  # The full resolution for Aria RGB camera 2880x2880
        "half": int(5 * 1e6),  # The half resolution for Aria RGB camera 1408x1408
        "arcata": int(9.171076 * 1e6),  # The
    }

    # the sampling frequency in time (ns) we use to calculate the potential offset
    _motion_sample_max = 2 * 1e6  # 2.5 * 1e6

    _max_rolling_shutter_sample = 8
    _max_exposure_sample = 2

    use_aria_rgb_crf: bool = False
    aria_rgb_crf: torch.Tensor = None

    def __init__(
        self,
        uid: int,
        closed_loop_traj: List[mps.ClosedLoopTrajectoryPose],
        camera2device: np.ndarray,
        timestamp_ns: int,
        camera_name: str,
        camera_modality: str,
        is_rolling_shutter: bool,
        FoVx: float,
        FoVy: float,
        image_width: int,
        image_height: int,
        cx: int = -1,
        cy: int = -1,
        image_name: str = None,
        image_path: str = None,
        mask_path: str = None,
        rolling_shutter_index_image_path: Path = None,
        sparse_depth_path: Path = None,
        camera_projection_model: Literal["linear", "spherical"] = "linear",
        exposure_duration_s: float = 1.0,
        gain: float = 1.0,
        radiance_weight: float = 1.0,
        scene_name: str = "",
        cache_image: bool = False,
        readout_time_ns: int = 0,
    ):
        """
        * closed_loop_traj: The Aria 1KHz trajectory
        * rolling_shutter_index_image_path: an image for the rolling shutter index per row
        * timestamp_ns: the central timestamp for the image
        * exposure_duration_s: the exposure time for the current pixel
        """
        super().__init__(
            uid=uid,
            w2c=None,
            FoVx=FoVx,
            FoVy=FoVy,
            image_width=image_width,
            image_height=image_height,
            cx=cx,
            cy=cy,
            image_name=image_name,
            image_path=image_path,
            mask_path=mask_path,
            camera_name=camera_name,
            camera_projection_model=camera_projection_model,
            camera_modality=camera_modality,
            exposure_duration_s=exposure_duration_s,
            gain=gain,
            radiance_weight=radiance_weight,
            scene_name=scene_name,
            cache_image=cache_image,
        )

        camera2device = SE3.from_matrix(camera2device)
        self._is_rolling_shutter = is_rolling_shutter

        exposure_ns = exposure_duration_s * 1e9
        exp_offset_ns = exposure_ns // 2

        if sparse_depth_path is not None:
            self.read_sparse_depth(sparse_depth_path)
            # we use the sparse point to calculate the possible pixel offset during 1ms, and use this value to compensate the motion
            sampled_pixel_offset = self._estimate_pixel_motion_offset(
                closed_loop_traj,
                timestamp_ns,
                exp_offset_ns=self._motion_sample_max // 2,
            )
        else:
            # we will use an imaginary value if this does not exist!
            sampled_pixel_offset = 1.0

        if camera_name in ["camera-rgb", "camera_rgb"]:  # rolling shutter camera
            assert is_rolling_shutter, "camera-rgb should be a rolling shutter camera!"

            readout_time = readout_time_ns  # self._readout_time_calib[read_time_type]

            # We choose the quantization range, we expect the maximum pixel offset is within 1 pixel if possible.
            # We use a maximum of 16 samples.
            rs_row_quantization_range = int(
                min(
                    max(
                        readout_time / self._motion_sample_max * sampled_pixel_offset, 1
                    ),
                    self._max_rolling_shutter_sample,
                )
            )
            self.rs_row_quantization_range = rs_row_quantization_range

            if (
                rs_row_quantization_range > 1
            ):  # is rolling shutter camera and require multiple samples across time
                print(
                    f"There are {sampled_pixel_offset:02f} pixels within {self._motion_sample_max/1e6} ms. Will take rs step: {rs_row_quantization_range}"
                )

            rs_step_ns = readout_time / rs_row_quantization_range
            # sampled rolling_shutter time
            if rs_row_quantization_range > 1:
                # We will use the center rolling-shutter time within each rolling-shutter bracket
                rs_read_start_ns = timestamp_ns - readout_time // 2
                rs_bracket_timestamps = [
                    rs_read_start_ns + idx * rs_step_ns
                    for idx in range(rs_row_quantization_range + 1)
                ]
                rs_sampled_timestamps = [
                    (rs_bracket_timestamps[i] + rs_bracket_timestamps[i + 1]) / 2
                    for i in range(rs_row_quantization_range)
                ]
            else:  # this indicate the frame is static!
                rs_sampled_timestamps = [timestamp_ns]

            if self._camera_name not in self._rs_row_index_image.keys():
                self._rs_row_index_image[self._camera_name] = (
                    np.array(Image.open(str(rolling_shutter_index_image_path))) / 255.0
                )

            # we will use center row timestamp to approximate
            self._timestamp_ns = timestamp_ns

        else:  # global shutter camera
            rs_sampled_timestamps = [timestamp_ns]
            self._timestamp_ns = timestamp_ns
            self.rs_row_quantization_range = 1

        # sampled motion blur time for each camera
        exposure_step_range = int(
            min(
                max(
                    round(exposure_ns / self._motion_sample_max * sampled_pixel_offset),
                    1,
                ),
                self._max_exposure_sample,
            )
        )
        if exposure_step_range > 1:
            # Require multiple samples within the exposure range
            # We will sample the mid-exposure time within a evenly divided exposure bracket.
            # This allows us to average the exposed images at the given exposure time and integrate them using the full exposure time value.
            exposure_step_ns = exposure_ns / exposure_step_range
            sampled_timestamps = []
            for t in rs_sampled_timestamps:
                t_exp_start = t - exp_offset_ns
                exposure_brackets_timestamps = [
                    t_exp_start + idx * exposure_step_ns
                    for idx in range(exposure_step_range + 1)
                ]
                exposure_sample_timestamps = [
                    (
                        exposure_brackets_timestamps[i]
                        + exposure_brackets_timestamps[i + 1]
                    )
                    / 2.0
                    for i in range(exposure_step_range)
                ]
                sampled_timestamps.append(exposure_sample_timestamps)
            print(f"camera {uid} has multiple exposure samples {exposure_step_range}")
        else:
            sampled_timestamps = [[t] for t in rs_sampled_timestamps]

        # generate an array of poses (M N), where M is the samples of the exposure bracket, and N is the rolling-shutter bracket
        self.sampled_timestamps = np.array(sampled_timestamps)

        # sampled the corresponding poses for the corresponding timestamp
        self._w2c_array = self.sample_viewmatrices(
            closed_loop_traj, self.sampled_timestamps, camera2device
        )

        # store the center view transform pose. Used if not using the pose array
        center_row_pose_info = interpolate_aria_pose(closed_loop_traj, timestamp_ns)
        c2w_SE3 = center_row_pose_info.transform_world_device @ camera2device
        w2c_SE3 = c2w_SE3.inverse()
        self._w2c_center = w2c_SE3.to_matrix()

        # @todo: for extremelly blurred frames, we can further reduce the resolutions being used in training.
        if self.use_aria_rgb_crf:
            if self.aria_rgb_crf is None:
                aria_rgb_crf_file = Path("data/aria_crf_irradiance2color_map.json")
                assert aria_rgb_crf_file.exists(), "Aria RGB crf fiile does not exist!"

                with open(aria_rgb_crf_file, "r") as f:
                    crf_lookup_tab = json.load(f)

                # irradiance (0,1) -> color (0,1)
                self.aria_rgb_crf = torch.Tensor(crf_lookup_tab["map1024"]) / 255.0
        else:
            self.aria_rgb_crf = None

    def sample_viewmatrices(
        self,
        closed_loop_traj: List[mps.ClosedLoopTrajectoryPose],
        sampled_timestamps_ns: np.ndarray,
        camera2device,
    ) -> np.ndarray:
        """
        will return a set of viewmatrix correspond to the sampled timestamps.

        This is currently assuming exposure time is within limit.
        @todo: Support exposure matrices
        """
        timestamps = sampled_timestamps_ns.flatten()

        w2c_array = []
        for timestamp in timestamps:
            pose_info = interpolate_aria_pose(closed_loop_traj, timestamp)

            # if pose_info.quality_score < 0.9:
            #     print(f"pose quality score below 1.0: {pose_info.quality_score}!")

            c2w_SE3 = pose_info.transform_world_device @ camera2device
            w2c_SE3 = c2w_SE3.inverse()
            w2c_array.append(w2c_SE3.to_matrix())

        w2c_array = np.stack(w2c_array)

        N_rs, N_exp = sampled_timestamps_ns.shape
        return w2c_array.reshape(N_rs, N_exp, 4, 4)

    def _estimate_pixel_motion_offset(
        self,
        closed_loop_traj: List[mps.ClosedLoopTrajectoryPose],
        timestamp: int,
        exp_offset_ns: int = 1e6,
        threshold_percentile: float = 50,
    ):
        """
        Calculate the moved pixel range given a time-window (2 * exp_offset_ns).

        threshold_precentile: the thresholding percentile
        """

        uv = self.sparse_point2d
        if uv is None:
            return 1e-3  # we almost cannot treat it as moving camera

        u = uv[:, 0]
        v = uv[:, 1]
        z = self.sparse_depth

        fx, fy, cx, cy = self.fx, self.fy, self._cx, self._cy
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z

        t_start = timestamp - exp_offset_ns
        t_end = timestamp + exp_offset_ns
        pose_start = interpolate_aria_pose(closed_loop_traj, t_start)
        pose_end = interpolate_aria_pose(closed_loop_traj, t_end)

        w2c_rel = (
            pose_end.transform_world_device.inverse()
            @ pose_start.transform_world_device
        )

        pt3d = np.stack([x, y, z])

        u_proj, v_proj, z_proj, mask = project(
            pt3d,
            T_w2c=w2c_rel.to_matrix(),
            calibK=self.intrinsic_np,
            frame_h=self.image_height,
            frame_w=self.image_width,
        )

        # calculate the follow vector
        proj_offset = np.stack(
            [(u[mask] - u_proj).abs(), (v[mask] - v_proj).abs()]
        ).max(axis=0)
        # exposure_flow_det = abs(u[mask] - u_proj)  abs(v[mask] - v_proj)

        # print(f"timestamp: {timestamp_start}")
        # print(f"maximum pixel offset during exposure time: {exposure_flow_det.max()}")
        # print(f"75 percentil offset during exposure time: {np.percentile(exposure_flow_det, 75)}")
        # print(f"50 percentil offset during exposure time: {np.percentile(exposure_flow_det, 50)}")
        return np.percentile(proj_offset, threshold_percentile)

    @property
    def is_moving_camera(self) -> bool:
        return self._w2c_array.shape[0] > 1 or self._w2c_array.shape[1] > 1

    @property
    def w2c_44_np(self) -> np.ndarray:
        """
        choose the center pixel poses among all
        """
        return self._w2c_center

    def reset_extrinsic(self, w2c):
        self._w2c_center = w2c

    @property
    def motion_w2c_array_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self._w2c_array).float()

    @property
    def is_rolling_shutter(self) -> bool:
        return self._is_rolling_shutter

    @property
    def rolling_shutter_index_image(self) -> torch.Tensor:
        return torch.from_numpy(
            np.round(
                self._rs_row_index_image[self._camera_name]
                * (self.rs_row_quantization_range - 1)
            )
        ).long()

    # @property
    # def motion_info(self):
    #     return torch.from_numpy(self._w2c_array).float(), torch.from_numpy(self._rs_row_index_image[self._camera_name]).long()

    # @property
    # def exposure_view_matrices(self) -> torch.Tensor:
    #     return torch.from_numpy(self._w2c_array).float()

    # @property
    # def has_exposure_viewmatrices(self) -> bool:
    #     return (self.exposure_w2c_array is not None)

    @property
    def time_s(self) -> float:
        return self._timestamp_ns / 1e9

    def update_time(self, value):
        self._timestamp_ns = value * 1e9


def to_quat(rot_3x3):
    try:
        quat = pyquaternion.Quaternion(matrix=rot_3x3)
    except:
        U, _, Vt = np.linalg.svd(rot_3x3)
        rot_orthognol = np.dot(U, Vt)
        quat = pyquaternion.Quaternion(matrix=rot_orthognol)
    return quat


def interpolate_piecewise(cam1: Camera, cam2: Camera, w: float):
    """
    Generate an interpolated camera given two existing cameras.
    It will interpolate properties between the two
    """
    cam_interp = copy.deepcopy(cam1)

    # R, t = cam1.get_extrinsics
    # interpolate using c2w
    c2w_src = cam1.c2w_44_np
    c2w_dst = cam2.c2w_44_np

    q_src = to_quat(c2w_src[:3, :3])
    q_dst = to_quat(c2w_dst[:3, :3])

    q = pyquaternion.Quaternion.slerp(q_src, q_dst, w)

    c2w_interp = np.eye(4)
    c2w_interp[:3, :3] = q.rotation_matrix
    c2w_interp[:3, 3] = (1 - w) * c2w_src[:3, 3] + w * c2w_dst[:3, 3]

    w2c_interp = np.linalg.inv(c2w_interp)

    cam_interp.reset_extrinsic(w2c=w2c_interp)
    return cam_interp


def interpolate_fps_piecewise(cameras: List[Camera], fps: int) -> List[Camera]:
    """
    Interpolate the cameras using a piecewise function of provided cameras.
    """
    assert len(cameras) > 1, "Require >=2 cameras to generate interpolated trajectory"

    last_cam = cameras[0]
    interp_cameras = []

    # we assume generating 60 fps frames between every two sampled camera poses.
    sample_rate = 1.0 / fps * (cameras[1].time_s - cameras[0].time_s)

    print("Generate interpolated trajectories for rendering...")
    for idx in tqdm(range(1, len(cameras))):

        cam = cameras[idx]
        interp_cameras.append(copy.copy(last_cam))

        # skip the first sample which we have added
        samples_t = np.arange(
            last_cam.time_s + sample_rate, cam.time_s, sample_rate
        ).tolist()

        for _, ts in enumerate(samples_t):

            t_normalized = (ts - last_cam.time_s) / (cam.time_s - last_cam.time_s)

            # generate interpolated pose in between
            cam_interp = interpolate_piecewise(last_cam, cam, t_normalized)

            cam_interp.update_time(ts)

            interp_cameras.append(cam_interp)

        last_cam = cam

    return interp_cameras


class CameraDataset(Dataset):
    """
    An example usage:
    ```
    for batch in train_loader:
        gt_image = batch['image'].to("cuda")[0]
        scene.get_camera(batch['idx'].item(), subset="train")
    """

    def __init__(
        self,
        cameras: List[Camera],
        name: str = "train",
        render_only: bool = False,
        scale: float = 1.0,
    ):
        self.cameras = cameras
        self.scale = scale
        self.name = name
        self.render_only = render_only

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        cam = self.cameras[idx]

        if self.render_only:
            return {"idx": idx, "image_id": cam.camera_id, "subset": self.name}
        else:
            image, mask = cam.get_image(self.scale)

            output = {
                "idx": idx,
                "subset": self.name,
                "scene_name": cam.scene_name,
                "image_id": cam.camera_id,
                "image_name": cam.image_name,
                "camera_model": cam.camera_projection_model_gsplat,
                "image": image,
                # "w2c": cam.w2c_44_np,
                "K": cam.intrinsic,
            }

            # combine the exposure estimate into a per-pixel irradiance multiplier in the screenspace.
            # if a pixel is not visible, the corresponding irradiance will be zero.
            irradiance_multiplier = cam.exposure_multiplier * cam.radiance_weight
            if cam.vignette_image is not None:
                irradiance_multiplier *= cam.vignette_image
            if cam.valid_mask is not None:
                irradiance_multiplier *= cam.valid_mask
                if mask is not None:

                    if mask.ndim == 2:
                        mask = mask[None]

                    mask = mask * cam.valid_mask

            output["irradiance_multiplier"] = irradiance_multiplier

            # if sparse depth is available
            if cam.sparse_depth is not None:
                assert (
                    cam.sparse_point2d is not None
                ), "require to load 2d positions for 3d depth"
                assert (
                    cam.sparse_inv_depth is not None
                ), "require to load the inverse depth"
                output["sparse_depth"] = cam.sparse_depth
                output["sparse_point2d"] = cam.sparse_point2d
                output["sparse_inv_depth"] = cam.sparse_inv_depth
                output["sparse_inv_distance_std"] = cam.sparse_point_inv_distance_std

            if mask is not None:
                output["mask"] = mask

        return output


@dataclass
class CameraPropertyBasicRGB:
    camera_name: str = "camera-rgb"
    apply_tonemapping: bool = False


@dataclass
class CameraPropertyAriaRGB:
    """
    Camera properity for Aria RGB camera
    """

    camera_name: str = "camera-rgb"
    vignette_image: torch.Tensor = None
    mask_image: torch.Tensor = None
    apply_tonemapping: bool = False
    gamma4sRGB: float = 2.2
    exposure_gain_ratio_ref: float = 4e-2
    shutter_type: str = "rollingshutter"


@dataclass
class CameraPropertyAriaSLAM:
    """
    Camera properity for Aria SLAM camera
    """

    camera_name: str = "camera-slam"
    vignette_image: torch.Tensor = None
    apply_tonemapping: bool = False
    gamma4sRGB: float = 2.0
    exposure_gain_ratio_ref: float = 4e-2
    shutter_type: str = "globalshutter"


def camera_to_JSON(id, camera: Camera):
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.image_width,
        "height": camera.image_height,
        "c2w": camera.c2w_44.tolist(),
        "fov_x": camera.fov_x,
        "fov_y": camera.fov_y,
        "fx": camera.fx,
        "fy": camera.fy,
        "exposure_multiplier": camera.exposure_multiplier,
        "radiance_weight": camera.radiance_weight,
        "camera_name": camera.camera_name,
        "camera_modality": camera.camera_modality,
    }
    return camera_entry


def create_camera_from_JSON(data):

    c2w = np.array(data["c2w"])
    w2c = np.linalg.inv(c2w)
    camera = Camera(
        uid=data["id"],
        w2c=w2c,
        FoVx=data["fov_x"],
        FoVy=data["fov_y"],
        image_width=data["width"],
        image_height=data["height"],
        image_name=data["img_name"],
        camera_name=data["camera_name"],
        camera_modality=data["camera_modality"],
        exposure_duration_s=data["exposure_multiplier"],
        gain=1.0,
        radiance_weight=data["radiance_weight"],
    )

    return camera
