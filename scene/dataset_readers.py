import glob
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, NamedTuple, Optional

import cv2
import exifread
import numpy as np
import projectaria_tools.core.mps as mps
from PIL import Image
from scene.cameras import (
    AriaCamera,
    Camera,
    focal2fov,
    fov2focal,
    interpolate_aria_pose,
    interpolate_fps_piecewise,
)

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from tqdm import tqdm
from utils.point_utils import BasicPointCloud, fetchPly, project, storePly


def get_scene_info(scene_cfg):

    source_path = Path(scene_cfg.source_path)

    assert source_path.exists, f"Source path {source_path} does not exist!"

    if scene_cfg.input_format == "colmap":
        assert (
            source_path / "sparse"
        ).exists(), "colmap format requires sparse folder."
        print("Use colmap reader to initialize the scene.")
        scene_info = readColmapSceneInfo(
            path=source_path,
            scene_cfg=scene_cfg,
        )
    elif scene_cfg.input_format == "aria":
        scene_info = readAriaSceneInfo(input_folder=source_path, scene_cfg=scene_cfg)
    else:
        raise RuntimeError("cannot recognize the input format!!!")

    return scene_info


class SceneType(Enum):
    COLMAP = 1
    ARIA = 2


@dataclass
class SceneInfo:
    point_cloud: Optional[BasicPointCloud] = None
    point_source_path: Optional[str] = None
    scene_type: Optional[SceneType] = None
    all_cameras: Optional[SceneType] = None
    train_cameras: Optional[list] = None
    valid_cameras: Optional[list] = None
    test_cameras: Optional[list] = None
    scene_scale: Optional[int] = 1.0
    camera_labels: Optional[set] = None
    overwrite_sh_degree: Optional[int] = None

    @property
    def subset_to_cameras(self):
        return {
            "train": self.train_cameras,
            "valid": self.valid_cameras,
            "test": self.test_cameras,
        }


def estimate_scene_camera_scale(cam_infos):
    """
    Estimate the scale of the scene according to the camera distributions
    """
    c2ws = np.stack([cam.c2w_44_np for cam in cam_infos])

    camera_locations = c2ws[:, :3, 3]
    scene_center = np.mean(camera_locations, axis=0)
    dists = np.linalg.norm(camera_locations - scene_center, axis=1)
    scene_scale = np.max(dists)
    return scene_scale


def pinhole_camera_rectify(
    image: np.array, opencv_distortion_params: np.array, downsample: int = 1
):

    assert (
        len(opencv_distortion_params) == 8
    ), "currently only consumes distortion model with 4 radial distortions"

    fx, fy, cx, cy, k1, k2, k3, k4 = opencv_distortion_params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coeff = np.array([k1, k2, k3, k4])

    image_height, image_width = image.shape[:2]

    K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
        K, dist_coeff, (image_width, image_height), 0
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, dist_coeff, None, K_undist, (image_width, image_height), cv2.CV_32FC1
    )

    full_image_mask = np.ones((image_height, image_width))
    image_undistorted = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    mask_undistorted = cv2.remap(full_image_mask, mapx, mapy, cv2.INTER_LINEAR) * 255

    image_undistorted_rgba = np.concatenate(
        [image_undistorted, mask_undistorted[..., None]], axis=-1
    )

    return image_undistorted_rgba.astype(np.uint8)


def equidistant_camera_rectify(
    image: np.array, opencv_distortion_params: np.array, downsample: int = 1
):
    """
    Undistort opencv fisheye images to a equidistant fisheye projection
    https://github.com/zmliao/Fisheye-GS/blob/697cd86359efaae853a52f0bef9758b700390dc7/prepare_scannetpp.py#L58
    """
    assert (
        len(opencv_distortion_params) == 8
    ), "currently only consumes distortion model with 4 radial distortions"

    fx, fy, cx, cy, k1, k2, k3, k4 = opencv_distortion_params
    fx = fx // downsample
    fy = fy // downsample
    cx = cx // downsample
    cy = cy // downsample

    H, W = image.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing="xy",
    )
    x1 = (grid_x - cx) / fx
    y1 = (grid_y - cy) / fy
    theta = np.sqrt(x1**2 + y1**2)
    # theta = np.arctan(radius)
    r = 1.0 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8
    mapx = fx * x1 * r + cx
    mapy = fy * y1 * r + cy

    full_image_mask = np.ones((H, W))

    image_undistorted = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    mask_undistorted = cv2.remap(full_image_mask, mapx, mapy, cv2.INTER_LINEAR) * 255

    image_undistorted_rgba = np.concatenate(
        [image_undistorted, mask_undistorted[..., None]], axis=-1
    )

    return image_undistorted_rgba.astype(np.uint8)


def visualize_cameras_aria(
    cameras: List[Camera],
    point_cloud: BasicPointCloud,
    close_loop_traj,
    readout_time=16e6,
):
    """
    visualize points and cameras using a rerun viewer
    """
    import rerun as rr

    rr.init(f"Visualize the cameras", spawn=True)

    points = point_cloud.points
    colors = point_cloud.colors
    scale = 1000
    rr.log(
        f"world/points_3D",
        rr.Points3D(points * scale, colors=colors, radii=0.005 * scale),
        timeless=True,
    )

    for frame_idx, camera in enumerate(cameras):

        w, h = camera.image_width, camera.image_height
        c2w = camera.c2w_44

        # calibK = camera.intrinsic
        # w2c = np.linalg.inv(c2w)

        # point3d_cam = w2c[:3, :3] @ points_subsampled.T + w2c[:3, 3:4]
        # point3d_proj = calibK @ point3d_cam

        # u_proj = point3d_proj[0] / point3d_proj[2]
        # v_proj = point3d_proj[1] / point3d_proj[2]
        # z = point3d_proj[2]

        # mask = (u_proj > 0) & (u_proj < w) & (v_proj > 0) & (v_proj < h) & (z > 0)
        points2d = (
            camera.sparse_point2d.numpy()
        )  # np.stack([u_proj[mask], v_proj[mask]]).T

        if points2d is None:
            return 1e-3  # we almost cannot treat it as moving camera

        u = points2d[:, 0]
        v = points2d[:, 1]
        z = camera.sparse_depth.numpy()

        fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z

        t_start = camera._timestamp_ns - readout_time / 2.0
        t_end = camera._timestamp_ns + readout_time / 2.0
        pose_start = interpolate_aria_pose(close_loop_traj, t_start)
        pose_end = interpolate_aria_pose(close_loop_traj, t_end)

        w2c_rel = (
            pose_end.transform_world_device.inverse()
            @ pose_start.transform_world_device
        )

        pt3d = np.stack([x, y, z])

        u_proj, v_proj, z_proj, mask = project(
            pt3d,
            T_w2c=w2c_rel.to_matrix(),
            calibK=camera.intrinsic_np,
            frame_h=camera.image_height,
            frame_w=camera.image_width,
        )

        reproj_origin = points2d[mask]  # np.stack((u_proj, v_proj), axis=1)
        reproj_vector2d = np.stack([u_proj - u[mask], v_proj - v[mask]], axis=1)

        proj_offset = np.absolute(reproj_vector2d).max(axis=1)

        rr.set_time_sequence("frame_idx", frame_idx)

        rr.log(
            f"world/device/rgb",
            rr.Pinhole(resolution=[w, h], focal_length=camera.fx, fov_y=camera.fov_y),
        )

        image = camera.image
        rr.log(
            f"world/device/rgb/image",
            rr.Image(image).compress(jpeg_quality=75),
        )

        rr.log(
            f"world/device",
            rr.Transform3D(
                translation=c2w[:3, 3] * scale,
                mat3x3=c2w[:3, :3],
            ),
        )

        rr.log(
            f"world/device/rgb/points_2D",
            rr.Points2D(points2d.astype(np.int32), colors=[0, 200, 0], radii=2),
        )
        rr.log(
            f"world/device/rgb/arrows",
            rr.Arrows2D(
                origins=reproj_origin,
                vectors=reproj_vector2d,
                colors=[255, 0, 0],
                radii=3,
            ),
        )
        rr.log(
            f"world/device/reprojection_75_percentile",
            rr.Scalar(np.percentile(proj_offset, 75)),
        )
        rr.log(
            f"world/device/reprojection_50_percentile",
            rr.Scalar(np.percentile(proj_offset, 50)),
        )
        rr.log(
            f"world/device/reprojection_25_percentile",
            rr.Scalar(np.percentile(proj_offset, 25)),
        )


def visualize_cameras(cameras: List[Camera], point_cloud: BasicPointCloud):
    """
    visualize points projected to the cameras
    """
    import rerun as rr

    rr.init(f"Visualize the cameras", spawn=True)

    points = point_cloud.points
    colors = point_cloud.colors
    if len(points) > 5e6:
        points_subsampled = points[::5]
    else:
        points_subsampled = points
    scale = 1000
    rr.log(
        f"world/points_3D",
        rr.Points3D(points * scale, colors=colors, radii=0.005 * scale),
        timeless=True,
    )

    for frame_idx, camera in enumerate(cameras):

        w, h = camera.image_width, camera.image_height
        c2w = camera.c2w_44

        calibK = camera.intrinsic
        w2c = np.linalg.inv(c2w)

        point3d_cam = w2c[:3, :3] @ points_subsampled.T + w2c[:3, 3:4]
        point3d_proj = calibK @ point3d_cam

        u_proj = point3d_proj[0] / point3d_proj[2]
        v_proj = point3d_proj[1] / point3d_proj[2]
        z = point3d_proj[2]

        mask = (u_proj > 0) & (u_proj < w) & (v_proj > 0) & (v_proj < h) & (z > 0)
        points2d = np.stack([u_proj[mask], v_proj[mask]]).T

        rr.set_time_sequence("frame_idx", frame_idx)

        rr.log(
            f"world/device/rgb",
            rr.Pinhole(resolution=[w, h], focal_length=camera.fx, fov_y=camera.fov_y),
        )

        image = camera.image
        rr.log(
            f"world/device/rgb/image",
            rr.Image(image).compress(jpeg_quality=75),
        )

        rr.log(
            f"world/device",
            rr.Transform3D(
                translation=c2w[:3, 3] * scale,
                mat3x3=c2w[:3, :3],
            ),
        )

        rr.log(
            f"world/device/rgb/points_2D",
            rr.Points2D(points2d.astype(np.int32), colors=[0, 200, 0], radii=2),
        )


def readColmapCameras(
    cam_extrinsics,
    cam_intrinsics,
    rescale_factor,
    input_folder,
    mask_folder,
):
    cam_list = []
    num_frames = len(cam_extrinsics)

    # added for more general settings (that are compatible with Aria data format)
    if (input_folder / "transforms.json").exists():
        metadata_json = input_folder / "transforms.json"
        metadata = json.load(open(metadata_json, "r"))
        print(f"load metadata json from {metadata_json}")
        frames = metadata["frames"]

        # read in the metadata that is sorted by the image name as keys
        # it should all started with "images/" as relative path
        metadata_dict = {frame["timestamp"]: frame for frame in frames}
        timestamps = np.array(sorted([frame["timestamp"] for frame in frames]))
        vignette_image_path = input_folder / "vignette.png"
        mask_image_path = input_folder / "mask.png"
    else:
        metadata_dict = None
        vignette_image_path = None
        mask_image_path = None
        timestamps = []

    images_folder = input_folder / "images"
    if rescale_factor > 1:
        scaled_images_folders = input_folder / f"images_{rescale_factor}"
    else:
        scaled_images_folders = images_folder

    print("read COLMAP cameras with metadata...")
    for idx, key in tqdm(enumerate(cam_extrinsics)):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        if extr.name != "":
            image_file = os.path.basename(extr.name)
        else:
            # this is the naming in Hyperreal datasets
            image_file = f"{extr.id:05d}_{extr.id:05d}.png"

        image_path = scaled_images_folders / image_file
        if not image_path.exists():
            print(f"Image does not exists: {image_path} Skip loading this camera!")
            continue

        img_loaded = Image.open(image_path)
        width, height = img_loaded.size

        exposure_duration_s = 1
        gain = 1.0

        # read in the exif metadata to acquire ISO information from ZIPNERF dataset.
        # note the detailed EXIF metadata only exists in the full resolution images.
        # For other dataset, it will skip this part.
        if image_file.lower().endswith((".jpg", ".jpeg")):
            with open(images_folder / image_file, "rb") as f:
                f.seek(0)
                tags = exifread.process_file(f)

                if "EXIF ISOSpeedRatings" in tags:
                    iso = tags["EXIF ISOSpeedRatings"].values[0]
                    # calculate a digital gain with ISO basis 100
                    gain = np.log(iso / 100)

                if "EXIF ExposureTime" in tags:
                    exposure_time_tag = tags.get("EXIF ExposureTime").values[
                        0
                    ]  # e.g. 1/400
                    exposure_duration_s = eval(str(exposure_time_tag))

        image_name = os.path.basename(image_path).split(".")[0]

        uid = intr.id
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(extr.qvec)
        w2c[:3, 3] = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0] / rescale_factor
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            cx, cy = intr.params[1:3] / rescale_factor
            camera_projection_model = "linear"
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0] / rescale_factor
            focal_length_y = intr.params[1] / rescale_factor
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            cx, cy = intr.params[2:4] / rescale_factor
            camera_projection_model = "linear"
        elif intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0] / rescale_factor
            focal_length_y = intr.params[0] / rescale_factor
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            assert intr.params[3] < 1e-2, "distortion should not be too large."
            cx, cy = intr.params[1:3] / rescale_factor
            camera_projection_model = "linear"
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0] / rescale_factor
            focal_length_y = intr.params[1] / rescale_factor
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            cx, cy = intr.params[2:4] / rescale_factor
            camera_projection_model = "linear"

            # save the undistorted images in disk
            undistorted_scaled_images_folder = (
                input_folder / f"images_{rescale_factor}_undistorted"
            )
            # save the undistorted image as RGBA images
            undistorted_image_path = (
                undistorted_scaled_images_folder / f"{image_name}.png"
            )
            if not undistorted_image_path.exists():
                image_undistorted_rgba = pinhole_camera_rectify(
                    np.array(img_loaded), intr.params, downsample=rescale_factor
                )

                if not undistorted_scaled_images_folder.exists():
                    undistorted_scaled_images_folder.mkdir()

                Image.fromarray((image_undistorted_rgba)).save(undistorted_image_path)
                print(
                    f"Cache rectified linear projection model to {undistorted_image_path}"
                )

            image_path = undistorted_image_path

        elif intr.model == "OPENCV_FISHEYE":
            focal_length_x = intr.params[0] / rescale_factor
            focal_length_y = intr.params[1] / rescale_factor
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            cx, cy = intr.params[2:4] / rescale_factor
            camera_projection_model = "spherical"

            undistorted_scaled_images_folder = (
                input_folder / f"images_{rescale_factor}_undistorted"
            )
            # save the undistorted image as RGBA images
            undistorted_image_path = (
                undistorted_scaled_images_folder / f"{image_name}.png"
            )

            if not undistorted_image_path.exists():
                image_undistorted_rgba = equidistant_camera_rectify(
                    np.array(img_loaded), intr.params, downsample=rescale_factor
                )

                if not undistorted_scaled_images_folder.exists():
                    undistorted_scaled_images_folder.mkdir()

                Image.fromarray((image_undistorted_rgba)).save(undistorted_image_path)
                print(
                    f"Cache rectified spherical projection model to {undistorted_image_path}"
                )

            image_path = undistorted_image_path

            # make the spherical rectification and save the output images
        else:
            assert False, f"Colmap camera model not handled: {intr.model}!"

        if not os.path.exists(image_path):
            print(
                f"Exclude camera {key} because its corresponding image {image_path} does not exist."
            )
            continue

        if mask_folder is not None:
            mask_path = os.path.join(mask_folder, image_file)
        else:
            mask_path = None

        # this is specifically for Aria data input
        if metadata_dict is not None:
            # image file example 'camera-rgb_4790379631137.0.png'
            # get timestamp 4790379631137.0, the +130000 was the offset for colmap poses that were generated
            # locate the right frame information using image path
            try:
                seek_timestamp = float(image_file[:-4].split("_")[-1]) + 130000
                time_index = np.searchsorted(timestamps, seek_timestamp)
                if time_index == 0 and seek_timestamp < timestamps[0]:
                    continue  # frame happens before video being recorded
                if time_index >= len(timestamps):
                    continue  # frame happens after video being recorded
                seek_timestamp = int(timestamps[time_index])
                frame = metadata_dict[seek_timestamp]
                exposure_duration_s = frame["exposure_duration_s"]
                gain = frame["gain"]
            except:
                raise RuntimeError(
                    f"Cannot read exposure & gain for frame {idx} with image file {image_file}"
                )

        cam = Camera(
            uid=uid,
            w2c=w2c,
            FoVx=FovX,
            FoVy=FovY,
            cx=cx,
            cy=cy,
            image_width=width,
            image_height=height,
            image_name=image_name,
            image_path=image_path,
            mask_path=mask_path,
            scale=1.0,
            camera_name="camera-rgb",
            camera_projection_model=camera_projection_model,
            camera_modality="rgb",
            exposure_duration_s=exposure_duration_s,
            gain=gain,
        )

        if cam.vignette_image is None and vignette_image_path is not None:
            cam.set_vignette_image(
                vignette_image_path=vignette_image_path,
                camera_name=metadata["camera_label"],
            )
        if cam.valid_mask is None and mask_image_path is not None:
            cam.set_valid_mask(
                mask_image_path=mask_image_path, camera_name=metadata["camera_label"]
            )
        cam_list.append(cam)

    return cam_list


def readColmapSceneInfo(
    path: str, scene_cfg, testhold: int = 8, visualize: bool = False
):
    """
    path: the scene path for Colmap subfolder. It supports the common LLFF, mipnerf dataset or self-generated dataset.
    images: the folder name for images
    testhold: hold every N images in the test set. According to convention, use 8 as default according to the nerf dataset convention.
    """

    images = scene_cfg.images
    masks = scene_cfg.masks
    factor = scene_cfg.data_factor

    assert factor in [1, 2, 4, 8], "data rescale factor can only exist within 1,2,4,8!"

    cam_extrinsics, cam_intrinsics = None, None
    try:
        cameras_extrinsic_file = path / "sparse" / "0" / "images.bin"
        cameras_intrinsic_file = path / "sparse" / "0" / "cameras.bin"
        cam_extrinsics = read_extrinsics_binary(str(cameras_extrinsic_file))
        cam_intrinsics = read_intrinsics_binary(str(cameras_intrinsic_file))
    except:
        cameras_extrinsic_file = path / "sparse" / "0" / "images.txt"
        cameras_intrinsic_file = path / "sparse" / "0" / "cameras.txt"
        cam_extrinsics = read_extrinsics_text(str(cameras_extrinsic_file))
        cam_intrinsics = read_intrinsics_text(str(cameras_intrinsic_file))

    assert (cam_extrinsics is not None) and (
        cam_intrinsics is not None
    ), "did not find the right structure to read input!"

    if masks != "":
        mask_folder = path / masks
    else:
        mask_folder = None

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        rescale_factor=factor,
        input_folder=path,
        mask_folder=mask_folder,
    )

    # calculate the average the exposure & gain ratio. does not affect if the input does not contain such information
    exp_all = np.array([cam.exposure_multiplier for cam in cam_infos_unsorted])
    exp_median = np.median(exp_all)
    for cam in cam_infos_unsorted:
        cam.radiance_weight = 1.0 / exp_median

    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    print(
        f"There are a total of {len(cam_infos)} cameras to be used in reconstruction."
    )

    valid_idx = np.arange(0, len(cam_infos), testhold)
    # Limit the number of validation images for a single scene to 2000
    train_idx = np.setdiff1d(np.arange(0, len(cam_infos)), valid_idx)

    # to make train test split consistent with Aria inputs
    train_camera_infos = [cam_infos[i] for i in train_idx]
    valid_camera_infos = [cam_infos[i] for i in valid_idx]
    test_camera_infos = valid_camera_infos

    print(
        f"We will use {len(train_camera_infos)} for training, {len(valid_camera_infos)} for validation."
    )

    scene_scale = estimate_scene_camera_scale(cam_infos)

    ply_path = str(path / "sparse/0/points3D.ply")
    bin_path = str(path / "sparse/0/points3D.bin")
    txt_path = str(path / "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # visualize the dataset.
    if visualize:
        visualize_cameras(train_camera_infos, pcd)

    scene_info = SceneInfo(
        point_cloud=pcd,
        all_cameras=cam_infos,
        train_cameras=train_camera_infos,
        valid_cameras=valid_camera_infos,
        test_cameras=test_camera_infos,
        scene_scale=scene_scale,
        scene_type=SceneType.COLMAP,
        camera_labels=["camera-rgb"],
    )

    return scene_info


def read_Aria_transform_json(
    transform_paths: List[Path],
    high_freq_trajectory: List[mps.ClosedLoopTrajectoryPose],
    input_folder: Path = None,
    scene_name: str = "none",
    start_timestamp_ns: int = -1,
    end_timestamp_ns: int = -1,
    read_vignette: bool = True,
    read_mask: bool = True,
    sample_interval_s: float = 0,
) -> List[Camera]:
    frames = []
    for transform_path in transform_paths:
        with open(transform_path) as json_file:
            transforms = json.loads(json_file.read())
        frames += transforms["frames"]

    camera_names = set()

    time_start_with_pose = (
        high_freq_trajectory[0].tracking_timestamp.total_seconds() * 1e9
    )
    time_end_with_pose = (
        high_freq_trajectory[-1].tracking_timestamp.total_seconds() * 1e9
    )

    # This sorts the frame list by first camera_name, then by capture time.
    frames.sort(key=lambda f: f["image_path"])

    all_cam_list = []
    last_sampled_timestamp_ns = 0
    for idx, frame in enumerate(frames):
        # skip frames according to the sample interval
        # if frame["timestamp"] <= last_sampled_timestamp_ns + sample_interval_s * 1e9:
        #     continue

        if input_folder is None:
            image_path_full = None
            image_name = None
        else:
            image_path_full = str(input_folder / frame["image_path"])
            image_name = image_path_full.split("/")[-1].split(".")[0]

            if not os.path.exists(image_path_full):
                print(f"{image_path_full} does not exist. Will skip!")
                continue

        # The center row timestamp for the camera. The 1e3 is a temporary fix for arcata camera
        timestamp_center = frame["timestamp"]

        if start_timestamp_ns > 0 and timestamp_center < start_timestamp_ns:
            # print(f"skip frames that before time {start_timestamp_ns}")
            continue

        if timestamp_center < time_start_with_pose + 1e6:  # skip first 100 ms
            print(
                f"skip frames that before time {start_timestamp_ns} in trajectory with valid pose"
            )
            continue

        if end_timestamp_ns > 0 and timestamp_center >= end_timestamp_ns:
            # print(f"skip frame that after time {end_timestamp_ns}")
            continue

        if timestamp_center > time_end_with_pose - 1e6:  # skip last 100 ms
            print(
                f"skip frames that after time {time_end_with_pose} in trajectory with valid pose"
            )
            continue

        if transforms["camera_label"].startswith("camera-slam"):
            camera_modality = "monochrome"
            camera_names.update(["camera-slam"])
            is_rolling_shutter = False
            rolling_shutter_index_image_path = None
        elif transforms["camera_label"].startswith("camera-rgb"):
            camera_modality = "rgb"
            camera_names.update(["camera-rgb"])
            is_rolling_shutter = True
            rolling_shutter_index_image_path = input_folder / "image_index.png"
        elif transforms["camera_label"].startswith(
            "camera_rgb"
        ):  # used for quest camera only
            camera_modality = "rgb"
            camera_names.update((["camera-rgb"]))
            is_rolling_shutter = True
            rolling_shutter_index_image_path = input_folder / "image_index.png"
        else:
            raise NotImplementedError(
                f"Unrecognized cameras labels {transforms['camera_label']}"
            )

        if (
            "mask_path" in frame.keys()
            and frame["mask_path"] != ""
            and input_folder is not None
        ):
            mask_path_full = input_folder / frame["mask_path"]
            assert mask_path_full.exists(), "mask file in the transform does not exist!"
            mask_path_full = str(mask_path_full)
        else:
            mask_path_full = None

        fx, fy, cx, cy = frame["fx"], frame["fy"], frame["cx"], frame["cy"]
        width, height = frame["w"], frame["h"]
        fovx = focal2fov(fx, width)
        fovy = focal2fov(fy, height)

        camera2device = np.asarray(frame["camera2device"])

        if "sparse_depth" in frame.keys():
            sparse_depth_path = input_folder / frame["sparse_depth"]
        else:
            sparse_depth_path = None

        cam = AriaCamera(
            uid=idx,
            closed_loop_traj=high_freq_trajectory,
            camera2device=camera2device,
            camera_name=transforms["camera_label"],
            camera_modality=camera_modality,
            is_rolling_shutter=is_rolling_shutter,
            rolling_shutter_index_image_path=rolling_shutter_index_image_path,
            timestamp_ns=timestamp_center,
            FoVx=fovx,
            FoVy=fovy,
            image_width=width,
            image_height=height,
            cx=cx,
            cy=cy,
            image_name=image_name,
            image_path=image_path_full,
            mask_path=mask_path_full,
            sparse_depth_path=sparse_depth_path,
            camera_projection_model=transforms["camera_model"],
            exposure_duration_s=frame["exposure_duration_s"],
            gain=frame["gain"],
            scene_name=scene_name,
            readout_time_ns=frame["timestamp_read_end"] - frame["timestamp_read_start"],
        )

        # if cam.sparse_depth is None:
        #     print(f"Camera {idx} at {frame['timestamp_read_start']} does not have valid depth. We will not use this pose in training/validation.")
        #     continue

        if cam.vignette_image is None and read_vignette:
            cam.set_vignette_image(
                vignette_image_path=input_folder / "vignette.png",
                camera_name=transforms["camera_label"],
            )
        if (
            transforms["camera_label"] == "camera-rgb"
            and cam.valid_mask is None
            and read_mask
        ):
            cam.set_valid_mask(
                mask_image_path=input_folder / "mask.png",
                camera_name=transforms["camera_label"],
            )

        all_cam_list.append(cam)

        last_sampled_timestamp_ns = frame["timestamp"]

    print(
        f"Found {len(all_cam_list)} cameras given criterion among {len(frames)} frames"
    )

    # calculate the average the exposure & gain ratio
    exp_all = np.array([cam.exposure_multiplier for cam in all_cam_list])
    exp_median = np.median(exp_all)

    for cam in all_cam_list:
        cam.radiance_weight = 1.0 / exp_median

    return all_cam_list, camera_names


def read_render_transform_json(
    transform: dict,
) -> List[Camera]:
    """
    read it from the transform file generated from interactive visualize tools.
    """
    # orientation_transform = np.asarray(transform["orientation_transform"]).reshape(4, 4)

    camera_names = set()

    all_cam_list = []
    for idx, frame in enumerate(transform):
        width = frame["width"]
        height = frame["height"]
        camera_names.update([frame["camera_name"]])

        c2w = np.asarray(frame["c2w"]).reshape(4, 4)
        # c2w = orientation_transform @ c2w
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        # c2w[..., :3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)

        cam = Camera(
            uid=frame["id"],
            w2c=w2c,
            FoVx=frame["fov_x"],
            FoVy=frame["fov_y"],
            image_width=width,
            image_height=height,
            camera_name=frame["camera_name"],
            exposure_duration_s=frame["exposure_multiplier"],
            radiance_weight=frame["radiance_weight"],
            gain=1.0,
        )

        all_cam_list.append(cam)

    return all_cam_list, camera_names


def readAriaSceneInfo(
    input_folder: Path,
    scene_cfg,
    visualize: bool = False,
):
    scene_name = scene_cfg.scene_name
    data_format = scene_cfg.data_format
    train_split = scene_cfg.train_split
    start_timestamp_ns = scene_cfg.start_timestamp_ns
    end_timestamp_ns = scene_cfg.end_timestamp_ns

    trajectory_file = input_folder / "closed_loop_trajectory.csv"
    closed_loop_traj = mps.read_closed_loop_trajectory(str(trajectory_file))

    # Go through the transforms and create the camera infos
    if train_split == "fixed":
        train_transform_paths = glob.glob(str(input_folder / "transforms_train.json"))
        train_cam_list, camera_names = read_Aria_transform_json(
            transform_paths=train_transform_paths,
            high_freq_trajectory=closed_loop_traj,
            input_folder=input_folder,
            scene_name=scene_name,
            start_timestamp_ns=start_timestamp_ns,
            end_timestamp_ns=end_timestamp_ns,
        )

        transform_valid_paths = glob.glob(str(input_folder / "transforms_valid.json"))
        assert len(transform_valid_paths) > 0, "No transform_valid.json found"
        valid_cam_list, valid_camera_names = read_Aria_transform_json(
            transform_paths=transform_valid_paths,
            high_freq_trajectory=closed_loop_traj,
            input_folder=input_folder,
            scene_name=scene_name,
        )
        camera_names.update(valid_camera_names)
    else:
        transform_paths = glob.glob(str(input_folder / data_format))
        all_cam_list, camera_names = read_Aria_transform_json(
            transform_paths=transform_paths,
            high_freq_trajectory=closed_loop_traj,
            input_folder=input_folder,
            scene_name=scene_name,
            start_timestamp_ns=start_timestamp_ns,
            end_timestamp_ns=end_timestamp_ns,
        )

    # Read the test set cameras if there are available
    test_view_folder = input_folder / "test_views"
    if test_view_folder.exists():
        test_json_paths = glob.glob(str(test_view_folder / "**/transforms_test.json"))
        print(f"Read test cameras from transforms_test.json within {test_view_folder}")

        test_cam_list, test_camera_names = [], set()
        for test_json_path in test_json_paths:
            test_input_folder = Path(test_json_path).parent
            test_view_closed_loop_traj_path = (
                test_input_folder / "closed_loop_trajectory.csv"
            )
            assert (
                test_view_closed_loop_traj_path.exists()
            ), f"cannot find test view closed loop trajectory! {test_view_closed_loop_traj_path}"

            test_closed_loop_traj = mps.read_closed_loop_trajectory(
                str(test_view_closed_loop_traj_path)
            )
            test_cam, test_camera_name = read_Aria_transform_json(
                transform_paths=[test_json_path],
                high_freq_trajectory=test_closed_loop_traj,
                input_folder=test_input_folder,
                scene_name=scene_name,
            )
            test_cam_list += test_cam
            test_camera_names.update(test_camera_name)

        # rest the test camera radiance weight to be consistnt with train views

        for cam in test_cam_list:
            cam.radiance_weight = all_cam_list[0].radiance_weight

    elif (input_folder / "transforms_test.json").exists():
        transform_test_path = input_folder / "transforms_test.json"
        print("Read test cameras from transforms_test.json")
        test_cam_list, test_camera_names = read_Aria_transform_json(
            transform_paths=[transform_test_path],
            high_freq_trajectory=closed_loop_traj,
            input_folder=input_folder,
            scene_name=scene_name,
        )
        camera_names.update(test_camera_names)
    else:
        test_cam_list, test_camera_names = None, None

    print(f"Using cameras: {camera_names}")

    if train_split == "all":
        print("will use all the cameras for the training & test split")
        train_camera_infos = all_cam_list
        valid_camera_infos = all_cam_list
        test_camera_infos = all_cam_list
    elif train_split == "fixed":
        train_camera_infos = train_cam_list
        valid_camera_infos = valid_cam_list
        test_camera_infos = test_cam_list
    else:
        if train_split == "4-1":
            print(
                "will use 4/5 of split for training, and 1/5 for validation and testing."
            )
            valid_interval = 5
        elif train_split == "7-1":
            print(
                "will use 7/8 of split for training, and 1/8 for validation and testing."
            )
            valid_interval = 8
        else:
            raise RuntimeError(f"Cannot recognize train_split: {train_split}")

        valid_idx = np.arange(0, len(all_cam_list), valid_interval)
        train_idx = np.setdiff1d(np.arange(0, len(all_cam_list)), valid_idx)

        train_camera_infos = [all_cam_list[i] for i in train_idx]
        valid_camera_infos = [all_cam_list[i] for i in valid_idx]

    if test_cam_list is not None:
        test_camera_infos = test_cam_list
    else:
        print("Did not find test cameras. Use validation cameras as test cameras")
        test_camera_infos = valid_camera_infos

    scene_scale = estimate_scene_camera_scale(all_cam_list)

    # read colored points if they are available, which does not offer too much value now. We are skipping this.
    # colored_points_path = input_folder / "colored_points.ply"
    # if False: #'camera-rgb' in camera_names and  colored_points_path.exists():
    #     print("Preprocessed 3D points with color exists. Will initialize using this!")
    #     ply_path = colored_points_path
    # else:
    # Get the real path if the input is a symbolic path
    points_path = (input_folder / "semidense_points.csv.gz").resolve()

    # read pointcloud
    points = mps.read_global_point_cloud(str(points_path))
    # filter the point cloud by inverse depth and depth
    filtered_points = []
    for point in points:
        if point.inverse_distance_std < 0.01 and point.distance_std < 0.02:
            filtered_points.append(point)

    # example: get position of this point in the world coordinate frame
    points_world = []
    for point in filtered_points:
        position_world = point.position_world
        points_world.append(position_world)

    xyz = np.stack(points_world, axis=0)
    point_cloud = BasicPointCloud(points=xyz, colors=None, normals=None)

    if visualize:
        # visualize_cameras_aria(train_camera_infos, point_cloud, closed_loop_traj)
        visualize_cameras(train_camera_infos, point_cloud)

    # save it together with with train info
    scene_info = SceneInfo(
        point_cloud=point_cloud,
        point_source_path=points_path,
        all_cameras=all_cam_list,
        train_cameras=train_camera_infos,
        valid_cameras=valid_camera_infos,
        test_cameras=test_camera_infos,
        scene_scale=scene_scale,
        scene_type=SceneType.ARIA,
        camera_labels=camera_names,
    )

    return scene_info


def readRenderInfo(
    render_cfg: dict,
    split: str = "valid",
    # load_nerf_normalization: str = "",
    # load_aria_sensor_config: bool = False,
):
    # Go through the transforms and create the camera infos
    with open(render_cfg.render_json) as json_file:
        transform = json.loads(json_file.read())

    assert (
        split in transform.keys()
    ), f"{split} is not in transform split of {transform.keys()}"

    render_cameras, render_names = read_render_transform_json(
        transform=transform[split],
    )

    # input_folder = Path(render_cfg.render_json).parent
    # trajectory_file = input_folder / "closed_loop_trajectory.csv"
    # closed_loop_traj = mps.read_closed_loop_trajectory(str(trajectory_file))

    # render_cameras, render_names = read_Aria_transform_json(
    #     transform_paths=[render_cfg.render_json],
    #     high_freq_trajectory=closed_loop_traj,
    #     input_folder=input_folder,
    #     read_vignette=False,
    #     read_mask=False,
    #     start_timestamp_ns=render_cfg.start_timestamp_ns,
    #     end_timestamp_ns=render_cfg.end_timestamp_ns,
    # )

    # valid_idx = np.arange(0, len(render_cameras), render_cfg.sample_interval)
    # render_cameras = [render_cameras[i] for i in valid_idx]

    if not np.isclose(render_cfg.gain_amplify, 1.0):
        for cam in render_cameras:
            cam.amplify_gain(render_cfg.gain_amplify)

    if render_cfg.render_fps > 1:
        render_cameras = interpolate_fps_piecewise(
            render_cameras, render_cfg.render_fps
        )

    # reset the render camera if needed
    for cam in render_cameras:
        # digital zoom
        if (render_cfg.zoom - 1.0) > 1e-1:
            cam.zoom(render_cfg.zoom)

        if (
            render_cfg.render_height > 0
            and render_cfg.render_height != cam.image_height
        ):
            cam.image_height = render_cfg.render_height

        if render_cfg.aspect_ratio > 0 and render_cfg.aspect_ratio != cam.aspect_ratio:
            cam.aspect_ratio = render_cfg.aspect_ratio

    # it will be used to normalize the Gaussian Points.
    scene_info = SceneInfo(
        point_cloud=None,
        all_cameras=render_cameras,
        train_cameras=[],
        valid_cameras=[],
        test_cameras=render_cameras,
        scene_scale=1.0,
        scene_type=SceneType.ARIA,
        camera_labels=render_names,
    )

    return scene_info


def aggregate_scene_infos(scene_infos):
    """
    Aggregate multiple scene infos into one
    """
    points_source_path = []
    points_agg = []
    colors_agg = []
    normals_agg = []
    all_cameras = []
    train_cameras = []
    valid_cameras = []
    test_cameras = []
    scene_type = None
    camera_labels = set()

    for scene_info in scene_infos:
        skip_merge = False
        for exisitng_source_path in points_source_path:
            if scene_info.point_source_path.samefile(exisitng_source_path):
                print(
                    f"There are duplicate point clouds read from {scene_info.point_source_path}. Skip merging."
                )
                skip_merge = True
                break

        if not skip_merge:
            points_source_path.append(scene_info.point_source_path)
            points_agg.append(scene_info.point_cloud.points)
            if scene_info.point_cloud.colors:
                colors_agg.append(scene_info.point_cloud.colors)
            if scene_info.point_cloud.normals:
                normals_agg.append(scene_info.point_cloud.normals)

        # if 'camera-rgb' in scene_info.camera_labels:
        #     # duplicate the number of RGB camera to balance modalities
        #     train_cameras += scene_info.train_cameras * 2
        # else:
        #     train_cameras += scene_info.train_cameras

        all_cameras += scene_info.all_cameras
        train_cameras += scene_info.train_cameras
        valid_cameras += scene_info.valid_cameras
        test_cameras += scene_info.test_cameras

        if scene_type is None:
            scene_type = scene_info.scene_type
        else:
            assert (
                scene_type == scene_info.scene_type
            ), "aggregated scene infos need to have the same scene type"

        camera_labels.update(scene_info.camera_labels)

    # Limit the number of validation images for a single scene to 5000
    if len(valid_cameras) > 5000:
        raise Warning(
            f"The validation camera number is huge! It will be very slow {len(valid_cameras)}"
        )

    all_cameras = sorted(all_cameras, key=lambda camera: camera.time_s)
    train_cameras = sorted(train_cameras, key=lambda camera: camera.time_s)
    valid_cameras = sorted(valid_cameras, key=lambda camera: camera.time_s)
    test_cameras = sorted(test_cameras, key=lambda camera: camera.time_s)

    # recalculate the exposure multiplier
    # exp_all = np.array([cam.exposure_multiplier for cam in all_cameras])
    # exp_median = np.median(exp_all)
    # global_radiance_weight = 1.0 / exp_median
    # for cam in all_cameras:
    #     cam.radiance_weight = global_radiance_weight
    # for cam in train_cameras:
    #     cam.radiance_weight = global_radiance_weight
    # for cam in valid_cameras:
    #     cam.radiance_weight = global_radiance_weight
    # for cam in test_cameras:
    #     cam.radiance_weight = global_radiance_weight

    points_agg = np.concatenate(points_agg, axis=0)
    if len(colors_agg) > 0:
        colors_agg = np.concatenate(colors_agg, axis=0)
        assert len(colors_agg) == len(
            points_agg
        ), "number of points that have colors doest not match point number"
    else:
        colors_agg = None

    if len(normals_agg) > 0:
        normals_agg = np.concatenate(normals_agg, axis=0)
        assert len(normals_agg) == len(
            points_agg
        ), "number of points that have normals doest not match point number"
    else:
        normals_agg = None

    pcd = BasicPointCloud(points=points_agg, colors=colors_agg, normals=normals_agg)

    scene_scale = estimate_scene_camera_scale(all_cameras)

    scene_info_agg = SceneInfo(
        point_cloud=pcd,
        point_source_path=points_source_path,
        all_cameras=all_cameras,
        train_cameras=train_cameras,
        valid_cameras=valid_cameras,
        test_cameras=test_cameras,
        scene_scale=scene_scale,
        scene_type=scene_type,
        camera_labels=camera_labels,
    )

    return scene_info_agg
