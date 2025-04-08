import os
import torch
import glob

from typing import List
from torch.utils.data import Dataset

import numpy as np

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import SceneInfo
from scene.dataset_readers import get_scene_info, aggregate_scene_infos, readRenderInfo
from scene.cameras import Camera, CameraDataset

from utils.point_utils import storePly

def load_or_init_gaussians(
    gaussians,
    scene_info: SceneInfo,
    model_path: str,
    load_iteration: int = None,
):
    loaded_iter = None
    if load_iteration:
        if load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
        else:
            loaded_iter = load_iteration
        print("Loading trained model at iteration {}".format(loaded_iter))

    if loaded_iter:
        gaussians.load_ply(os.path.join(model_path,
                                        "point_cloud",
                                        "iteration_" + str(loaded_iter),
                                        "point_cloud.ply"),
                            og_number_points=len(scene_info.point_cloud.points))
    else:
        gaussians.create_from_pcd(scene_info.point_cloud, scene_info.nerf_normalization["radius"])
        
    if not loaded_iter:
        storePly(
            os.path.join(model_path, "input.ply"), 
            scene_info.point_cloud.points,
            scene_info.point_cloud.colors * 255,
        )
        

    return gaussians


def initialize_scene_info(cfg):
    """
    Initialize the 3D scene and cameras from configurations
    """

    data_dir, scene_names = cfg.scene_name.split("/")
    scene_names = scene_names.split("+")

    source_paths = []
    for scene_name in scene_names: 
        source_paths += [d for d in glob.glob(f"{cfg.data_root}/{data_dir}/{scene_name}") if os.path.isdir(d)]

    if len(source_paths) == 1:
        cfg.source_path = source_paths[0]
        scene_info = get_scene_info(cfg)
    else: 
        print(f"There are {len(source_paths)} paths provided. Will extract and merge them in one scene.")
        scenes_info = []

        for source_path in source_paths: 
            print(f"load and aggregated {source_path}")

            cfg.source_path = source_path
            scene_info = get_scene_info(cfg)
            scenes_info.append(scene_info)

        scene_info = aggregate_scene_infos(scenes_info)
    return scene_info


def initialize_eval_info(cfg): 
    cfg.model.use_3d_smooth_filter=False
    cfg.viewer.use_trainer_viewer=False
    
    if cfg.scene.start_timestamp_ns > 0:
        cfg.render.start_timestamp_ns = cfg.scene.start_timestamp_ns
        cfg.render.end_timestamp_ns = cfg.scene.end_timestamp_ns

    scene_info = get_scene_info(cfg.scene)
    return scene_info


def initialize_render_info(cfg):     
    # automatically reset a few parameters in rendering 
    cfg.model.use_3d_smooth_filter=False
    cfg.viewer.use_trainer_viewer=False

    # overwrite start & end timestamp if specified in the scene
    if cfg.scene.start_timestamp_ns > 0:
        cfg.render.start_timestamp_ns = cfg.scene.start_timestamp_ns

    if cfg.scene.end_timestamp_ns > 0:
        cfg.render.end_timestamp_ns = cfg.scene.end_timestamp_ns

    return readRenderInfo(
        render_cfg = cfg.render, 
    )


class ConcatCameraDataset(Dataset):
    def __init__(self, datasets):
        self.total_cameras = 0 
        dataset_nums = [-1]
        for d in datasets: 
            self.total_cameras += len(d)
            dataset_nums.append(self.total_cameras-1)
        self.datasets = datasets
        self.dataset_nums_acc = np.array(dataset_nums)

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.dataset_nums_acc, idx, side='left') - 1
        sample_idx = idx - (self.dataset_nums_acc[dataset_idx] + 1)
        # print(f"idx: {idx} dataset_idx: {dataset_idx}, len: {len(self.datasets[dataset_idx])} sample_idx: {sample_idx}")
        return self.datasets[dataset_idx].__getitem__(sample_idx)

    def __len__(self):
        return self.total_cameras


def get_data_loader(
        cameras: List[Camera], 
        subset: str, 
        batch_size: int = 1,
        scale: float = 1.0, 
        shuffle: bool = False, 
        render_only: bool = False,
        num_workers=12, 
    ):

    dataset = CameraDataset(
        cameras=cameras, 
        name = subset,
        render_only = render_only,
        scale = scale
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
    )
    return dataloader
