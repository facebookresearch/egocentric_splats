import os
from scene.dataset_readers import SceneInfo
from model.vanilla_gsplat import VanillaGSplat
from model.GS2D_gsplat import Gaussians2D

def load_from_model_path(
    cfg,
    model_path: str, 
    ply_subpath: str = None,
    bilateral_grid_path: str = None,
):
    """
    load a pretrain model from a given path
    """
    assert model_path is not None, "model_path must be specified"
    assert os.path.exists(model_path), f"model_path {model_path} does not exist"

    if cfg.train_model == "vanillaGS":
        model = VanillaGSplat(cfg=cfg, scene_info=SceneInfo(), train_mode=False)
        print("Load model with vanilla Gaussian rasterizer")
    elif cfg.train_model == "2dgs":
        model = Gaussians2D(cfg=cfg, scene_info=SceneInfo(), train_mode=False)
        print("Load model with 2D Gaussian rasterizer")
    else: 
        raise RuntimeError(f"cannot recognize the trained model {cfg.train_model}")

    model.load_ply(os.path.join(model_path, ply_subpath))

    if bilateral_grid_path is not None: 
        model.load_bilateral_grid(bilateral_grid_path)

    return model