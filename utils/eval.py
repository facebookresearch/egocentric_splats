import json

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def erode_mask(mask, target_size, thres=0.5):
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype == np.float32:
        mask = (mask * 255).clip(0, 255).astype(np.uint8)
    # shrink mask by a small margin to prevent inaccurate mask boundary.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    if target_size is not None:
        mask = cv2.resize(mask, (target_size, target_size))
    return (mask > 0.5).astype(np.float32)


def calc_depth_distance_scale_invariant(depth_pred, depth_gt, mask_gt):
    """
    Calculate the scale invariant depth distance.
    Note that this will slightly erode the mask (5x5) to prevent boundary artifacts

    params:
        depth_pred: numpy.ndarray of shape [H, W].
        depth_gt: numpy.ndarray of shape [H, W].
        mask_gt: numpy.ndarray of shape [H, W]. ground truth foreground mask.
    """
    assert depth_pred.shape == depth_gt.shape
    mask_gt = erode_mask(mask_gt, depth_pred.shape[0])
    depth_gt_masked = depth_gt[np.where(mask_gt > 0.5)]
    depth_pred_masked = depth_pred[np.where(mask_gt > 0.5)]
    if (depth_pred_masked**2).sum() <= 1e-6:
        depth_pred = np.ones_like(depth_gt)
        depth_pred_masked = depth_pred[np.where(mask_gt > 0.5)]
    scale = (depth_gt_masked * depth_pred_masked).sum() / (depth_pred_masked**2).sum()
    depth_pred = scale * depth_pred
    # depth_pred *= gt_median / (pred_median+1e-9)
    return float((((depth_pred - depth_gt) ** 2) * mask_gt).sum() / mask_gt.sum())


def calc_normal_distance(normal_pred, normal_gt, mask_gt):
    """
    Calculate the normal difference

    params:
        normal_pred: numpy.ndarray of shape [H, W, 3].
        normal_gt: numpy.ndarray of shape [H, W, 3].
        mask_gt: numpy.ndarray of shape [H, W]. ground truth foreground mask.
    """
    assert normal_pred.shape == normal_gt.shape, (normal_pred.shape, normal_gt.shape)
    mask_gt = erode_mask(mask_gt, normal_pred.shape[0])
    cos_dist = (1 - (normal_pred * normal_gt).sum(axis=-1)) * mask_gt
    return float(cos_dist.sum() / mask_gt.sum())


def run_eval(test_output_json: Path, gt_json: Path):
    """
    test_output_json: the output json file path
    gt_json: the ground truth json file path
    """
    with open(test_output_json, "r") as f:
        test_output = json.load(f)

    with open(gt_json, "r") as f:
        gt = json.load(f)["frames"]

    if "depth_path" not in gt[0].keys() or "normal_path" not in gt[0].keys():
        return None

    input_folder_for_gt = gt_json.parent

    # find the corresponding gt pair between the test output and the ground truth
    gt_image_dict = {}
    gt_depth_dict = {}
    gt_normal_dict = {}
    gt_mask_dict = {}
    for frame in gt:
        image_name = frame["image_path"].split("/")[-1]

        gt_image_dict[image_name] = input_folder_for_gt / frame["image_path"]
        gt_depth_dict[image_name] = input_folder_for_gt / frame["depth_path"]
        gt_normal_dict[image_name] = input_folder_for_gt / frame["normal_path"]
        gt_mask_dict[image_name] = input_folder_for_gt / frame["mask_path"]

    # evaluate depth
    depth_eval_dict = {}
    depth_eval_sum = 0
    for idx, (key, test_depth_path) in tqdm(enumerate(test_output["depth"].items())):

        image_name = key.split("/")[-1]
        gt_depth_path = gt_depth_dict[image_name]
        gt_mask_path = gt_mask_dict[image_name]

        gt_depth = np.array(Image.open(gt_depth_path)) / 255.0
        gt_mask = np.array(Image.open(gt_mask_path)) / 255.0
        test_depth = np.array(Image.open(test_depth_path)) / 255.0

        depth_distance = calc_depth_distance_scale_invariant(
            depth_pred=test_depth, depth_gt=gt_depth, mask_gt=gt_mask
        )

        depth_eval_dict[image_name] = depth_distance
        depth_eval_sum += depth_distance

    depth_eval_average = depth_eval_sum / len(depth_eval_dict)
    print(f"The average scale-invarant depth loss: {depth_eval_average}")

    depth_eval_stats = {
        "average": depth_eval_average,
        "frames": depth_eval_dict,
    }

    # calculate the average

    # evaluate normal
    normal_eval_dict = {}
    normal_eval_sum = 0
    for idx, (key, test_normal_path) in tqdm(enumerate(test_output["normal"].items())):

        image_name = key.split("/")[-1]
        gt_normal_path = gt_normal_dict[image_name]
        gt_mask_path = gt_mask_dict[image_name]

        gt_normal = (np.array(Image.open(gt_normal_path)) / 255.0 - 0.5) * 2
        gt_mask = np.array(Image.open(gt_mask_path)) / 255.0
        test_normal = (np.array(Image.open(test_normal_path)) / 255.0 - 0.5) * 2

        normal_distance = calc_normal_distance(
            normal_pred=test_normal, normal_gt=gt_normal, mask_gt=gt_mask
        )

        normal_eval_dict[image_name] = normal_distance
        normal_eval_sum += normal_distance

    normal_eval_average = normal_eval_sum / len(normal_eval_dict)
    print(f"The average normal cosine distance: {normal_eval_average}")

    normal_eval_stats = {
        "average": normal_eval_average,
        "frames": normal_eval_dict,
    }

    test_output["depth_scale_invariant_eval"] = depth_eval_stats
    test_output["normal_cosine_l1_distance_eval"] = normal_eval_stats

    return test_output
