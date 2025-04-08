from enum import Enum
from math import exp
from typing import List, Optional

import einops
import torch

from fused_ssim import fused_ssim
from torch.autograd import Variable
from torch.nn.functional import conv2d, huber_loss, l1_loss, mse_loss
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from utils.image_utils import psnr, rgb_to_ycbcr


def calculcate_inverse_depth_loss(
    render_depth: torch.Tensor,
    sparse_point2d: torch.Tensor,
    sparse_inv_depth: torch.Tensor,
    sparse_inv_distance_std: Optional[torch.Tensor] = None,
    losses: List[str] = ["huber"],
    huber_delta: float = 0.5,
):
    """
    Calculate the inverse depth loss from a rendered depth

    render_depth: (B, H, W)
    sparse_point2d: the pixel coordinate of a sparse point cloud in 2d. (B, N, 2), where N is the number of sparse points.
    sparse_inv_depth: the inverse depth value correspond to sparse_point2d. (B, N)
    sparse_inv_distance_std: standard deviation of the inverse distance estimate for each 2d point.
    """
    device = render_depth.device

    H, W = render_depth.shape[1:]
    sparse_points_norm = torch.stack(
        [
            sparse_point2d[..., 0] / (W - 1) * 2 - 1,
            sparse_point2d[..., 1] / (H - 1) * 2 - 1,
        ],
        dim=-1,
    )  # (B,N,2)
    grid = einops.rearrange(sparse_points_norm, "b n c -> b n 1 c").to(device)
    point_depths = torch.nn.functional.grid_sample(
        render_depth[:, None], grid, align_corners=True
    )  # (b, 1, N, 1)
    point_depths = einops.rearrange(point_depths, "b 1 n 1 -> b n")

    render_inv_depth = torch.where(
        point_depths > 1e-1, 1.0 / point_depths, torch.zeros_like(point_depths)
    )

    final_loss = {}
    if "l1" in losses:
        l1_error = l1_loss(render_inv_depth, sparse_inv_depth, reduce=False)
        if sparse_inv_distance_std is not None:
            final_loss["l1"] = (l1_error / sparse_inv_distance_std).mean()
        else:
            final_loss["l1"] = l1_error.mean()

    if "huber" in losses:
        huber_error = huber_loss(
            render_inv_depth, sparse_inv_depth, delta=huber_delta, reduction="none"
        )
        if sparse_inv_distance_std is not None:
            final_loss["huber"] = (huber_error / sparse_inv_distance_std).mean()
        else:
            final_loss["huber"] = huber_error.mean()

    if "l2" in losses:
        mse_error = mse_loss(render_inv_depth, sparse_inv_depth, reduction="none")
        if sparse_inv_distance_std is not None:
            final_loss["l2"] = (mse_error / (sparse_inv_distance_std**2)).mean()
        else:
            final_loss["l2"] = mse_error.mean()

    return final_loss


class ColorSpace(Enum):

    RGB = 0
    YCB = 1
    LUMINANCE = 2


class ImageLoss:

    def __init__(
        self,
        color_space: str = "rgb",  # "ycb"
        lpips_net_type: str = "alex",
        apply_grad_loss: bool = True,
        device: str = "cuda",
    ):

        if color_space == "rgb":
            print("Calculate losses using RGB color space")
            self.color_space = ColorSpace.RGB
        elif color_space == "ycb":
            print("Calculate losses using YUV color space")
            self.color_space = ColorSpace.YCB
        elif color_space == "luminance":
            print("Calculate losses using only luminance space")
            self.color_space = ColorSpace.LUMINANCE
        else:
            raise NotImplementedError(f"cannot recognize color space {color_space}")

        if lpips_net_type == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(device)
        elif lpips_net_type == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(device)
        else:
            raise ValueError(f"Unknown LPIPS network: {lpips_net_type}")

        if apply_grad_loss:
            self.grad_x = torch.tensor(
                [
                    [-1, -2, 0, 2, 1],
                    [-2, -4, 0, 4, 2],
                    [-3, -6, 0, 6, 3],
                    [-2, -4, 0, 4, 2],
                    [-1, -2, 0, 2, 1],
                ],
                dtype=torch.float32,
            ).to(device)[None, None]
            self.grad_y = torch.tensor(
                [
                    [-1, -2, -3, -2, -1],
                    [-2, -4, -6, -4, -2],
                    [0, 0, 0, 0, 0],
                    [2, 4, 6, 4, 2],
                    [1, 2, 3, 2, 1],
                ],
                dtype=torch.float32,
            ).to(device)[None, None]

    def _rgb2color(self, rgb):
        """
        convert the input image space to the target space
        """
        if self.color_space == ColorSpace.RGB:
            return rgb
        elif self.color_space == ColorSpace.YCB:
            return rgb_to_ycbcr(rgb)

    def _calculate_grad_magnitude(self, image):
        grad_x = torch.nn.functional.conv2d(image, self.grad_x, padding=2)
        grad_y = torch.nn.functional.conv2d(image, self.grad_y, padding=2)
        return torch.cat([grad_x, grad_y], dim=1)

    def __call__(
        self,
        capture_image: torch.Tensor,
        gt_image: torch.Tensor,
        losses: List[str],
        valid_mask: Optional[torch.Tensor] = None,
        read_noise_factor: float = 1.0,
        shot_noise_factor: float = 0,
    ):

        B, C, H, W = capture_image.shape

        if valid_mask is None:
            total_pixels = B * C * H * W
        else:
            total_pixels = valid_mask.sum() * C  # mask is (HxW)

        final_loss = {}
        capture_image_ycb, gt_image_ycb = None, None
        if "l1" in losses:

            if self.color_space == ColorSpace.RGB:
                # this is the vanilla L1 loss. Taking average of all samples equally.
                L_l1 = l1_loss(capture_image, gt_image)

            elif self.color_space == ColorSpace.YCB:
                capture_image_ycb = rgb_to_ycbcr(capture_image)
                gt_image_ycb = rgb_to_ycbcr(gt_image)
                capture_luminance = capture_image_ycb[:, 0:1]
                gt_luminance = gt_image_ycb[:, 0:1]

                l1_pixels = l1_loss(capture_image_ycb, gt_image_ycb, reduction="none")

                if valid_mask is not None:
                    pixel_mask = (
                        valid_mask.clone().to(l1_pixels).expand(l1_pixels.shape)
                    )
                    l1_pixels = pixel_mask * l1_pixels

                # for YCB space, we give more weights to the losses in Y space.
                # We assigned weight 422 to different channel
                total_loss = (
                    l1_pixels[:, 0] * 4 + l1_pixels[:, 1] * 2 + l1_pixels[:, 2] * 2
                )
                L_l1 = total_loss.sum() / total_pixels
            elif self.color_space == ColorSpace.LUMINANCE:
                capture_luminance = capture_image
                gt_luminance = gt_image

                l1_pixels = l1_loss(capture_luminance, gt_luminance, reduction="none")
                if valid_mask is not None:
                    pixel_mask = (
                        valid_mask.clone().to(l1_pixels).expand(l1_pixels.shape)
                    )
                    l1_pixels = pixel_mask * l1_pixels

                L_l1 = l1_pixels.sum() / total_pixels

            final_loss["l1"] = L_l1

        if "l1_grad" in losses:
            if capture_luminance is None:
                capture_image_ycb = rgb_to_ycbcr(capture_image)
                capture_luminance = capture_image_ycb[:, 0:1]
            if gt_luminance is None:
                gt_image_ycb = rgb_to_ycbcr(gt_image)
                gt_luminance = gt_image_ycb[:, 0:1]

            # calculate the gradient magnitude in the image space.
            capture_grad = self._calculate_grad_magnitude(capture_luminance)
            gt_grad = self._calculate_grad_magnitude(gt_luminance)

            l1_grad_pixels = l1_loss(capture_grad, gt_grad, reduction="none")

            if valid_mask is not None:
                pixel_mask = (
                    valid_mask.clone().to(l1_grad_pixels).expand(l1_grad_pixels.shape)
                )
                l1_grad_pixels *= pixel_mask

            L_l1_grad = l1_grad_pixels.sum() / total_pixels

            final_loss["l1_grad"] = L_l1_grad

        if "huber" in losses:
            L_huber = huber_loss(capture_image, gt_image, delta=0.4, reduction="none")

            noise_variance = read_noise_factor + capture_image * shot_noise_factor
            L_huber_noise_aware = L_huber / noise_variance
            L_huber = L_huber_noise_aware.mean()

            final_loss["huber"] = L_huber

        if "ssim" in losses:
            L_ssim = fused_ssim(capture_image, gt_image, padding="valid")
            final_loss["ssim"] = L_ssim
        else:
            L_ssim = None

        if "dssim" in losses:
            if L_ssim is not None:
                L_dssim = 1.0 - L_ssim
            else:
                L_dssim = 1.0 - fused_ssim(capture_image, gt_image, padding="valid")
                # L_dssim = 1.0 - ssim(capture_image, gt_image)
            final_loss["dssim"] = L_dssim

        if "psnr" in losses:
            with torch.no_grad():
                L_psnr = psnr(capture_image, gt_image, mask=valid_mask).mean()
            final_loss["psnr"] = L_psnr

        if "lpips" in losses:
            if self.color_space == ColorSpace.LUMINANCE:
                L_lpips = self.lpips(
                    capture_image.repeat(1, 3, 1, 1), gt_image.repeat(1, 3, 1, 1)
                )
            else:
                L_lpips = self.lpips(capture_image, gt_image)
            final_loss["lpips"] = L_lpips

        return final_loss


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(
        rho * torch.log(rho / (rho_hat + 1e-5))
        + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-5))
    )


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
