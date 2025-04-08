import einops
import numpy as np
import torch
from PIL import Image


def rgb_to_luminance(rgb):
    # ITU-R BT.601
    if rgb.dim() == 4:
        assert rgb.shape[1] == 3, "RGB input needs (B, C, H, W) shape"
        r = rgb[:, 0:1, ...]
        g = rgb[:, 1:2, ...]
        b = rgb[:, 2:3, ...]
    elif rgb.dim() == 3:
        assert rgb.shape[0] == 3, "RGB input needs (C, H, W) shape"
        r = rgb[0:1, ...]
        g = rgb[1:2, ...]
        b = rgb[2:3, ...]
    else:
        raise RuntimeError(f"Unknown rgb input tensor shape {rgb.shape}")

    return 0.299 * r + 0.587 * g + 0.114 * b


def rgb_to_ycbcr(rgb):
    # JPEG conversion: https://en.wikipedia.org/wiki/YCbCr
    if rgb.dim() == 4:
        assert rgb.shape[1] == 3, "RGB input needs (B, C, H, W) shape"
        r = rgb[:, 0, ...]
        g = rgb[:, 1, ...]
        b = rgb[:, 2, ...]
    elif rgb.dim() == 3:
        assert rgb.shape[0] == 3, "RGB input needs (B, C, H, W) shape"
        r = rgb[0, ...]
        g = rgb[1, ...]
        b = rgb[2, ...]
    else:
        raise RuntimeError(f"Unknown rgb input tensor shape {rgb.shape}")

    y = 0.299 * r + 0.587 * g + 0.114 * b
    offset = 0.5
    cr = -0.168736 * r - 0.331264 * g + 0.5 * b + offset
    cb = 0.5 * r - 0.418688 * g - 0.081312 * b + offset

    if rgb.dim() == 4:
        return torch.stack((y, cb, cr), 1)
    elif rgb.dim() == 3:
        return torch.stack((y, cb, cr), 0)


def ycbcr_to_rgb(ycbcr):
    # JPEG conversion: https://en.wikipedia.org/wiki/YCbCr
    if ycbcr.dim() == 4:
        assert ycbcr.shape[1] == 3, "YCbCr input needs (B, C, H, W) shape"
        y = ycbcr[:, 0, ...]
        cb = ycbcr[:, 1, ...]
        cr = ycbcr[:, 2, ...]
    elif ycbcr.dim() == 3:
        assert ycbcr.shape[0] == 3, "YCbCr input needs (C, H, W) shape"
        y = ycbcr[0, ...]
        cb = ycbcr[1, ...]
        cr = ycbcr[2, ...]
    else:
        raise RuntimeError(f"Unknown YCbCr input tensor shape {ycbcr.shape}")

    offset = 0.5
    r = y + 1.402 * (cr - offset)
    g = y - 0.344136 * (cb - offset) - 0.714136 * (cr - offset)
    b = y + 1.772 * (cb - offset)

    if ycbcr.dim() == 4:
        return torch.stack((r, g, b), 1)
    elif ycbcr.dim() == 3:
        return torch.stack((r, g, b), 0)


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[1], -1).mean(1, keepdim=True)


def psnr(img1, img2, mask=None):
    if mask is None:
        mse = (((img1 - img2)) ** 2).view(img1.shape[1], -1).mean(1, keepdim=True)
    else:
        mask = mask.to(img1).expand(img1.shape)
        img1 = img1 * mask
        img2 = img2 * mask
        mse = (((img1 - img2)) ** 2).sum() / mask.sum()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def convert_image_tensor2array(image: torch.Tensor) -> np.ndarray:
    assert image.dim() == 3
    assert (
        image.max() <= 1.0
    ), "image max cannot exceed 1.0. It will cause overflow issue in saved images."
    assert (
        image.min() >= 0
    ), "image min canno 0.0. It will cause overflow issue in saved images."

    if image.shape[0] == 1:
        return (
            einops.repeat(image, "1 h w -> h w 3").contiguous().cpu().numpy() * 255
        ).astype(np.uint8)
    elif image.shape[0] == 3:
        return (
            einops.rearrange(image, "c h w -> h w c").contiguous().cpu().numpy() * 255
        ).astype(np.uint8)
    elif image.shape[-1] == 1:
        return (
            einops.repeat(image, "h w 1 -> h w 3").contiguous().cpu().numpy() * 255
        ).astype(np.uint8)
    elif image.shape[-1] == 3:
        return (image.cpu().numpy() * 255).astype(np.uint8)
    else:
        raise RuntimeError(f"cannot convert image tensor of shape {image.shape}")


def save_image(image_array: np.ndarray, path: str):
    Image.fromarray(image_array).save(path)


def apply_monochrome_gamma(image: torch.Tensor, gamma: float = 2.0):
    """
    The Aria 10->8 bit image, with gamma reciprocal value 0.5.
    Details see:
    https://fb.workplace.com/groups/2734561303234150/permalink/4570688789621383

    Implementation at: https://fburl.com/code/ysz580ky
    """
    image_out = torch.zeros_like(image)
    mask_linear = image < 0.017946

    gamma_reciprocal = 1.0 / gamma
    image_out[mask_linear] = image[mask_linear]
    image_out[~mask_linear] = (
        1.07179 * torch.pow(image[~mask_linear], gamma_reciprocal) - 0.07179
    )
    return image_out


def linear_to_sRGB(image, gamma):
    """
    Convert linear values to sRGB values.
    """
    image_out = torch.zeros_like(image)
    mask_linear = image < 0.0031308
    gamma_reciprocal = 1.0 / gamma
    image_out[mask_linear] = 12.92 * image[mask_linear]
    image_out[~mask_linear] = (
        1.055 * torch.pow(image[~mask_linear], gamma_reciprocal) - 0.055
    )
    return image_out


def sRGB_to_linear(srgb, gamma):
    """
    Convert sRGB values to linear RGB values.

    Args:
    srgb (torch.Tensor): An sRGB image tensor with values in [0, 1].

    Returns:
    torch.Tensor: A linear RGB image tensor with values in [0, 1].
    """
    # Apply the inverse gamma correction
    # sRGB values less than 0.04045 are handled differently than those above
    mask = srgb <= 0.04045
    linear_below = srgb / 12.92
    linear_above = ((srgb + 0.055) / 1.055) ** gamma

    # Combine the results
    linear_rgb = torch.where(mask, linear_below, linear_above)

    return linear_rgb


def apply_aria_rgb_gamma(linear_image):
    """
    convert an Aria RGB image to  following the CRF from the test samples
    See details in: https://fb.workplace.com/groups/1581078722644621/permalink/1757373578348467/

    The RGB image in lower half is close to gamma 1.6 and in upper half close to gamma 2.2
    """
    mask = linear_image < 0.5
    srgb_lower = linear_to_sRGB(linear_image, gamma=1.6)
    srgb_upper = linear_to_sRGB(linear_image, gamma=2.2)

    aria_rgb = torch.where(mask, srgb_lower, srgb_upper)
    return aria_rgb
