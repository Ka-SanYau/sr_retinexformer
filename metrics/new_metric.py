# import basicsr.metrics
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import math
import lpips

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import METRIC_REGISTRY


# from basicsr.utils.registry import METRIC_REGISTRY

lpips_model = lpips.LPIPS(net='alex')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_model = lpips_model.to(device)  # Move model to the appropriate device


@METRIC_REGISTRY.register()
def calculate_lpips_ccq(img1, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        img1 (ndarray/tensor): Image 1, with range [0, 1].
        img2 (ndarray/tensor): Image 2, with range [0, 1].
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: LPIPS similarity result.
    """
    
    # Ensure the input images have the same shape
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

    # print("img1. shape: ", img1.shape)
    # print("img2. shape: ", img2.shape)
    
    # Convert images to PyTorch tensors if they are numpy arrays
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0)
    
    if img1.dtype != torch.float32:
        img1 = img1.float()
    if img2.dtype != torch.float32:
        img2 = img2.float()

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    
    # Ensure the pixel values are in the range [0, 1]
    if img1.max() > 1.0 or img2.max() > 1.0:
        # print("img1.max(): ", img1.max())
        img1 = img1 / 255.0
        img2 = img2 / 255.0

    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1

    # Calculate LPIPS
    img1 = img1.to(device)
    img2 = img2.to(device)



    lpips_value = lpips_model.forward(img1, img2)
    
    return lpips_value.item()
