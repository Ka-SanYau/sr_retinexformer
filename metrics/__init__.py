import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import scandir
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.metrics.niqe import calculate_niqe
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from .new_metric import calculate_lpips_ccq

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_lpips_ccq']

loss_folder = osp.dirname(osp.abspath(__file__))
# print("loss_folder", loss_folder)
loss_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(loss_folder) if v.endswith('_metric.py')]
# print("loss_filenames: ", loss_filenames)
_model_modules = [importlib.import_module(f'metrics.{file_name}') for file_name in loss_filenames]


# def calculate_metric(data, opt):
#     """Calculate metric from data and options.

#     Args:
#         opt (dict): Configuration. It must contain:
#             type (str): Model type.
#     """
#     opt = deepcopy(opt)
#     metric_type = opt.pop('type')
#     metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
#     return metric
