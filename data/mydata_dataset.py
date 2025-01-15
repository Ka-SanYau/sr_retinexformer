import math
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from .transforms import paired_random_crop
# .utils import paired_random_crop
import numpy as np
import cv2
import torch
import os.path as osp

from torchvision import transforms

# # 定义将图像缩小一半的变换
# resize_transform = transforms.Resize((original_height // 2, original_width // 2))

import torch.nn.functional as F

@DATASET_REGISTRY.register()
class PairedImageDataset_v2(data.Dataset):
    def __init__(self, opt):
        super(PairedImageDataset_v2, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # 缩放 lq 和 gt 图像，缩小到原来的一半
        # 注意：interpolate 要求输入为 4D 张量，所以需要先 unsqueeze，再缩放，最后 squeeze 恢复
        img_lq = F.interpolate(img_lq.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
        img_gt = F.interpolate(img_gt.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
    


def get_color_mul_dim_asdf(height, width, rgb, channel,d_model=64):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将RGB图像转换为张量并移动到GPU
    # rgb = torch.exp(rgb)

    # rgb = rgb.clone().detach() * 64
    # rgb = rgb.to(device)
    rgb = torch.clamp(rgb, min=0, max=1)
    rgb = rgb.clone().detach() * 64
    # rgb = rgb.to(device)
    # rgb = torch.tensor(rgb * 64, dtype=torch.float32).to(device)
    # torch.
    
    # 创建位置编码张量
    pe = torch.zeros((d_model, height, width))
    
    # 创建位置坐标
    # x = torch.linspace(1, height, steps=height, device=device).view(-1, 1).repeat(1, width)
    # y = torch.linspace(1, width, steps=width, device=device).view(1, -1).repeat(height, 1)

   
    for i in range(d_model):
        div_term = 10000 ** (i / d_model)
        if i % 2 == 0:
            pe[i] = (
                # torch.sin(x / div_term) +
                # torch.sin(y / div_term) +
                (1/2) * torch.sin(rgb[channel, :, :] / div_term) + 0.5

                # + torch.sin(rgb[b, 1, :, :] / div_term) +
                # torch.sin(rgb[b, 2, :, :] / div_term)
            )
        else:
            pe[i] = (
                # torch.cos(x / div_term) +
                # torch.cos(y / div_term) +
                (1/2) * torch.cos(rgb[channel, :, :] / div_term) + 0.5
                # + torch.cos(rgb[b, 1, :, :] / div_term) +
                # torch.cos(rgb[b, 2, :, :] / div_term)
            )
    
    return pe

def get_color_mul_dim_asdf_addlightmean(height, width, rgb, channel,d_model=64):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将RGB图像转换为张量并移动到GPU
    # rgb = torch.exp(rgb)

    # rgb = rgb.clone().detach() * 64
    # rgb = rgb.to(device)
    rgb = torch.clamp(rgb, min=0, max=1)
    rgb = rgb.clone().detach() * 64
    # rgb = rgb.to(device)
    # rgb = torch.tensor(rgb * 64, dtype=torch.float32).to(device)
    # torch.
    
    # 创建位置编码张量
    pe = torch.zeros((d_model, height, width))
    
    # 创建位置坐标
    # x = torch.linspace(1, height, steps=height, device=device).view(-1, 1).repeat(1, width)
    # y = torch.linspace(1, width, steps=width, device=device).view(1, -1).repeat(height, 1)

   
    for i in range(d_model):
        div_term = 10000 ** (i / d_model)
        if i % 2 == 0:
            pe[i] = (
                # torch.sin(x / div_term) +
                # torch.sin(y / div_term) +
                torch.sin(rgb[channel, :, :] / div_term)

                # + torch.sin(rgb[b, 1, :, :] / div_term) +
                # torch.sin(rgb[b, 2, :, :] / div_term)
            )
        else:
            pe[i] = (
                # torch.cos(x / div_term) +
                # torch.cos(y / div_term) +
                torch.cos(rgb[channel, :, :] / div_term)
                # + torch.cos(rgb[b, 1, :, :] / div_term) +
                # torch.cos(rgb[b, 2, :, :] / div_term)
            )
    

    pe += rgb[channel, :, :].mean()

    min_val = pe.min()
    max_val = pe.max()
    
    pe = (pe - min_val) / (max_val - min_val)

    pe = torch.clamp(pe, 0, 1)

    return pe

@DATASET_REGISTRY.register()
class PairedImageDataset_asdf(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset_asdf, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        h, w = img_lq.shape[1:]
        ce1 = get_color_mul_dim_asdf(h, w, img_lq, 0)
        ce2 = get_color_mul_dim_asdf(h, w, img_lq, 1)
        ce3 = get_color_mul_dim_asdf(h, w, img_lq, 2)
        # ce1 = get_color_mul_dim_asdf_addlightmean(h, w, img_lq, 0)
        # ce2 = get_color_mul_dim_asdf_addlightmean(h, w, img_lq, 1)
        # ce3 = get_color_mul_dim_asdf_addlightmean(h, w, img_lq, 2)

        # ce1 = ce1.mean(dim=0, keepdim=True)
        # ce2 = ce2.mean(dim=0, keepdim=True)
        # ce3 = ce3.mean(dim=0, keepdim=True)
        

        # pe = (pe - min_val) / (max_val - min_val)

        r_mean = img_lq[0,:,:].mean()
        g_mean = img_lq[1,:,:].mean()
        b_mean = img_lq[2,:,:].mean()

        # ce1 = ce1.mean(dim=0, keepdim=True) + r_mean
        # ce2 = ce2.mean(dim=0, keepdim=True) + g_mean
        # ce3 = ce3.mean(dim=0, keepdim=True) + b_mean

        ce = torch.cat([ce1.mean(dim=0, keepdim=True)+r_mean, ce2.mean(dim=0, keepdim=True) +g_mean, ce3.mean(dim=0, keepdim=True) +b_mean], dim=0)
        # ce = torch.cat([ce1, ce2, ce3.mean(dim=0, keepdim=True)]+b_mean, dim=0)
        

        img_lq = torch.cat([img_lq, ce], dim=0)

        # r_mean = img_lq[0,:,:].mean()
        # g_mean = img_lq[1,:,:].mean()
        # b_mean = img_lq[2,:,:].mean()

        # r_min = img_lq[0,:,:].min()
        # r_max = img_lq[0,:,:].max()
        # g_min = img_lq[1,:,:].min()
        # g_max = img_lq[1,:,:].max()
        # b_min = img_lq[2,:,:].min()
        # b_max = img_lq[2,:,:].max()

        # print(f"R channel - min: {r_min}, max: {r_max}")
        # print(f"G channel - min: {g_min}, max: {g_max}")
        # print(f"B channel - min: {b_min}, max: {b_max}")

        # print(f"ccq, r: {r_mean}, g: {g_mean}, b:{b_mean}")
        # ccq = input("please input a number: ")


        

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
