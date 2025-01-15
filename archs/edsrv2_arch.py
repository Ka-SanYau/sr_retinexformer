import torch
from torch import nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY



global ccq_i
ccq_i = True
def get_color_mul_dim_for_resnet(batch_size, height, width, rgb, channel,d_model=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # rgb = torch.tensor(rgb * 64, dtype=torch.float32).to(device)
    rgb_255 = torch.tensor((rgb * 255).byte(), dtype=torch.int32).to(device)

    # 创建位置编码张量
    pe = torch.zeros((batch_size, d_model, height, width), device=device)
    
    # 创建位置坐标
    # x = torch.linspace(1, height, steps=height, device=device).view(-1, 1).repeat(1, width)
    # y = torch.linspace(1, width, steps=width, device=device).view(1, -1).repeat(height, 1)

    for b in range(batch_size):
        for i in range(d_model):
            div_term = 10000 ** (i / d_model)
            if i % 2 == 0:
                pe[b, i] = (
                    # torch.sin(x / div_term) +
                    # torch.sin(y / div_term) +
                    (1/2) * torch.sin(rgb_255[b, channel, :, :] / div_term) + 0.5

                    # + torch.sin(rgb[b, 1, :, :] / div_term) +
                    # torch.sin(rgb[b, 2, :, :] / div_term)
                )
            else:
                pe[b, i] = (
                    # torch.cos(x / div_term) +
                    # torch.cos(y / div_term) +
                    (1/2) * torch.cos(rgb_255[b, channel, :, :] / div_term) + 0.5
                    # + torch.cos(rgb[b, 1, :, :] / div_term) +
                    # torch.cos(rgb[b, 2, :, :] / div_term)
                )
    
    return pe


def get_poe_enhanced(batch_size, height, width, rgb, channel,d_model=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # rgb = torch.tensor(rgb * 64, dtype=torch.float32).to(device)
    rgb_255 = torch.tensor((rgb * 255).byte(), dtype=torch.int32).to(device)

    # 创建位置编码张量
    pe = torch.zeros((batch_size, d_model, height, width), device=device)
    
    # 创建位置坐标
    # x = torch.linspace(1, height, steps=height, device=device).view(-1, 1).repeat(1, width)
    # y = torch.linspace(1, width, steps=width, device=device).view(1, -1).repeat(height, 1)

    for b in range(batch_size):
        for i in range(d_model):
            div_term = 10000 ** (i / d_model)
            if i % 2 == 0:
                pe[b, i] = (
                    (1/2) * torch.sin(rgb_255[b, channel, :, :] / div_term) + 0.5
                )
            else:
                pe[b, i] = (
                    (1/2) * torch.cos(rgb_255[b, channel, :, :] / div_term) + 0.5
                )
    
    p1 = pe[:, 0:16].mean(dim=1, keepdim=True)  # Mean across channels 0-15
    p2 = pe[:, 16:32].mean(dim=1, keepdim=True)  # Mean across channels 16-31
    p3 = pe[:, 32:48].mean(dim=1, keepdim=True)  # Mean across channels 32-47
    p4 = pe[:, 48:64].mean(dim=1, keepdim=True)   # Mean across channels 48-63

    return p1, p2, p3, p4



def get_poe_3(batch_size, height, width, rgb, channel,d_model=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # rgb = torch.tensor(rgb * 64, dtype=torch.float32).to(device)
    rgb_255 = torch.tensor((rgb * 255).byte(), dtype=torch.int32).to(device)

    # 创建位置编码张量
    pe = torch.zeros((batch_size, d_model, height, width), device=device)
    
    # 创建位置坐标
    # x = torch.linspace(1, height, steps=height, device=device).view(-1, 1).repeat(1, width)
    # y = torch.linspace(1, width, steps=width, device=device).view(1, -1).repeat(height, 1)

    for b in range(batch_size):
        for i in range(d_model):
            div_term = 10000 ** (i / d_model)
            if i % 2 == 0:
                pe[b, i] = (
                    (1/2) * torch.sin(rgb_255[b, channel, :, :] / div_term) + 0.5
                )
            else:
                pe[b, i] = (
                    (1/2) * torch.cos(rgb_255[b, channel, :, :] / div_term) + 0.5
                )
    
    # p1 = pe[:, 0:16].mean(dim=1, keepdim=True)  # Mean across channels 0-15
    # p2 = pe[:, 16:32].mean(dim=1, keepdim=True)  # Mean across channels 16-31
    # p3 = pe[:, 32:48].mean(dim=1, keepdim=True)  # Mean across channels 32-47
    # p4 = pe[:, 48:64].mean(dim=1, keepdim=True)   # Mean across channels 48-63

    return pe




@ARCH_REGISTRY.register()
class EDSR_v2(nn.Module):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 poemap=False,
                 poe_enhance=False,
                 poe_3=False,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EDSR_v2, self).__init__()

        self.img_range = img_range

        self.poe = poemap
        self.poe_enhance = poe_enhance
        self.poe_3 = poe_3

        if self.poe:
            num_in_ch = num_in_ch + 3


        if self.poe_enhance:
            num_in_ch = num_in_ch + 12

        if self.poe_3:
            num_in_ch = num_in_ch + 12
        

        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv_first_poe = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):


        # self.mean = self.mean.type_as(x)


        if self.poe:
            bsize, _, h, w = x.shape
            color_md_1 = get_color_mul_dim_for_resnet(bsize, h, w, x, 0, 5)
            color_md_2 = get_color_mul_dim_for_resnet(bsize, h, w, x, 1, 5)
            color_md_3 = get_color_mul_dim_for_resnet(bsize, h, w, x, 2, 5)

            color_md_1 = torch.mean(color_md_1, dim=1, keepdim=True)
            color_md_2 = torch.mean(color_md_2, dim=1, keepdim=True)
            color_md_3 = torch.mean(color_md_3, dim=1, keepdim=True)

            color_md_new = torch.cat([color_md_1, color_md_2, color_md_3], dim=1)
            # color_md_new = color_md_new * self.img_range
            # x = torch.cat([x, color_md_new], dim=1)

        # if self.poe:
        #     bsize, _, h, w = x.shape
        #     color_md_1 = get_color_mul_dim_for_resnet(bsize, h, w, x, 0, 32)
        #     color_md_2 = get_color_mul_dim_for_resnet(bsize, h, w, x, 1, 32)
        #     color_md_3 = get_color_mul_dim_for_resnet(bsize, h, w, x, 2, 32)

        #     # color_md_1 = torch.mean(color_md_1, dim=1, keepdim=True)
        #     # color_md_2 = torch.mean(color_md_2, dim=1, keepdim=True)
        #     # color_md_3 = torch.mean(color_md_3, dim=1, keepdim=True)

        #     color_md_new = torch.cat([color_md_1, color_md_2, color_md_3], dim=1)
            # color_md_new = (color_md_1 + color_md_2 + color_md_3) / 3
            # color_md_new = color_md_new * self.img_range
            # x = torch.cat([x, color_md_new], dim=1)


        if self.poe_enhance:
            bsize, _, h, w = x.shape
            color_r_p1, color_r_p2, color_r_p3, color_r_p4 = get_poe_enhanced(bsize, h, w, x, 0)
            color_g_p1, color_g_p2, color_g_p3, color_g_p4 = get_poe_enhanced(bsize, h, w, x, 1)
            color_b_p1, color_b_p2, color_b_p3, color_b_p4 = get_poe_enhanced(bsize, h, w, x, 2)

            color_md_new = torch.cat([color_r_p1, color_g_p1, color_b_p1, color_r_p2, color_g_p2, color_b_p2, color_r_p3, color_g_p3 , color_b_p3, color_r_p4, color_g_p4, color_b_p4], dim=1)
            # color_md_new = color_md_new * self.img_range
            # x = torch.cat([x, color_md_new], dim=1)


        if self.poe_3:
            bsize, _, h, w = x.shape
            r = get_poe_3(bsize, h, w, x, 0)
            g = get_poe_3(bsize, h, w, x, 1)
            b = get_poe_3(bsize, h, w, x, 2)

            color_md_new = torch.cat([r, g, b], dim=1)
            # color_md_new = color_md_new * self.img_range

        # x = (x - self.mean) * self.img_range

        if self.poe:
            x = torch.cat([x, color_md_new], dim=1)

        if self.poe_enhance:
            x = torch.cat([x, color_md_new], dim=1)

        if self.poe_3:
            x = torch.cat([x, color_md_new], dim=1)    

        global ccq_i

        if ccq_i:
            print(torch.mean(x))
            print("x. shape: ", x.shape)
            ccq_i = False

        x = self.conv_first(x)
        # x2 = self.conv_first_poe(color_md_new)
        # if self.poe:
        #     x = color_md_new
        # x= torch.cat([x, x2], dim=1)

        res = self.conv_after_body(self.body(x))
        # if self.poe:
        #     import math
        #     x = self.conv_first(x)
        #     res_ce = self.conv_after_body(self.body(color_md_new))
        #     res = x + res + (res_ce / math.sqrt(32))

        res += x
        x = self.conv_last(self.upsample(res))
        # x = x / self.img_range + self.mean


        return x
