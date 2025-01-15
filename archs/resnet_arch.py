import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as F


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
    

    # p1 = pe[:, 0:16].mean(dim=1)
    # p2 = pe[:, 16:32].mean(dim=1)
    # p3 = pe[:, 32:48].mean(dim=1)
    # p4 = pe[:, 48:64].mean(dim=1)
    # p1 = pe[:, 0:16].mean(dim=1, keepdim=True)  # Mean across channels 0-15
    # p2 = pe[:, 16:32].mean(dim=1, keepdim=True)  # Mean across channels 16-31
    # p3 = pe[:, 32:48].mean(dim=1, keepdim=True)  # Mean across channels 32-47
    # p4 = pe[:, 48:64].mean(dim=1, keepdim=True)   # Mean across channels 48-63

    return pe


@ARCH_REGISTRY.register()
class ResNet50Enhancement(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, cat_poe=False, pretrained=False):
        """
        输入：
        - in_channels: 输入图像通道数，默认 3 RGB 图像
        - out_channels: 输出图像通道数，默认 3。
        - pretrained: 是否加载预训练权重，默认 False。
        """
        super(ResNet50Enhancement, self).__init__()

        # 加载 ResNet50 主干，不加载预训练权重
        self.resnet50 = resnet50(pretrained=pretrained)

        self.poe = cat_poe

        if self.poe:
            in_channels = in_channels + 3

        # 修改输入层以适应任意通道数
        self.resnet50.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 移除全连接层 (fc) 和全局池化层 (avgpool)
        self.resnet50.fc = nn.Identity()
        self.resnet50.avgpool = nn.Identity()

        # 上采样部分：用于将特征图恢复到输入图像的大小
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),  # 上采样1
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 上采样2
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 上采样3
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 上采样4
            nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),   # 恢复输出通道数
            nn.Sigmoid()  # 将输出限制在 [0, 1] 范围内
        )

    def forward(self, x):
        """
        网络前向传播，输入 x 为 4D Tensor，形状为 [batch_size, channels, height, width]。
        """
        original_size = x.shape[2:]  # 保存输入图像的原始尺寸
        
        


        if self.poe:
            bsize, _, h, w = x.shape
            color_md_1 = get_color_mul_dim_for_resnet(bsize, h, w, x, 0)
            color_md_2 = get_color_mul_dim_for_resnet(bsize, h, w, x, 1)
            color_md_3 = get_color_mul_dim_for_resnet(bsize, h, w, x, 2)

            color_md_1 = torch.mean(color_md_1, dim=1, keepdim=True)
            color_md_2 = torch.mean(color_md_2, dim=1, keepdim=True)
            color_md_3 = torch.mean(color_md_3, dim=1, keepdim=True)

            color_md_new = torch.cat([color_md_1, color_md_2, color_md_3], dim=1)
            x = torch.cat([x, color_md_new], dim=1)


        global ccq_i
        if ccq_i:
            print(torch.mean(x))
            print("x. shape: ", x.shape)
            ccq_i = False



        # ResNet50 特征提取
        features = self.resnet50.conv1(x)
        features = self.resnet50.bn1(features)
        features = self.resnet50.relu(features)
        features = self.resnet50.maxpool(features)

        features = self.resnet50.layer1(features)
        features = self.resnet50.layer2(features)
        features = self.resnet50.layer3(features)
        features = self.resnet50.layer4(features)  # 输出 [batch_size, 2048, H/32, W/32]

        # 上采样恢复到原始尺寸
        output = self.upsample(features)  # 恢复部分分辨率
        output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)  # 调整为原始尺寸

        return output
    
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import ResidualBlockNoBN, default_init_weights, make_layer


# @ARCH_REGISTRY.register()
class MSRResNet(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out
