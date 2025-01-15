## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from .pe import *
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import ResidualBlockNoBN, default_init_weights, make_layer


global llf
llf = True

global llf2
llf2 = True

global ccq_i
ccq_i = True


def get_poe_new(batch_size, height, width, rgb, channel,d_model=32):
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
                    torch.sin(rgb_255[b, channel, :, :] / div_term)
                )
            else:
                pe[b, i] = (
                    torch.cos(rgb_255[b, channel, :, :] / div_term)
                )
    
    # p1 = pe[:, 0:16].mean(dim=1, keepdim=True)  # Mean across channels 0-15
    # p2 = pe[:, 16:32].mean(dim=1, keepdim=True)  # Mean across channels 16-31
    # p3 = pe[:, 32:48].mean(dim=1, keepdim=True)  # Mean across channels 32-47
    # p4 = pe[:, 48:64].mean(dim=1, keepdim=True)   # Mean across channels 48-63

    return pe


def get_poe_new2(batch_size, height, width, rgb, channel,d_model=32):
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
                    (1/2)*torch.sin(rgb_255[b, channel, :, :] / div_term) + 0.5
                )
            else:
                pe[b, i] = (
                    (1/2)*torch.cos(rgb_255[b, channel, :, :] / div_term) + 0.5
                )
    
    # p1 = pe[:, 0:16].mean(dim=1, keepdim=True)  # Mean across channels 0-15
    # p2 = pe[:, 16:32].mean(dim=1, keepdim=True)  # Mean across channels 16-31
    # p3 = pe[:, 32:48].mean(dim=1, keepdim=True)  # Mean across channels 32-47
    # p4 = pe[:, 48:64].mean(dim=1, keepdim=True)   # Mean across channels 48-63

    return pe



class DynamicColorAttention(nn.Module):
    def __init__(self, feature_dim, color_dim, reduction=2):
        """
        Dynamic Attention module for Color Encoding.
        
        Args:
            feature_dim (int): Number of channels in the image features (F).
            color_dim (int): Number of channels in the Color Encoding (CE).
            reduction (int): Reduction ratio for the attention mechanism.
        """
        super(DynamicColorAttention, self).__init__()
        
        # Reduce dimensionality to a bottleneck space for efficient attention computation
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(feature_dim + color_dim, feature_dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Generate attention weights for the Color Encoding
        self.attention = nn.Conv2d(feature_dim // reduction, color_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, color_encoding):
        """
        Args:
            features (Tensor): Input image features of shape (B, C, H, W).
            color_encoding (Tensor): Color Encoding of shape (B, C_ce, H, W).
        
        Returns:
            Tensor: Fused features of shape (B, C, H, W).
        """
        # Concatenate features and color_encoding along the channel dimension
        x = torch.cat([features, color_encoding], dim=1)  # Shape: (B, C + C_ce, H, W)
        
        # Reduce to bottleneck space
        x = self.reduce_channels(x)  # Shape: (B, C // reduction, H, W)
        
        # Compute attention weights for the Color Encoding
        attention_weights = self.attention(x)  # Shape: (B, C_ce, H, W)
        attention_weights = self.sigmoid(attention_weights)  # Normalize to [0, 1]
        
        # Apply attention weights to Color Encoding
        color_encoding = color_encoding * attention_weights  # Shape: (B, C_ce, H, W)
        
        # Fuse features and color_encoding (use addition or concatenation)
        fused_features = features + color_encoding  # Shape: (B, C, H, W)
        
        return fused_features

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)



    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # x = self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        # x = self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
@ARCH_REGISTRY.register()
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_feat = 64,
        num_res_block = 1,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,       ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        concat_poe_map = False
    ):
        super(Restormer, self).__init__()
        self.concat_poe_map = concat_poe_map

        self.patch_embed = OverlapPatchEmbed(inp_channels, 48)

        self.color_split_feature = nn.Conv2d(1, 16, 3, 1, 1, bias=bias)


        # cat ce features ------------
        # self.dw_conv_first = nn.Conv2d(48, 48, 3, 1, 1, groups=48, bias=bias)
        self.conv_first = nn.Conv2d(48, 48, 3, 1, 1, bias=bias)
        self.dynamic_ce1 = DynamicColorAttention(feature_dim=48, color_dim=48, reduction=2)
        self.dynamic_ce2 = DynamicColorAttention(feature_dim=48*2, color_dim=48*2, reduction=2)
        self.dynamic_ce3 = DynamicColorAttention(feature_dim=48*2*2, color_dim=48*2*2, reduction=2)
        # self.conv_first = nn.Conv2d(inp_channels, num_feat, 3, 1, 1, bias=bias)
        # self.conv_ce = nn.Conv2d(96, num_feat, 3, 1, 1, bias=bias)

        self.conv_first2 = nn.Conv2d(inp_channels, dim, 3, 1, 1, bias=bias)
        self.conv_first3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)

        # self.body = make_layer(ResidualBlockNoBN, num_res_block, num_feat=num_feat)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=bias)
        self.conv_last = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias)
        self.conv_ce_down_feat = nn.Conv2d(dim+192, dim, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Cat ce features end ---------


        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        # self.down3_4 = Downsample(int(dim*2*)) ## From Level 3 to Level 4
        # self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.output = nn.Conv2d(int(dim*2**1), 48, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        global ccq_i

        x = inp_img

        feat = self.patch_embed(x)

        if self.concat_poe_map:
            bsize, _, h, w = x.shape
            # feat = self.patch_embed(x)

            tmp = x[:,:3]
            # color_md_1 = get_poe_new(bsize, h, w, tmp, 0, 32)
            # color_md_2 = get_poe_new(bsize, h, w, tmp, 1, 32)
            # color_md_3 = get_poe_new(bsize, h, w, tmp, 2, 32)
            color_md_1 = get_poe_new2(bsize, h, w, tmp, 0, 16)
            color_md_2 = get_poe_new2(bsize, h, w, tmp, 1, 16)
            color_md_3 = get_poe_new2(bsize, h, w, tmp, 2, 16)

            # ce_mean1 = torch.mean(color_md_1, dim=1, keepdim=True)
            # ce_mean2 = torch.mean(color_md_2, dim=1, keepdim=True)
            # ce_mean3 = torch.mean(color_md_3, dim=1, keepdim=True)
            
            # ce_mean = torch.cat([ce_mean1, ce_mean2, ce_mean3], dim=1)
            # ce_mean_fea = self.patch_embed(ce_mean)

            color_md_new = torch.cat([color_md_1, color_md_2, color_md_3], dim=1)
            
            # ce_mean_fea = ce_mean_fea + color_md_new

            feat2 = self.conv_first(color_md_new)

            # color_md_new = color_md_1 + color_md_2 + color_md_3



            # color_md_new = torch.cat([color_md_1, color_md_2, color_md_3], dim=1)
            # tmp = torch.cat([feat, self.conv_ce(color_md_new)], dim=1)
            # feat = torch.cat([feat, color_md_1], dim=1)
            # feat = self.conv_first3(color_md_new)
            if ccq_i:
                print("self.conv_first(x) add color_md_new.")
        
            # feat = self.conv_ce_down_feat(feat)
            # feat = feat + color_md_new
            # feat = self.lrelu(self.conv_ce_down_feat(feat))
            # feat = self.dynamic_ce1(feat, color_md_new)
            # feat = torch.cat([r_feature, g_feature, b_feature], dim=1)
            # feat = self.conv_ce_down_feat(feat)
            # feat = self.conv_ce_down_feat(torch.cat([feat, feat2], dim=1))
            feat = self.lrelu(torch.cat([feat, feat2], dim=1))
        else:
            feat = self.patch_embed(x)

        inp_enc_level1 = x

        if self.concat_poe_map:
            
            if ccq_i:
                # print(inp_enc_level1.shape)
                print("cat ce...")
                print(inp_img.shape)
                ccq_i = False

        out_enc_level1 = self.encoder_level1(feat)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)

        # dynamix ce to it
        # if self.concat_poe_map:
        #     color_md_new = self.down1_2(color_md_new)
        #     inp_enc_level2 = self.dynamic_ce2(inp_enc_level2, color_md_new)


        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        # dynamix ce to it
        # if self.concat_poe_map:
        #     color_md_new = self.down2_3(color_md_new)
        #     inp_enc_level3 = self.dynamic_ce3(inp_enc_level3, color_md_new)

        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        # inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level3) 
        # latent = self.latent(inp_enc_level4) 

        # inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = latent
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1)
            # out_dec_level1 = F.pixel_shuffle(out_dec_level1, 4)
            out_dec_level1 = out_dec_level1 + inp_img

        return out_dec_level1

