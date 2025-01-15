import torch
import torch.nn as nn
# num_feat = 64
# bias = False

# class OverlapPatchEmbed(nn.Module):
#     def __init__(self, in_c=3, embed_dim=48, bias=False):
#         super(OverlapPatchEmbed, self).__init__()

#         self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

#     def forward(self, x):
#         x = self.proj(x)

#         return x

# df = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=bias)

# abc = [OverlapPatchEmbed() for i in range(4)]
# abc_new = nn.Sequential(*abc)
# # print(abc_new)
# for i in range(4):
#     print(*abc)

from einops import rearrange
import numbers


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

# 创建一个形状为 [1, 3, 2, 2] 的 tensor，初始化为 0
tensor = torch.zeros(1, 2, 2, 2)

# 设置最后一个 2x2 的矩阵为 [1, 2, 3, 4]
# 对应索引 [0, 0]（第一个通道）
tensor[0, 0] = torch.tensor([[1, -1], [1, 0]])
tensor[0, 1] = torch.tensor([[-1, 0], [0, 0]])
# tensor[0, 2] = torch.tensor([[9, 10], [11, 12]])


bias_free_norm = WithBias_LayerNorm(2)
ten1= to_3d(tensor)
ten2 = bias_free_norm(ten1)


print(tensor)
print(ten1)
print(ten2)
ten3 = to_4d(ten2, h=2, w=2)
print(ten3)
