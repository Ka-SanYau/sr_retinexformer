# 测试 PyTorch 的 GPU 加速功能
import torch

# 创建一个随机张量
x = torch.rand(3, 3).cuda()  # 将张量移动到 GPU

# 打印张量的设备信息
print(x.device)

# 执行简单的张量运算
y = x * 2
print(y)
print(torch.version.cuda)

print(torch.backends.cudnn.enabled)