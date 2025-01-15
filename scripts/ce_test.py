import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_poe_3(batch_size, height, width, rgb, channel,d_model=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # rgb = torch.tensor(rgb * 64, dtype=torch.float32).to(device)
    # rgb_255 = torch.tensor((rgb * 255).byte(), dtype=torch.int32).to(device)

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
                    torch.sin(rgb[b, channel, :, :] / div_term)
                )
            else:
                pe[b, i] = (
                    torch.cos(rgb[b, channel, :, :] / div_term)
                )
    
    # p1 = pe[:, 0:16].mean(dim=1, keepdim=True)  # Mean across channels 0-15
    # p2 = pe[:, 16:32].mean(dim=1, keepdim=True)  # Mean across channels 16-31
    # p3 = pe[:, 32:48].mean(dim=1, keepdim=True)  # Mean across channels 32-47
    # p4 = pe[:, 48:64].mean(dim=1, keepdim=True)   # Mean across channels 48-63

    return pe


# Create a tensor with shape [1, 1, 16, 16]
tensor = torch.zeros(1, 1, 16, 16)

# Fill the tensor with values ranging from 0 to 255
for i in range(16):
    for j in range(16):
        tensor[0, 0, i, j] = (i * 16 + j) * (255 / (16 * 16 - 1))


tensor = tensor.to(device=device)

# Print the tensor
print(tensor)

ce = get_poe_3(1, 16, 16, tensor, 0, 7)
ce = torch.mean(ce, dim=1, keepdim=True)
print(ce)
