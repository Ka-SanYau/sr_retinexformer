import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# def get_pee(height, width, d_model):
#     """
#     生成 Sinusoidal Positional Encoding
#     """
#     pe = np.zeros((height, width, d_model))
#     for i in range(height):
#         for j in range(width):
#             for k in range(0, d_model, 2):
#                 pe[i, j, k] = np.sin(i / (10000 ** (k / d_model)))
#                 pe[i, j, k + 1] = np.cos(i / (10000 ** (k / d_model)))
#                 pe[i, j, k + 2] = np.sin(j / (10000 ** (k / d_model)))
#                 pe[i, j, k + 3] = np.cos(j / (10000 ** (k / d_model)))
#     return torch.tensor(pe, dtype=torch.float32)


# import numpy as np

# def get_pee2(height, width, channels, d_model):
#     pe = np.zeros((height, width, channels))
    
#     for i in range(height):
#         for j in range(width):
#             for k in range(channels):
#                 if k % 2 == 0:
#                     pe[i, j, k] = np.sin(i / (10000 ** (k / d_model)))
#                 else:
#                     pe[i, j, k] = np.cos(i / (10000 ** ((k - 1) / d_model)))
                
#                 # Ensure k + 3 is within bounds
#                 if k + 3 < channels:
#                     pe[i, j, k + 3] = np.cos(j / (10000 ** (k / d_model)))
#     pe_tensor = torch.tensor(pe, dtype=torch.float32).cuda()
#     return pe_tensor

# # # 假设输入图像为 img，形状为 (batch_size, channels, height, width)
# # batch_size, channels, height, width = img.shape
# # d_model = channels  # 通常 d_model 与 channels 大小一致

# # # 生成 positional encoding
# # positional_encoding = get_positional_encoding(height, width, d_model)

# # # 将 positional encoding 添加到图像
# # img_with_pe = img + positional_encoding.unsqueeze(0)  # 添加 batch 维度

import torch
import numpy as np

def get_pee2(height, width, d_model=10):
    channels = d_model
    pe = np.zeros((height, width, channels))
    
    for px in range(height):
        for py in range(width):
            for i in range(int(d_model / 2)):
                
                pe[px, py, 2*i] = np.sin(px / (10000 ** (2*i / d_model))) + \
                    np.sin(px / (10000 ** ((2*i + 1)/ d_model)))
            
                pe[px, py, 2*i + 1] = np.cos(px / (10000 ** (2*i  / d_model))) + \
                    np.cos(px / (10000 ** ((2*i + 1) / d_model)))
                
                # Ensure k + 3 is within bounds
                # if k + 3 < channels:
                #     pe[i, j, k + 3] = np.cos(j / (10000 ** (k / d_model)))
    pe_tensor = torch.tensor(pe, dtype=torch.float32).cuda()
    # pe_tensor = torch.tensor(pe, dtype=torch.float32)
    return pe_tensor

def get_pe(height, width, rgb, d_model=10):
    channels = d_model
    pe = np.zeros((height, width, channels))
    # rgb = rgb.cpu()
    print(rgb.shape)
    
    for px in range(height):
        for py in range(width):
            for i in range(d_model):
                print(px)
                print(py)
                r = rgb[px, py, 0]
                
                g = rgb[px, py, 1]
                b = rgb[px, py, 2]
                # print("r = ", r)
                # print("g = ", g)
                # print("b = ", b)
                
                if (i % 2) == 0:
                    pe[px, py, i] = np.sin(px / (10000 ** (i / d_model))) + \
                        np.sin(py / (10000 ** (i/ d_model))) + np.sin(r* 255/ (10000 ** (i / d_model))) + \
                        np.sin(g* 255/ (10000 ** (i / d_model))) + np.sin(b* 255/ (10000 ** (i / d_model)))
                else:
                    pe[px, py, i] = np.cos(px / (10000 ** ((i-1) / d_model))) + \
                        np.cos(py / (10000 ** ((i-1)/ d_model))) + np.cos(r* 255/ (10000 ** (i / d_model))) + \
                        np.cos(g* 255/ (10000 ** (i / d_model))) + np.cos(b* 255/ (10000 ** (i / d_model)))

                # pe[px, py, 2*i + 1] = np.cos(px / (10000 ** (2*i  / d_model))) + \
                #     np.cos(px / (10000 ** ((2*i + 1) / d_model)))
                
                # Ensure k + 3 is within bounds
                # if k + 3 < channels:
                #     pe[i, j, k + 3] = np.cos(j / (10000 ** (k / d_model)))
    pe_tensor = torch.tensor(pe, dtype=torch.float32).cuda()
    # pe_tensor = torch.tensor(pe, dtype=torch.float32)
    return pe_tensor


def get_pe3(height, width, rgb, d_model=10):
    channels = d_model
    pe = torch.zeros((channels, height, width)).cuda()
    # print(rgb.shape)
    
    for px in range(height):
        for py in range(width):
            for i in range(d_model):
                
                if (i % 2) == 0:
                    # pe[i, px, py] = np.sin((px + rgb[px, py].cpu()*255) / (10000 ** (i / d_model))) + \
                    #     np.sin((py + rgb[px, py].cpu()*255)/ (10000 ** (i/ d_model))) 
                    pe[i, px, py] = np.sin((px + rgb[px, py]*255) / (10000 ** (i / d_model))) + \
                        np.sin((py + rgb[px, py]*255)/ (10000 ** (i/ d_model))) 
                else:
                    # pe[i, px, py] = np.cos((px  + rgb[px, py].cpu()*255) / (10000 ** ((i-1) / d_model))) + \
                    #     np.cos((py  + rgb[px, py].cpu()*255) / (10000 ** ((i-1)/ d_model)))
                    pe[i, px, py] = np.cos((px  + rgb[px, py]*255) / (10000 ** ((i-1) / d_model))) + \
                        np.cos((py  + rgb[px, py]*255) / (10000 ** ((i-1)/ d_model)))
                    
    # pe_tensor = torch.tensor(pe, dtype=torch.float32).cuda()
    # pe_tensor = torch.tensor(pe, dtype=torch.float32)
    return pe


def get_pe4(height, width, single_channel, d_model=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将输入的单通道图像转换为PyTorch张量并移动到GPU
    single_channel = torch.tensor(single_channel * 255, dtype=torch.float32).to(device)
    
    # 创建位置编码张量
    pe = torch.zeros((d_model, height, width), device=device)
    
    # 创建位置坐标
    x = torch.arange(height, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(2)
    print(x.shape)
    y = torch.arange(width, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(2)
    print(y.shape)

    for i in range(d_model):
        div_term = 10000 ** (i / d_model)
        if i % 2 == 0:
            pe[i] = torch.sin((x + single_channel) / div_term) + torch.sin((y + single_channel) / div_term)
        else:
            pe[i] = torch.cos((x + single_channel) / div_term) + torch.cos((y + single_channel) / div_term)
    
    return pe

def get_pe_all(batch_size, height, width, rgb, d_model=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将RGB图像转换为张量并移动到GPU
    rgb = torch.tensor(torch.exp(rgb) * 255, dtype=torch.float32).to(device)
    # torch.
    
    # 创建位置编码张量
    pe = torch.zeros((batch_size, d_model, height, width), device=device)
    
    # 创建位置坐标
    x = torch.linspace(1, height, steps=height, device=device).view(-1, 1).repeat(1, width)
    y = torch.linspace(1, width, steps=width, device=device).view(1, -1).repeat(height, 1)

    for b in range(batch_size):
        for i in range(d_model):
            div_term = 10000 ** (i / d_model)
            if i % 2 == 0:
                pe[b, i] = (
                    torch.sin(x / div_term) +
                    torch.sin(y / div_term) +
                    torch.sin(rgb[b, 0, :, :] / div_term) +
                    torch.sin(rgb[b, 1, :, :] / div_term) +
                    torch.sin(rgb[b, 2, :, :] / div_term)
                )
            else:
                pe[b, i] = (
                    torch.cos(x / div_term) +
                    torch.cos(y / div_term) +
                    torch.cos(rgb[b, 0, :, :] / div_term) +
                    torch.cos(rgb[b, 1, :, :] / div_term) +
                    torch.cos(rgb[b, 2, :, :] / div_term)
                )
    
    return pe

def get_pe_single_rgb(batch_size, height, width, rgb, cc= 0,d_model=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将RGB图像转换为张量并移动到GPU
    rgb = torch.tensor(torch.exp(rgb) * 255, dtype=torch.float32).to(device)
    # torch.
    
    # 创建位置编码张量
    pe = torch.zeros((batch_size, d_model, height, width), device=device)
    
    # 创建位置坐标
    x = torch.linspace(1, height, steps=height, device=device).view(-1, 1).repeat(1, width)
    y = torch.linspace(1, width, steps=width, device=device).view(1, -1).repeat(height, 1)

    
    for b in range(batch_size):
        for i in range(d_model):
            div_term = 10000 ** (i / d_model)
            if i % 2 == 0:
                pe[b, i] = (
                    torch.sin(x / div_term) + # shape = [height, width]
                    torch.sin(y / div_term) +
                    torch.sin(rgb[b, cc, :, :] / div_term) 
                )
            else:
                pe[b, i] = (
                    torch.cos(x / div_term) +
                    torch.cos(y / div_term) +
                    torch.cos(rgb[b, cc, :, :] / div_term)
                )
    
    return pe



def get_color_mul_dim(batch_size, height, width, rgb, channel,d_model=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将RGB图像转换为张量并移动到GPU
    rgb = torch.exp(rgb) - 1e-3
    rgb = torch.tensor(rgb * 255, dtype=torch.float32).to(device)
    # torch.
    
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
                    (1/2) * torch.sin(rgb[b, channel, :, :] / div_term) + 0.5

                    # + torch.sin(rgb[b, 1, :, :] / div_term) +
                    # torch.sin(rgb[b, 2, :, :] / div_term)
                )
            else:
                pe[b, i] = (
                    # torch.cos(x / div_term) +
                    # torch.cos(y / div_term) +
                    (1/2) * torch.cos(rgb[b, channel, :, :] / div_term) + 0.5
                    # + torch.cos(rgb[b, 1, :, :] / div_term) +
                    # torch.cos(rgb[b, 2, :, :] / div_term)
                )
    
    return pe

def get_color_mul_dim2(batch_size, height, width, rgb, channel,d_model=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将RGB图像转换为张量并移动到GPU
    rgb = torch.exp(rgb) - 1e-3
    rgb = torch.tensor(rgb * 255, dtype=torch.float32).to(device)
    # torch.
    
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
                    -(1/2) * torch.sin(rgb[b, channel, :, :] / div_term) - 0.5

                    # + torch.sin(rgb[b, 1, :, :] / div_term) +
                    # torch.sin(rgb[b, 2, :, :] / div_term)
                )
            else:
                pe[b, i] = (
                    # torch.cos(x / div_term) +
                    # torch.cos(y / div_term) +
                    -(1/2) * torch.cos(rgb[b, channel, :, :] / div_term) - 0.5
                    # + torch.cos(rgb[b, 1, :, :] / div_term) +
                    # torch.cos(rgb[b, 2, :, :] / div_term)
                )
    
    return pe


def get_color_mul_dim_for_restormer(batch_size, height, width, rgb, channel,d_model=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将RGB图像转换为张量并移动到GPU
    # rgb = torch.exp(rgb) - 1e-3
    # rgb = torch.tensor(rgb * 255, dtype=torch.float32).to(device)

    rgb = torch.tensor(rgb * 64, dtype=torch.float32).to(device)

    # torch.
    
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
                    (1/2) * torch.sin(rgb[b, channel, :, :] / div_term) + 0.5

                    # + torch.sin(rgb[b, 1, :, :] / div_term) +
                    # torch.sin(rgb[b, 2, :, :] / div_term)
                )
            else:
                pe[b, i] = (
                    # torch.cos(x / div_term) +
                    # torch.cos(y / div_term) +
                    (1/2) * torch.cos(rgb[b, channel, :, :] / div_term) + 0.5
                    # + torch.cos(rgb[b, 1, :, :] / div_term) +
                    # torch.cos(rgb[b, 2, :, :] / div_term)
                )
    
    return pe


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # # 将RGB图像转换为张量并移动到GPU
    # rgb = torch.tensor(rgb * 255, dtype=torch.float32).to(device)
    
    # # 创建位置编码张量
    # pe = torch.zeros((d_model, height, width), device=device)
    
    # # 创建位置坐标
    # x = torch.linspace(0, height-1, steps=height, device=device).view(-1, 1).repeat(1, width)
    # y = torch.linspace(0, width-1, steps=width, device=device).view(1, -1).repeat(height, 1)
    
    # for i in range(d_model):
    #     div_term = 10000 ** (i / d_model)
    #     if i % 2 == 0:
    #         pe[i] = (
    #             torch.sin(x / div_term) +
    #             torch.sin(y / div_term) +
    #             torch.sin(rgb[:, :, 0] / div_term) +
    #             torch.sin(rgb[:, :, 1] / div_term) +
    #             torch.sin(rgb[:, :, 2] / div_term)
    #         )
    #     else:
    #         pe[i] = (
    #             torch.cos(x / div_term) +
    #             torch.cos(y / div_term) +
    #             torch.cos(rgb[:, :, 0] / div_term) +
    #             torch.cos(rgb[:, :, 1] / div_term) +
    #             torch.cos(rgb[:, :, 2] / div_term)
    #         )
    
    # return pe

from .utils import util
# from 
import cv2
import os
import matplotlib.pyplot as plt



def showTensor2(bsize,tmp , cond = 2):
    print(tmp.shape)
    # normal_imgs = util.tensor2img(tmp)
    # tmp = torch.exp(tmp)
    # tmp = tmp - 1e-3
    imgs = []
    for i in range(bsize):
        img = util.tensor2img(tmp[i,])
        imgs.append(img)
        if cond == 1:
            prefix = "/home/qc-lab/research/tw_files/test/graymap"
            save_path = os.path.join(prefix,'{}_graymap.png'.format(i))
            # print(save_path)
            util.save_img(img, save_path)

        elif cond == 11:
            prefix = "/home/qc-lab/research/tw_files/test/graymap_raw"
            save_path = os.path.join(prefix,'{}_graymap_raw.png'.format(i))
            # print(save_path)
            util.save_img(img, save_path)

        elif cond == 22:
            prefix = "/home/qc-lab/research/tw_files/test/graymap_nor"
            save_path = os.path.join(prefix,'{}_graymap_nor.png'.format(i))
            # print(save_path)
            util.save_img(img, save_path)
        
        elif cond == 3:
            prefix = "/home/qc-lab/research/tw_files/test/ace"
            save_path = os.path.join(prefix,'{}_ace.png'.format(i))
            # print(save_path)
            util.save_img(img, save_path)

        # cv2.imshow(f'Image {i+1}', img)
        # cv2.waitKey(0)

    plt.figure(figsize=(160, 160))

# 显示每张图片
    for i, img in enumerate(imgs):
        # plt.figure(figsize=(img.shape[1], img.shape[0])) 
        plt.subplot(1, len(imgs), i + 1)  # 根据图片数量调整布局
        plt.imshow(img)
        plt.axis('off')  # 隐藏坐标轴
        plt.title(f'Image {i+1}')

    plt.show()
    # tmp = input("abc: ")
    # print(normal_imgs.shape)


def showTensor(bsize,tmp, cond=1):
    print(tmp.shape)
    # normal_imgs = util.tensor2img(tmp)
    tmp = torch.exp(tmp)
    tmp = tmp - 1e-3
    imgs = []
    for i in range(bsize):
        img = util.tensor2img(tmp[i,])
        imgs.append(img)
        if cond == 1:
            prefix = "/home/qc-lab/research/tw_files/test/input_lli"
            save_path = os.path.join(prefix,'{}_input.png'.format(i))
            # print(save_path)
            util.save_img(img, save_path)
        elif cond == 11:
            prefix = "/home/qc-lab/research/tw_files/test/graymap"
            save_path = os.path.join(prefix,'{}_graymap.png'.format(i))
            # print(save_path)
            util.save_img(img, save_path)
        # cv2.imshow(f'Image {i+1}', img)
        # cv2.waitKey(0)

    plt.figure(figsize=(160, 160))

# 显示每张图片
    for i, img in enumerate(imgs):
        # plt.figure(figsize=(img.shape[1], img.shape[0])) 
        plt.subplot(1, len(imgs), i + 1)  # 根据图片数量调整布局
        plt.imshow(img)
        plt.axis('off')  # 隐藏坐标轴
        plt.title(f'Image {i+1}')

    plt.show()
    # tmp = input("abc: ")
    # print(normal_imgs.shape)



