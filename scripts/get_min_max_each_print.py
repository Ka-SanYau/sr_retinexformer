from PIL import Image
import os

import numpy as np

def get_max_min_rgb(image_path):
    """获取图像中 R、G、B 通道的最大值和最小值"""
    with Image.open(image_path) as img:
        # 确保图像是 RGB 格式
        img = img.convert('RGB')
        
        # 拆分为 R, G, B 三个通道
        r, g, b = img.split()

        # 获取每个通道的最小值和最大值
        r_min, r_max = r.getextrema()
        g_min, g_max = g.getextrema()
        b_min, b_max = b.getextrema()

        # r_mean = r.mean()
        # g_mean = g.getextrema()
        # b_mean = b.getextrema()

        return {
            'r_min': r_min, 'r_max': r_max,
            'g_min': g_min, 'g_max': g_max,
            'b_min': b_min, 'b_max': b_max
        }

def calculate_image_mean_rgb(image_path):
    """计算单张图像的 R、G、B 通道的均值"""
    with Image.open(image_path) as img:
        # 确保图像是 RGB 格式
        img = img.convert('RGB')
        
        # 将图像转为 numpy 数组
        img_np = np.array(img)
        
        # 分别计算 R、G、B 通道的均值
        r_mean = np.mean(img_np[:, :, 0])  # Red channel mean
        g_mean = np.mean(img_np[:, :, 1])  # Green channel mean
        b_mean = np.mean(img_np[:, :, 2])  # Blue channel mean

        return r_mean, g_mean, b_mean

def process_images_in_folder(folder_path):
    """遍历文件夹中的所有图像，输出每张图像的 R、G、B 通道的最大值和最小值"""
    for filename in os.listdir(folder_path):
        # 只处理常见的图像格式
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            rgb_stats = get_max_min_rgb(image_path)

            r_m, g_m, b_m = calculate_image_mean_rgb(image_path)
            
            print(f"Image: {filename}")
            print(f"R channel - Min: {rgb_stats['r_min']}, Max: {rgb_stats['r_max']}, Mean: {r_m}")
            print(f"G channel - Min: {rgb_stats['g_min']}, Max: {rgb_stats['g_max']}, Mean: {g_m}")
            print(f"B channel - Min: {rgb_stats['b_min']}, Max: {rgb_stats['b_max']}, Mean: {b_m}\n")

if __name__ == "__main__":
    # 替换为你想要处理的图像文件夹路径
    # folder_path = "path/to/your/images/folder"
    # folder_path = "/home/qc-lab/research/code/low-light/freetest/train_basicsr/datasets/NTIRE_2024/Input"
    folder_path = "/home/qc-lab/research/code/low-light/freetest/train_basicsr/datasets/NTIRE_2024/Train_GT"
    process_images_in_folder(folder_path)