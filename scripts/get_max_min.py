from PIL import Image
import os

def update_global_min_max(image_path, global_min_max):
    """更新全局 R、G、B 通道的最大值和最小值"""
    with Image.open(image_path) as img:
        # 确保图像是 RGB 格式
        img = img.convert('RGB')
        
        # 拆分为 R, G, B 三个通道
        r, g, b = img.split()

        # 获取每个通道的最小值和最大值
        r_min, r_max = r.getextrema()
        g_min, g_max = g.getextrema()
        b_min, b_max = b.getextrema()

        # 更新全局 min 和 max
        global_min_max['r_min'] = min(global_min_max['r_min'], r_min)
        global_min_max['r_max'] = max(global_min_max['r_max'], r_max)
        global_min_max['g_min'] = min(global_min_max['g_min'], g_min)
        global_min_max['g_max'] = max(global_min_max['g_max'], g_max)
        global_min_max['b_min'] = min(global_min_max['b_min'], b_min)
        global_min_max['b_max'] = max(global_min_max['b_max'], b_max)

def process_images_in_folder(folder_path):
    """遍历文件夹中的所有图像，计算所有图像的 R、G、B 通道的全局最大值和最小值"""
    # 初始化全局 R、G、B 最小值和最大值
    global_min_max = {
        'r_min': float('inf'), 'r_max': float('-inf'),
        'g_min': float('inf'), 'g_max': float('-inf'),
        'b_min': float('inf'), 'b_max': float('-inf')
    }

    # 遍历文件夹中的所有图像
    for filename in os.listdir(folder_path):
        # 只处理常见的图像格式
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            update_global_min_max(image_path, global_min_max)

    # 输出全局的 R、G、B 通道的最小值和最大值
    print("Global R channel - Min: {}, Max: {}".format(global_min_max['r_min'], global_min_max['r_max']))
    print("Global G channel - Min: {}, Max: {}".format(global_min_max['g_min'], global_min_max['g_max']))
    print("Global B channel - Min: {}, Max: {}".format(global_min_max['b_min'], global_min_max['b_max']))

if __name__ == "__main__":
    # 替换为你想要处理的图像文件夹路径
    # folder_path = "path/to/your/images/folder"
    folder_path = "/home/qc-lab/research/code/low-light/freetest/train_basicsr/datasets/NTIRE_2024/Input"
    # folder_path = "/home/qc-lab/research/code/low-light/freetest/train_basicsr/datasets/NTIRE_2024/high"
    process_images_in_folder(folder_path)