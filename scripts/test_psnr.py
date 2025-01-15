import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

i=0
from t1 import calculate_ssim, calculate_psnr, load_image_pillow

# def calculate_psnr_torch(img1, img2):
#     """Calculate the PSNR between two tensors."""
#     mse = torch.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')  # Perfect match
#     PIXEL_MAX = 1.0  # Since the images are normalized to [0, 1]
#     return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def load_image_as_tensor(image_path):
    """Load an image from file and convert it to a PyTorch tensor."""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()  # Convert the image to a [0, 1] tensor
    return transform(image)

def calculate_average_psnr(folder1, folder2):
    """Calculate the average PSNR between all images in two folders."""
    image_filenames = sorted(os.listdir(folder1))
    psnr_values = []

    for filename in image_filenames:
        img1_path = os.path.join(folder1, filename)
        img2_path = os.path.join(folder2, filename)

        # Check if both images exist
        if not os.path.exists(img2_path):
            print(f"Image {filename} is missing in folder2. Skipping...")
            continue

        # Load the images as tensors
        img1 = load_image_pillow(img1_path)
        img2 = load_image_pillow(img2_path)

        # Ensure both images have the same shape
        if img1.shape != img2.shape:
            print(f"Image {filename} dimensions do not match. Skipping...")
            continue

        # Calculate PSNR for the current pair
        psnr_value = calculate_psnr(img1, img2, 0).item()  # Convert tensor PSNR to Python float
        global i
        i += 1
        print(f"idx: {i}, its psnr: {psnr_value}")
        psnr_values.append(psnr_value)

    # Calculate average PSNR
    if len(psnr_values) == 0:
        print("No valid image pairs found.")
        return 0

    average_psnr = sum(psnr_values) / len(psnr_values)
    return average_psnr

if __name__ == "__main__":
    # Define the two folders
    folder1 = "/home/qc-lab/research/code/low-light/freetest/train_basicsr/datasets/mini_val/high"
    folder2 = "/media/qc-lab/back_up/1/LLIE/SYSU-FVL-T2/Enhancement/results/NtireLL"
    # folder2 = "/home/qc-lab/research/code/low-light/freetest/train_basicsr/datasets/mini_val/low"



    # Calculate the average PSNR
    avg_psnr = calculate_average_psnr(folder1, folder2)
    print(f"Average PSNR: {avg_psnr:.4f} dB")