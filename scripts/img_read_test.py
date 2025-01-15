import torch
import numpy as np

# from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor


# my_dict = {
#     "type": "disk"
# }

# # file_client = FileClient(my_dict.pop('type'), **my_dict)

# gt_path = "/home/qc-lab/research/datasets/IR_dataset/train/high/6.png"

# img_bytes = FileClient(gt_path, 'gt')
import cv2

# img = cv2.imread(gt_path)

# print(img)

# img = img.astype(np.float32) / 255.

# print(img)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# print(img)

# img = torch.from_numpy(img.transpose(2, 0, 1))
# img = img.float()
# print(img)


array = np.arange(256)

array = array.reshape((16, 16))

print(array)

img = array.astype(np.float32) / 255.
print(img)

img = torch.from_numpy(img)
print(img)


img = img.numpy()*255
img = img.astype(np.uint8)
print(img)
# img_tensor = torch.from_numpy(img)

# print(img_tensor)
