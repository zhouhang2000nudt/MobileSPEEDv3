import sys
sys.path.insert(0, sys.path[0]+"/../")

import cv2
import numpy as np

import matplotlib.pyplot as plt

from typing import List
from PIL import Image

from MobileSPEEDv3.utils.utils import wrap_boxes, rotate_image, Camera, rotate_cam
from MobileSPEEDv3.utils.config import get_config
from MobileSPEEDv3.utils.dataset import Speed, prepare_Speed
from MobileSPEEDv3.utils.vis import visualize

import albumentations as A
from torchvision.transforms import Compose, Resize

import torch

def CropAndPad(img: np.array, bbox: List[float]):
    # 对图片进行裁剪
    # 裁剪后padding回原来的大小
    x_min, y_min, x_max, y_max = bbox
    height, width = img.shape[:2]
    crop_x_min = np.random.randint(0, x_min+1)
    crop_y_min = np.random.randint(0, y_min+1)
    crop_x_max = np.random.randint(x_max, width)
    crop_y_max = np.random.randint(y_max, height)
    img = img[crop_y_min:crop_y_max+1, crop_x_min:crop_x_max+1]
    img = cv2.copyMakeBorder(img, crop_y_min, 0, 0, 0,
                             np.random.choice([cv2.BORDER_REPLICATE, cv2.BORDER_CONSTANT]),
                             value=(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)))
    img = cv2.copyMakeBorder(img, 0, height-crop_y_max-1, 0, 0,
                             np.random.choice([cv2.BORDER_REPLICATE, cv2.BORDER_CONSTANT]),
                             value=(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)))
    img = cv2.copyMakeBorder(img, 0, 0, crop_x_min, 0,
                             np.random.choice([cv2.BORDER_REPLICATE, cv2.BORDER_CONSTANT]),
                             value=(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)))
    img = cv2.copyMakeBorder(img, 0, 0, 0, width-crop_x_max-1,
                             np.random.choice([cv2.BORDER_REPLICATE, cv2.BORDER_CONSTANT]),
                             value=(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)))
    return img



BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

category_ids = [1]
category_id_to_name = {1: 'satellite'}


A_tranform = A.Compose([
                A.OneOf([
                    A.AdvancedBlur(blur_limit=(3, 7),
                                   rotate_limit=25,
                                   p=0.2),
                    A.Blur(blur_limit=(3, 7), p=0.2),
                    A.GaussNoise(var_limit=(5, 15),
                                 p=0.2),
                    A.GaussianBlur(blur_limit=(3, 7),
                                   p=0.2),
                    ], p=1),
                A.ColorJitter(brightness=0.3,
                              contrast=0.3,
                              saturation=0.3,
                              hue=0.3,
                              p=1),
                A.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                         p=0),
                A.Compose([
                    A.BBoxSafeRandomCrop(p=1.0),
                    A.PadIfNeeded(min_height=1200, min_width=1920, border_mode=cv2.BORDER_REPLICATE, position="random", p=1.0),
                ], p=0),
            ],
            p=1,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))

transform = Compose([
    Resize(480),
])


a = torch.tensor([[1],
                  [2],
                  [3],
                  [4]])
b = torch.reshape(a, (1, 4))

print(a * b)

ori = [-0.419541, -0.484436, -0.214179, 0.73718]
pos = [-0.21081, -0.094466, 6.705986]
bbox = [539, 222, 1036, 700]


config = get_config()
prepare_Speed(config)
speed = Speed("train")
for i in range(len(speed)):
    image, y = speed[i]
    break
print(y["filename"])
print(image.shape)
pos = y["pos"]
ori = y["ori"]
bbox = y["bbox"]
print("ori", ori)
print("pos", pos)
print("bbox", bbox)
# image = cv2.resize(image, (1920, 1200))


# speed = Speed("self_supervised_train")
# image_1, image_2 = speed[0]
# visualize(image_1, [bbox], category_ids, category_id_to_name, ori, pos)
# visualize(image_2, [bbox], category_ids, category_id_to_name, ori, pos)

# image_name = "/home/zh/pythonhub/yaolu/MobileSPEEDv3/test/img000001.jpg"
# image = cv2.imread(image_name)
# ori = [-0.419541, -0.484436, -0.214179, 0.73718]
# pos = [-0.21081, -0.094466, 6.705986]
# bbox = [539, 222, 1036, 700]

# image_name = "/home/zh/pythonhub/yaolu/MobileSPEEDv3/test/img008549.jpg"
# image = cv2.imread(image_name)
# ori = [-0.451242, 0.259848, 0.550933, 0.652175]
# pos = [-0.162408, -0.399831, 10.428022]
# bbox = [704, 449, 1109, 611]


# image, pos, ori, M = rotate_cam(image, np.array(pos), np.array(ori), Camera.K, 20)
# bbox = wrap_boxes(np.array([bbox]), M, width=1920, height=1200).tolist()[0]
# transformed = A_tranform(image=image, bboxes=[bbox], category_ids=category_ids)
# print(transformed["image"].shape)

# image = CropAndPad(image, bbox)
# print(image.shape)

# image = Image.fromarray(image)
# bbox = list(transformed["bboxes"][0])
# # image = transform(image)
# image = np.array(image)


visualize(image, [bbox], category_ids, category_id_to_name, ori, pos, Camera.K, scale=2.5)