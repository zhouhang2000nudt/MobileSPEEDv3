import sys
sys.path.insert(0, sys.path[0]+"/../")

import cv2
import numpy as np

import matplotlib.pyplot as plt

from typing import List
from PIL import Image

from MobileSPEEDv3.utils.utils import Camera, visualize_axes, rotate_image, wrap_boxes, rotate_cam
from MobileSPEEDv3.utils.config import get_config
from MobileSPEEDv3.utils.dataset import Speed, prepare_Speed

import albumentations as A
from torchvision.transforms import Compose, Resize

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

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    # x_min, y_min, w, h = bbox
    x_min, y_min, x_max, y_max = bbox

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img

def visualize(image, bboxes, category_ids, category_id_to_name, ori, pos):
    bboxes[0][0] = int(bboxes[0][0])
    bboxes[0][1] = int(bboxes[0][1])
    bboxes[0][2] = int(bboxes[0][2])
    bboxes[0][3] = int(bboxes[0][3])
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    axis_length = 200
    visualize_axes(ax, np.array(ori), np.array(pos), Camera.K, axis_length)
    ax.imshow(img, cmap='gray')
    plt.show()

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


ori = [-0.419541, -0.484436, -0.214179, 0.73718]
pos = [-0.21081, -0.094466, 6.705986]
bbox = [539, 222, 1036, 700]

config = get_config()
prepare_Speed(config)


speed = Speed("train")
for i in range(len(speed)):
    image, y = speed[i]
pos = y["pos"]
ori = y["ori"]
bbox = y["bbox"]


# speed = Speed("self_supervised_train")
# image_1, image_2 = speed[0]
# visualize(image_1, [bbox], category_ids, category_id_to_name, ori, pos)
# visualize(image_2, [bbox], category_ids, category_id_to_name, ori, pos)

# image_name = "/home/zh/pythonhub/yaolu/MobileSPEEDv3/test/img008549.jpg"
# image = cv2.imread(image_name)
# ori = [-0.451242, 0.259848, 0.550933, 0.652175]
# pos = [-0.162408, -0.399831, 10.428022]
# bbox = [704, 449, 1109, 611]


# image, pos, ori, M = rotate_cam(image, pos, ori, Camera.K, 10)
# bbox = wrap_boxes(np.array([bbox]), M, width=1920, height=1200).tolist()[0]
# bbox[0] = int(bbox[0])
# bbox[1] = int(bbox[1])
# bbox[2] = int(bbox[2])
# bbox[3] = int(bbox[3])
# transformed = A_tranform(image=image, bboxes=[bbox], category_ids=category_ids)
# print(transformed["image"].shape)

# image = CropAndPad(image, bbox)
# print(image.shape)

# image = Image.fromarray(image)
# bbox = list(transformed["bboxes"][0])
# # image = transform(image)
# image = np.array(image)




visualize(image, [bbox], category_ids, category_id_to_name, ori, pos)