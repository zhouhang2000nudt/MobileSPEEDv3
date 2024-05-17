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

def CropAndPad(img: np.ndarray, bbox: List[float]):
    pass

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
image_name = "/home/zh/pythonhub/yaolu/MobileSPEEDv3/test/img000012.jpg"
image = cv2.imread(image_name)


ori = [-0.044155, -0.393581, -0.707956, -0.584759]
pos = [-0.111403, 0.452985, 13.204556]
bbox = [791, 426, 1486, 1192]

# ori = [-0.419541, -0.484436, -0.214179, 0.73718]
# pos = [-0.21081, -0.094466, 6.705986]
# bbox = [539, 222, 1036, 700]

config = get_config()
prepare_Speed(config)
speed = Speed("train")
image, y = speed[0]
pos = y["pos"]
ori = y["ori"]
bbox = y["bbox"]



image, pos, ori, M = rotate_cam(image, pos, ori, Camera.K, 10)
bbox = wrap_boxes(np.array([bbox]), M, width=1920, height=1200).tolist()[0]
bbox[0] = int(bbox[0])
bbox[1] = int(bbox[1])
bbox[2] = int(bbox[2])
bbox[3] = int(bbox[3])

dice = np.random.rand()

visualize(image, [bbox], category_ids, category_id_to_name, ori, pos)