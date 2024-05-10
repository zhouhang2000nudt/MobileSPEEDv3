import cv2
import matplotlib.pyplot as plt

from .utils import visualize_axes


BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (0, 0, 0) # White

def visualize_bbox(img, bbox, class_name, text_pos_error, text_ori_error, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    # staellite
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.5 * text_height)*3), (x_min + int(text_width), y_min - int(1.5 * text_height)*2), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height) - int(1.5 * text_height)*2),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.5,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
        thickness=2
    )
    
    # pos_error
    ((text_width, text_height), _) = cv2.getTextSize(text_pos_error, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.5 * text_height)*2), (x_min + int(text_width), y_min - int(1.5 * text_height)*1), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=text_pos_error,
        org=(x_min, y_min - int(0.3 * text_height) - int(1.5 * text_height)*1),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.5,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
        thickness=2
    )
    
    # ori_error
    ((text_width, text_height), _) = cv2.getTextSize(text_ori_error, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.5 * text_height)), (x_min + int(text_width), y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=text_ori_error,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.5,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
        thickness=2
    )
    return img

def visualize(image, bboxes, category_ids, category_id_to_name, text_pos_error, text_ori_error):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name, text_pos_error, text_ori_error)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)


def vis_axes(image_original, q_gt, loc_gt, K):
    fig, ax_1 = plt.subplots(1, 1, figsize=(12, 8))
    ax_1.imshow(image_original, cmap='gray')
    ax_1.set_xticks([])
    ax_1.set_yticks([])
    axis_length = 200
    visualize_axes(ax_1, q_gt, loc_gt, K, axis_length)
    plt.show()