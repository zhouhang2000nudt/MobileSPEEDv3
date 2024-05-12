from torchvision.transforms import v2
from torch.utils.data import Dataset, random_split, Subset
from pathlib import Path
from threading import Thread
from tqdm import tqdm
from numpy import ndarray
from .utils import rotate_image, rotate_cam, Camera
from PIL import Image

import albumentations as A
import cv2 as cv
import lightning as L
import numpy as np

import json
import torch
import itertools


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def prepare_Speed(config: dict):
    # 准备数据集

    # 为Speed类添加属性
    Speed.config = config
    Speed.data_dir = Path(config["data_dir"])
    Speed.image_dir = Speed.data_dir / "images"
    Speed.label_file = Speed.data_dir / "labels.json"
    Speed.test_img_dir = Speed.data_dir / "test"
    Speed.real_test_img_dir = Speed.data_dir / "real_test"

    # 设置transform
    Speed.transform = {
        # 训练集的数据转化
        "train": {
            "transform": v2.Compose([
                v2.ToTensor(),
            ]),
            "A_transform": A.Compose([
                A.OneOf([
                    A.AdvancedBlur(blur_limit=(3, 5),
                                   rotate_limit=25,
                                   p=0.1),
                    A.Blur(blur_limit=(3, 5), p=0.1),
                    A.GaussNoise(var_limit=(5, 15),
                                 p=0.1),
                    A.GaussianBlur(blur_limit=(3, 5),
                                   sigma_limit=1.5,
                                   p=0.1),
                    ], p=0.1),
                A.ImageCompression(
                    quality_lower=95,
                    quality_upper=100,
                    p=0.1
                ),
                A.ColorJitter(brightness=0.2,
                              contrast=0.2,
                              saturation=0.2,
                              hue=0.2,
                              p=0.1),
                # A.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                #          p=0.2),
                # A.Compose([
                #     A.BBoxSafeRandomCrop(p=1.0),
                #     A.PadIfNeeded(min_height=480, min_width=768, border_mode=cv.BORDER_REPLICATE, position="random", p=1.0),
                # ], p=1),
            ],
            p=1,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))
        },
        # 验证集的数据转化
        "val": {
            "transform": v2.Compose([
                v2.ToTensor(),
            ]),
            "A_transform": None,
        },
        "test": {
            "transform": v2.Compose([
                v2.ToTensor(),
            ]),
            "A_transform": None,
        }
    }

    # 设置标签字典
    Speed.labels = json.load(open(Speed.label_file, "r"))
    Speed.test_labels = json.load(open(Speed.data_dir / "test.json", "r"))
    Speed.test_labels.extend(json.load(open(Speed.data_dir / "real_test.json", "r")))
    
    # 采样列表
    Speed.img_name = list(Speed.labels.keys())
    num = len(Speed.img_name)
    train_num = int(num * Speed.config["split"][0])
    val_num = num - train_num
    Speed.train_index, Speed.val_index = random_split(Speed.img_name, [train_num, val_num])
    # Speed.test_index = list(Speed.test_labels.keys())

    # 缓存图片
    if Speed.config["ram"]:
        Speed.img_dict = {}
        Speed.read_img()
        Speed.fake_img = np.zeros((1200, 1980, 3), dtype=np.uint8)
    
    # 生成所有可能的布尔值组合
    bool_values = [True, False]
    combinations = list(itertools.product(bool_values, repeat=4))
    # 将布尔值组合转换为独热编码
    for i, combo in enumerate(combinations):
        one_hot = [0] * 16
        one_hot[i] = 1
        Speed.encode_dict[combo] = one_hot
        Speed.decode_dict[i] = combo


class ImageReader(Thread):
    def __init__(self, img_name: list, config: dict, image_dir: Path):
        Thread.__init__(self)
        self.config: dict = config
        self.image_dir: Path = image_dir
        self.image_name: list = img_name
        self.img_dict: dict = {}
    
    def run(self):
        for img_name in tqdm(self.image_name):
            img = cv.imread(str(self.image_dir / img_name.replace(".jpg", ".png")), cv.IMREAD_GRAYSCALE)
            # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            # img = cv.resize(img, (self.config["imgsz"][1], self.config["imgsz"][0]))
            self.img_dict[img_name] = img
    
    def get_result(self) -> dict:
        return self.img_dict


class Speed(Dataset):
    data_dir: Path          # 数据集根目录
    image_dir: Path         # 图片目录
    po_file: Path           # 位姿json文件
    bbox_file: Path         # bbox json文件
    labels: dict            # 标签字典
    test_labels: dict       # 测试集标签字典
    config: dict            # 配置字典
    img_name: list     # 样本id列表
    transform: dict   # 数据转化方法字典
    train_index: Subset     # 训练集图片名列表
    val_index: Subset       # 验证集图片名列表
    test_index: list        # 测试集图片名列表
    img_dict: dict = {} # 图片字典
    fake_img: ndarray       # 伪图片
    camera: Camera = Camera
    encode_dict: dict = {}  # 编码姿态的字典
    decode_dict: dict = {}  # 解码姿态的字典
    

    def __init__(self, mode: str = "train"):
        self.A_transform = Speed.transform[mode]["A_transform"]
        self.transform = Speed.transform[mode]["transform"]
        self.sample_index = Speed.train_index if mode == "train" else Speed.val_index if mode == "val" else Speed.test_index
        self.mode = mode
    
    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, index) -> tuple:
        filename = self.sample_index[index].strip()                  # 图片文件名
        if Speed.config["ram"]:
            image = Speed.img_dict[filename]                         # 伪图片
        else:
            image = cv.imread(str(self.image_dir / filename.replace(".jpg", ".png")), cv.IMREAD_GRAYSCALE)   # 读取图片
        
        if self.mode != "test":
            ori = torch.tensor(self.labels[filename]["ori"])   # 姿态  (,4)
            pos = torch.tensor(self.labels[filename]["pos"])   # 位置  (,3)
            
            bbox = self.labels[filename]["bbox"]
            # 进行Albumentation增强
            if self.A_transform is not None:
                transformed = self.A_transform(image=image, bboxes=[bbox], category_ids=[1])
                image = transformed["image"]
        
            dice = np.random.rand(1)
            if self.mode == "train":
                if Speed.config["Rotate"]["Rotate_img"] and dice <= Speed.config["Rotate"]["p"]:
                    image, pos, ori = rotate_image(image, pos, ori, Speed.camera.K, Speed.config["Rotate"]["img_angle"])                  # bbox (1, 4)
                elif Speed.config["Rotate"]["Rotate_cam"] and dice > Speed.config["Rotate"]["p"]:
                    image, pos, ori = rotate_cam(image, pos, ori, Speed.camera.K, Speed.config["Rotate"]["cam_angle"])
            
            cls = torch.tensor(self.encode_dict[tuple((ori >= 0).tolist())])
            ori = ori ** 2
            
            y: dict = {
                "filename": filename,
                "pos": pos,
                "ori": ori,
                "cls": cls,
            }
        else:
            y: dict = {
                "filename": filename,
            }

        # resize
        image = Image.fromarray(image)
        image = image.resize((Speed.config["imgsz"][1], Speed.config["imgsz"][0]))
        
        # 使用torchvision转换图片
        image = self.transform(image)       # (1, 480, 768)
        image = image.repeat(3, 1, 1)       # (3, 480, 768)

        return image, y

    @staticmethod
    def divide_data(lst: list, n: int):
        # 将列表lst分为n份，最后不足一份单独一组
        k, m = divmod(len(lst), n)
        return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    
    @staticmethod
    def read_img(thread_num: int = 12):
        # 将采样列表中的图片读入内存
        img_divided: list = Speed.divide_data(Speed.img_name, thread_num)
        thread_list: list[ImageReader] = []
        for sub_img_name in img_divided:
            thread_list.append(ImageReader(sub_img_name, Speed.config, Speed.image_dir))
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()
        for thread in thread_list:
            Speed.img_dict.update(thread.get_result())


class SpeedDataModule(L.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config: dict = config
        prepare_Speed(config)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.speed_data_train: Speed = Speed("train")
            self.speed_data_val: Speed = Speed("val")
        elif stage == "validate":
            self.speed_data_val: Speed = Speed("val")
    
    def train_dataloader(self) -> MultiEpochsDataLoader:
        return MultiEpochsDataLoader(
            self.speed_data_train,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["workers"],
            persistent_workers=True
        )
    
    def val_dataloader(self) -> MultiEpochsDataLoader:
        return MultiEpochsDataLoader(
            self.speed_data_val,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["workers"],
            persistent_workers=True
        )



if __name__ == "__main__":
    import yaml

    with open("./cfg/base.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    prepare_Speed(config)
    speed_dataset = SpeedDataModule(config)
    speed_dataset.setup("fit")
    speed_dataset.speed_data_train[0]
    speed_dataset.speed_data_val[0]
    print()