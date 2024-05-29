import sys
sys.path.insert(0, sys.path[0]+"/../")

from MobileSPEEDv3.utils.utils import Camera
from MobileSPEEDv3.utils.config import get_config
from MobileSPEEDv3.utils.dataset import Speed, prepare_Speed
from MobileSPEEDv3.utils.vis import visualize

category_ids = [1]
category_id_to_name = {1: 'satellite'}

config = get_config()
camera = Camera(config)
config["ram"] = False
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

visualize(image, [bbox], category_ids, category_id_to_name, ori, pos, camera.K, scale=1200/config["imgsz"][0])