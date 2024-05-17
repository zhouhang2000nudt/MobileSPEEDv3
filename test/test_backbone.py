import sys
sys.path.insert(0, sys.path[0]+"/../")

import torch
import torch.nn as nn

import timm
import torchvision.models as models

from rich import print
from ptflops import get_model_complexity_info
from MobileSPEEDv3.model import Mobile_SPEEDv3
from MobileSPEEDv3.utils.config import get_config


def profile_model(model):
    flops, params = get_model_complexity_info(model, (3, 480, 768), as_strings=True, print_per_layer_stat=False, verbose=False, flops_units="GMac", param_units="M", output_precision=10)
    print(flops)
    print(params)


print(timm.list_models("*mobilenet*"))

timm_model = timm.create_model("mobilenetv3_large_100", pretrained=False, features_only=True)
del timm_model.blocks[-1]
print(timm_model.default_cfg)
profile_model(timm_model)
print(timm_model)
print(timm_model.feature_info.channels())
print(timm_model.feature_info.reduction())

print("=====================================")

config = get_config()
full_model = Mobile_SPEEDv3(config)
full_model.eval()
full_model.switch_repvggplus_to_deploy()
profile_model(full_model)