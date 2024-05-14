import torch
import torch.nn as nn
import timm

from typing import List, Union
from torch import Tensor

from .block import SPPF, FPNPAN, RepECPHead, Conv2dNormActivation
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class Mobile_SPEEDv3(nn.Module):
    def __init__(self, config: dict):
        super(Mobile_SPEEDv3, self).__init__()

        # features
        # 第 6  层的输出为 p3 (40,  60, 96)
        # 第 12 层的输出为 p4 (112, 30, 48)
        # 第 15 层的输出为 p5 (160, 15, 24)
        # self.features = timm.create_model(config["backbone"], pretrained=config["pretrained"], features_only=True, out_indices=(2, 3, 4))
        # assert self.features.feature_info.reduction() == [8, 16, 32], "Feature reduction must be [8, 16, 32]"
        
        self.features = mobilenet_v3_large(weights = MobileNet_V3_Large_Weights.DEFAULT).features
        print(self.features)
        SPPF_in_channels = 960
        SPPF_out_channels = 960 // 2
        neck_in_channels = [40, 112, SPPF_out_channels]
        
        
        # SPPF_in_channels = self.features.feature_info.channels()[-1]
        # if self.features.feature_info.channels()[-1] > 4*self.features.feature_info.channels()[-2]:
        #     SPPF_out_channels = self.features.feature_info.channels()[-1] // 2
        # else:
        #     SPPF_out_channels = self.features.feature_info.channels()[-1]
        # neck_in_channels = self.features.feature_info.channels()
        # neck_in_channels[-1] = SPPF_out_channels
        
        
        self.SPPF = SPPF(in_channels=SPPF_in_channels,
                         out_channels=SPPF_out_channels)
        
        
        self.FPNPAN = FPNPAN(in_channels=neck_in_channels, out_channels=config["out_channels"])
        
        neck_out_channels = config["out_channels"]
        
        self.RepECPHead = RepECPHead(in_channels=neck_out_channels, 
                                     expand_ratio=config["RepECPHead_expand_ratio"],
                                     pos_dim=config["pos_dim"],
                                     ori_dim=config["ori_dim"],
                                     cls_dim=config["cls_dim"])
        
    def forward(self, x: Tensor):
        p3 = self.features[:7](x)
        p4 = self.features[7:13](p3)
        p5 = self.features[13:](p4)
        
        # p3, p4, p5 = self.features(x)
        
        p5 = self.SPPF(p5)
        
        p3, p4, p5 = self.FPNPAN([p3, p4, p5])
        
        pos, ori, cls = self.RepECPHead([p3, p4, p5])
        
        return pos, ori, cls