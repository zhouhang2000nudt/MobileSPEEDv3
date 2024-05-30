import torch
import torch.nn as nn

from typing import List, Union
from torch import Tensor

from .block import SPPF, FPNPAN, RepECPHead, DCNv2, ShortBiFPN, SFPN, Head_single, ECP, RSC, Align

import timm

class Mobile_SPEEDv3_timm(nn.Module):
    def __init__(self, config: dict):
        super(Mobile_SPEEDv3_timm, self).__init__()
        
        self.backbone = config["backbone"]
        self.features = timm.create_model(config["backbone"], pretrained=config["pretrained"], features_only=True, out_indices=[-3, -2, -1])
        
        chennels = self.features.feature_info.channels()
        
        self.ECP_pos = ECP(in_channels=chennels, expand_ratio=config["expand_ratio"], pool_size=config["pool_size"])
        self.ECP_ori = ECP(in_channels=chennels, expand_ratio=config["expand_ratio"], pool_size=config["pool_size"])
        
        self.RSC = RSC()
        
        head_in_features = sum([int(chennels[i] * config["expand_ratio"][i] * config["pool_size"][i]**2) for i in range(3)])
        self.pos_head = Head_single(head_in_features, dim=3)
        self.yaw_head = Head_single(head_in_features, dim=int(360 // config["stride"] + 1 + 2 * config["neighbor"]), softmax=True)
        self.pitch_head = Head_single(head_in_features, dim=int(180 // config["stride"] + 1 + 2 * config["neighbor"]), softmax=True)
        self.roll_head = Head_single(head_in_features, dim=int(360 // config["stride"] + 1 + 2 * config["neighbor"]), softmax=True)
        
    def forward(self, x: Tensor):
        p3, p4, p5 = self.features(x)
        
        pos = self.ECP_pos([p3, p4, p5])
        pos = self.RSC(pos)
        pos = self.pos_head(pos)
        
        ori = self.ECP_ori([p3, p4, p5])
        ori = self.RSC(ori)
        yaw = self.yaw_head(ori)
        pitch = self.pitch_head(ori)
        roll = self.roll_head(ori)
        
        return pos, yaw, pitch, roll

    def switch_repvggplus_to_deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        if hasattr(self, 'stage1_aux'):
            self.__delattr__('stage1_aux')
        if hasattr(self, 'stage2_aux'):
            self.__delattr__('stage2_aux')
        if hasattr(self, 'stage3_first_aux'):
            self.__delattr__('stage3_first_aux')
        self.deploy = True