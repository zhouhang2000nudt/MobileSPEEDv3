import torch
import torch.nn as nn
import timm

from typing import List, Union
from torch import Tensor

from .block import SPPF, FPNPAN, SFPN, Head_single, ECP, RSC, Align

class Mobile_SPEEDv3(nn.Module):
    def __init__(self, config: dict):
        super(Mobile_SPEEDv3, self).__init__()
        
        if config["self_supervised"]:
            self.supervised = True
        else:
            self.supervised = False
        self.backbone = config["backbone"]
        self.features = timm.create_model(config["backbone"], pretrained=config["pretrained"], features_only=True, out_indices=[-3, -2, -1])
        
        neck_in_channels = self.features.feature_info.channels()
        
        neck_out_channels = [
            max(64, int(neck_in_channels[0] / 2)),
            max(128, int(neck_in_channels[1] / 2)),
            max(256, int(neck_in_channels[2] / 2))
        ]
        
        self.chennel_align = Align(in_channels=neck_in_channels, out_channels=neck_out_channels)
        
        self.neck = SFPN(in_channels=neck_out_channels, fuse_mode="cat", SE=config["SE"])
        
        self.ECP_pos = ECP(in_channels=neck_out_channels, expand_ratio=config["expand_ratio"], pool_size=config["pool_size"])
        self.ECP_ori = ECP(in_channels=neck_out_channels, expand_ratio=config["expand_ratio"], pool_size=config["pool_size"])
        
        self.RSC = RSC()
        
        head_in_features = sum([int(neck_out_channels[i] * config["expand_ratio"][i] * config["pool_size"][i]**2) for i in range(3)])
        self.pos_head = Head_single(head_in_features, dim=3)
        self.yaw_head = Head_single(head_in_features, dim=int(360 // config["stride"] + 1 + 2 * config["neighbor"]), softmax=True)
        self.pitch_head = Head_single(head_in_features, dim=int(180 // config["stride"] + 1 + 2 * config["neighbor"]), softmax=True)
        self.roll_head = Head_single(head_in_features, dim=int(360 // config["stride"] + 1 + 2 * config["neighbor"]), softmax=True)
        
    def forward_once(self, x: Tensor):
        p3, p4, p5 = self.features(x)
        
        p3, p4, p5_pos, p5_ori = self.chennel_align([p3, p4, p5])
        
        pos_fs, ori_fs = self.neck([p3, p4, p5_pos, p5_ori])
        p3_pos, p4_pos, p5_pos = pos_fs
        p3_ori, p4_ori, p5_ori = ori_fs
        
        pos = self.ECP_pos([p3_pos, p4_pos, p5_pos])
        pos = self.RSC(pos)
        pos = self.pos_head(pos)
        
        ori = self.ECP_ori([p3_ori, p4_ori, p5_ori])
        ori = self.RSC(ori)
        yaw = self.yaw_head(ori)
        pitch = self.pitch_head(ori)
        roll = self.roll_head(ori)
        return pos, yaw, pitch, roll
    
    def forward(self, x1: Tensor, x2: Tensor = None):
        if self.supervised:
            return self.forward_once(x1), self.forward_once(x2)
        return self.forward_once(x1)

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