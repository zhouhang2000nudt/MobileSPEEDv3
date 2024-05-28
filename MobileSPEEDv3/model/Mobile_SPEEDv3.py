import torch
import torch.nn as nn

from typing import List, Union
from torch import Tensor

from .block import SPPF, FPNPAN, RepECPHead, DCNv2, ShortBiFPN, SFPN, Head_single, ECP, RSC, Align
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models import regnet_y_1_6gf, RegNet_Y_1_6GF_Weights

class Mobile_SPEEDv3(nn.Module):
    def __init__(self, config: dict):
        super(Mobile_SPEEDv3, self).__init__()
        
        self.backbone = config["backbone"]
        
        if config["backbone"] == "mobilenet_v3_large":
            if config["pretrained"]:
                self.features = mobilenet_v3_large(weights = MobileNet_V3_Large_Weights.DEFAULT).features[:-1]
            else:
                self.features = mobilenet_v3_large().features[:-1]
        elif config["backbone"] == "mobilenet_v3_small":
            if config["pretrained"]:
                self.features = mobilenet_v3_small(weights = MobileNet_V3_Small_Weights.DEFAULT).features[:-1]
            else:
                self.features = mobilenet_v3_small().features[:-1]
        elif config["backbone"] == "efficientnet_b1":
            if config["pretrained"]:
                self.features = efficientnet_b1(weights = EfficientNet_B1_Weights.DEFAULT).features[:-1]
            else:
                self.features = efficientnet_b1().features[:-1]
        if config["backbone"] == "mobilenet_v3_large":
            SPPF_in_channels = 160
            SPPF_out_channels = 160
        elif config["backbone"] == "mobilenet_v3_small":
            pass
        elif config["backbone"] == "efficientnet_b1":
            SPPF_in_channels = 320
            SPPF_out_channels = 320
        self.SPPF_pos = SPPF(in_channels=SPPF_in_channels,
                             out_channels=SPPF_out_channels)
        self.SPPF_ori = SPPF(in_channels=SPPF_in_channels,
                             out_channels=SPPF_out_channels)
                
        if config["backbone"] == "mobilenet_v3_large":
            neck_in_channels = [40, 112, SPPF_out_channels]
        elif config["backbone"] == "mobilenet_v3_small":
            pass
        elif config["backbone"] == "efficientnet_b1":
            neck_in_channels = [112, 192, SPPF_out_channels]
        neck_out_channels = [40, 80, 160]
        
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
        
    def forward(self, x: Tensor):
        # if self.backbone == "mobilenet_v3_large":
        #     p3 = self.features[:7](x)
        #     p4 = self.features[7:13](p3)
        #     p5 = self.features[13:](p4)
        # elif self.backbone == "mobilenet_v3_small":
        #     pass
        # elif self.backbone == "efficientnet_b1":
        #     p3 = self.features[:6](x)
        #     p4 = self.features[:7](x)
        #     p5 = self.features[:8](x)
        
        p3 = self.features[:7](x)
        p4 = self.features[7:13](p3)
        p5 = self.features[13:](p4)
        
        p3, p4, p5 = self.chennel_align([p3, p4, p5])
        
        p5_pos = self.SPPF_pos(p5)
        p5_ori = self.SPPF_ori(p5)
        
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