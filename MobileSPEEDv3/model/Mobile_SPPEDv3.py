import torch.nn as nn

from functools import partial
from torch import Tensor
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from .block import SPPF, FPNPAN, RepECPHead, Conv2dNormActivation


class Mobile_SPEEDv3(nn.Module):
    def __init__(self, pos_dim: int, ori_dim: int, num_class: int = 16, pretrained: bool = True):
        super(Mobile_SPEEDv3, self).__init__()

        # features
        # 第 6  层的输出为 p3 (40,  60, 96)
        # 第 12 层的输出为 p4 (112, 30, 48)
        # 第 15 层的输出为 p5 (160, 15, 24)
        self.features = mobilenet_v3_large(weights = MobileNet_V3_Large_Weights.DEFAULT).features[:-1]
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        self.features[0] = Conv2dNormActivation(
                1,
                16,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        
        self.SPPF = SPPF(in_channels=self.features[-1].out_channels,
                         out_channels=self.features[-1].out_channels)
        
        # 把SPPF加到features的最后一层
        self.features.add_module("SPPF", self.SPPF)
        
        neck_in_channels = [self.features[6].out_channels, self.features[12].out_channels, self.features[15].out_channels]
        neck_out_channels = neck_in_channels
        self.FPNPAN = FPNPAN(in_channels=neck_in_channels)
        
        self.RepECPHead = RepECPHead(in_channels=neck_out_channels, 
                                     expand_ratio=[2, 1.5, 1],
                                     pool_size=[3, 2, 1],
                                     pos_dim=pos_dim,
                                     ori_dim=ori_dim,)
        
    def forward(self, x: Tensor):
        p3 = self.features[:7](x)
        p4 = self.features[7:13](p3)
        p5 = self.features[13:](p4)
        
        p3, p4, p5 = self.FPNPAN([p3, p4, p5])
        
        pos, ori, cls = self.RepECPHead([p3, p4, p5])
        
        return pos, ori, cls