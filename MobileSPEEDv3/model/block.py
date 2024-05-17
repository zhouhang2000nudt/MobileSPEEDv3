import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Union
from .RepVGG import RepVGGplusBlock
from functools import partial
from torchvision.ops import Conv2dNormActivation
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig

from timm.layers.conv_bn_act import ConvBnAct
from timm.models._efficientnet_blocks import InvertedResidual

# ==================== block ===================




# ================== block end =================


# ==================== tail ====================

class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 5):
        super(SPPF, self).__init__()
        c_ = in_channels // 2
        self.conv1 = Conv2dNormActivation(in_channels=in_channels, out_channels=c_, kernel_size=1, stride=1)
        self.conv2 = Conv2dNormActivation(in_channels=c_*4, out_channels=out_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
    
    def forward(self, x):
        x = self.conv1(x)
        y5x5 = self.pool(x)
        y9x9 = self.pool(y5x5)
        y13x13 = self.pool(y9x9) 
        return self.conv2(torch.cat([x, y5x5, y9x9, y13x13], dim=1))

# ================== tail end ==================





# ==================== neck ====================

class FPNPAN(nn.Module):
    def __init__(self, in_channels: List[int], fuse_mode: str = "cat"):
        super(FPNPAN, self).__init__()
        self.UpSample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.fuse_mode = fuse_mode
        
        if fuse_mode == "cat":
            fused_channel_p45 = in_channels[1] + in_channels[2]
            fused_channel_p34 = in_channels[0] + in_channels[1]
        elif fuse_mode == "add":
            raise NotImplementedError("Not implemented yet")
            pass
        
        # 上采样通路
        self.p4_fuseconv_up = RepVGGplusBlock(in_channels=fused_channel_p45, out_channels=in_channels[1], kernel_size=3, stride=1, padding=1)
        self.p3_fuseconv_up = RepVGGplusBlock(in_channels=fused_channel_p34, out_channels=in_channels[0], kernel_size=3, stride=1, padding=1)
        
        # 下采样通路
        self.p3_downconv_down = ConvBnAct(in_channels=in_channels[0], out_channels=in_channels[0], kernel_size=3, stride=2)
        
        self.p4_fuseconv_down = RepVGGplusBlock(in_channels=fused_channel_p34, out_channels=in_channels[1], kernel_size=3, stride=1, padding=1)
        self.p4_downconv_down = ConvBnAct(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3, stride=2)
        
        self.p5_fuseconv_down = RepVGGplusBlock(in_channels=fused_channel_p45, out_channels=in_channels[2], kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        p3, p4, p5 = x      # in: 40, 60, 96; p4: 112, 30, 48; p5: 160, 15, 24
        
        # 上采样通路
        if self.fuse_mode == "cat":
            p4_fused_up = self.p4_fuseconv_up(torch.cat([F.interpolate(p5, size=p4.shape[2:], mode="bilinear", align_corners=True), p4], dim=1)) # 112, 30, 48
        
        if self.fuse_mode == "cat":
            p3_fused_up = self.p3_fuseconv_up(torch.cat([F.interpolate(p4_fused_up, size=p3.shape[2:], mode="bilinear", align_corners=True), p3], dim=1)) # 40, 60, 96    out
        
        # 下采样通路
        if self.fuse_mode == "cat":
            p4_fused_down = self.p4_fuseconv_down(torch.cat([self.p3_downconv_down(p3_fused_up), p4_fused_up], dim=1)) # 112, 30, 48    out
        
        if self.fuse_mode == "cat":
            p5_fused_down = self.p5_fuseconv_down(torch.cat([self.p4_downconv_down(p4_fused_down), p5], dim=1)) # 160, 15, 24    out
        
        return p3_fused_up, p4_fused_down, p5_fused_down


# ================== neck end ==================





# ==================== head ====================

class ECP(nn.Module):
    def __init__(self, in_channels: List[int], expand_ratio: Union[int, List[float]], pool_size: List[int]):
        super(ECP, self).__init__()
        self.ECP_p3 = nn.Sequential(
            RepVGGplusBlock(
                in_channels=in_channels[0],
                out_channels=int(in_channels[0] * expand_ratio[0]),
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.AdaptiveAvgPool2d((pool_size[0], pool_size[0])),
        )
        self.ECP_p4 = nn.Sequential(
            RepVGGplusBlock(
                in_channels=in_channels[1],
                out_channels=int(in_channels[1] * expand_ratio[1]),
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.AdaptiveAvgPool2d((pool_size[1], pool_size[1])),
        )
        self.ECP_p5 = nn.Sequential(
            RepVGGplusBlock(
                in_channels=in_channels[2],
                out_channels=int(in_channels[2] * expand_ratio[2]),
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.AdaptiveAvgPool2d((pool_size[2], pool_size[2])),
        )
    
    def forward(self, x):
        p3, p4, p5 = x
        p3 = self.ECP_p3(p3)
        p4 = self.ECP_p4(p4)
        p5 = self.ECP_p5(p5)
        return p3, p4, p5


class RSC(nn.Module):
    def __init__(self):
        super(RSC, self).__init__()
    
    def forward(self, x):
        p3, p4, p5 = x
        return torch.cat([p3.reshape(p3.size(0), -1), p4.reshape(p4.size(0), -1), p5.reshape(p5.size(0), -1)], dim=1)


class Head(nn.Module):
    def __init__(self, in_features: int, pos_dim: int, ori_dim: int, cls_dim: int = 16):
        super(Head, self).__init__()
        features_ = in_features // 2
        self.pos_dim = pos_dim
        self.ori_dim = ori_dim
        self.cls_dim = cls_dim
        self.fc = nn.Sequential(
            nn.Linear(in_features, features_),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.pos_fc = nn.Sequential(
            nn.Linear(features_ // 2, features_ // 4),
            nn.Linear(features_ // 4, pos_dim)
        )
        self.ori_cls_fc = nn.Sequential(
            nn.Linear(features_ // 2, features_ // 4),
            nn.Linear(features_ // 4, ori_dim + cls_dim),
        )
    
    def forward(self, x):
        x = self.fc(x)
        pos_feature, ori_cls_feature = torch.chunk(x, 2, dim=1)
        pos = self.pos_fc(pos_feature)
        ori_cls = self.ori_cls_fc(ori_cls_feature)
        ori, cls = torch.split(ori_cls, [self.ori_dim, self.cls_dim], dim=1)
        ori = torch.softmax(ori, dim=1)
        cls = torch.softmax(cls, dim=1)
        return pos, ori, cls

class RepECPHead(nn.Sequential):
    def __init__(self, in_channels: List[int], expand_ratio: Union[int, List[float]], pool_size: List[int], pos_dim: int, ori_dim: int, cls_dim: int = 16):
        if isinstance(expand_ratio, int):
            expand_ratio = [expand_ratio] * 3
        super(RepECPHead, self).__init__(
            ECP(in_channels, expand_ratio, pool_size),
            RSC(),
            Head(sum([int(in_channels[i] * expand_ratio[i] * pool_size[i]**2) for i in range(3)]), pos_dim, ori_dim, cls_dim),
        )


# ================== head end ==================



#https://oneapi.xty.app/v1/chat/completions
#sk-ReUPocItmhmGioyMC63d6e0eBf584e3a812dF4E97696Aa92