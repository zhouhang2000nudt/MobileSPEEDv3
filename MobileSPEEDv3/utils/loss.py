import torch
import torch.nn as nn

from torch import Tensor
from functools import partial

@torch.jit.script
def entropy(p: Tensor, q: Tensor, eps1: float = 0, eps2: float = 0):
    return torch.sum(p * torch.log(p / (q+eps1) + eps2), dim=1)

# 回归损失函数
@torch.jit.script
def MAE_Loss(pre: Tensor, label: Tensor):
    return torch.mean(torch.sum(torch.abs(pre - label), dim=1))

@torch.jit.script
def MSE_Loss(pre: Tensor, label: Tensor):
    return torch.mean(torch.sum((pre - label) ** 2, dim=1))

@torch.jit.script
def Huber_Loss(pre: Tensor, label: Tensor, delta: float = 1.0):
    abs_err = torch.abs(pre - label)
    loss = torch.where(abs_err < delta, 0.5 * abs_err ** 2, delta * abs_err - 0.5 * delta**2)
    return torch.mean(torch.sum(loss, dim=1))

@torch.jit.script
def Log_Cosh_Loss(pre: Tensor, label: Tensor):
    return torch.mean(torch.sum(torch.log(torch.cosh(pre - label)), dim=1))

# 分类损失函数
@torch.jit.script
def CrossEntropy_Loss(pre: Tensor, label: Tensor):
    return torch.mean(torch.sum(-label * torch.log(pre), dim=1))

@torch.jit.script
def JS_Divergence(pre: Tensor, label: Tensor):
    m = 0.5 * (pre + label)
    return torch.mean(0.5 * (entropy(pre, m, eps2=1e-12) + entropy(label, m, eps1=0)))

@torch.jit.script
def KL_Divergence(pre: Tensor, label: Tensor):
    return torch.mean(entropy(pre, label, eps1=1e-12, eps2=1e-12))

@torch.jit.script
def Focal_Loss(pre: Tensor, label: Tensor, gamma: float = 2.0, alpha: float = 0.25):
    eps = 1e-7
    CE = -label * torch.log(pre + eps)
    FLoss = alpha * torch.pow(1 - pre, gamma) * CE
    return torch.mean(torch.sum(FLoss, dim=1))


def get_reg_loss(loss_type: str, **kwargs):
    if loss_type not in ["MAE", "MSE", "Huber", "Log_Cosh"]:
        raise ValueError("Invalid loss type.")
    if loss_type == "MAE":
        return MAE_Loss
    elif loss_type == "MSE":
        return MSE_Loss
    elif loss_type == "Huber":
        return partial(Huber_Loss, **kwargs)
    elif loss_type == "Log_Cosh":
        return Log_Cosh_Loss

def get_cls_loss(loss_type: str, **kwargs):
    if loss_type not in ["CrossEntropy", "Focal", "JS_Divergence", "KL_Divergence"]:
        raise ValueError("Invalid loss type.")
    if loss_type == "CrossEntropy":
        return CrossEntropy_Loss
    elif loss_type == "Focal":
        return partial(Focal_Loss, **kwargs)
    elif loss_type == "JS_Divergence":
        return JS_Divergence
    elif loss_type == "KL_Divergence":
        return KL_Divergence

# ====================位置损失====================

class PoseLoss(nn.Module):
    def __init__(self, loss_type: str, **kwargs):
        super(PoseLoss, self).__init__()
        self.loss = get_reg_loss(loss_type, **kwargs)
    
    def forward(self, pos_pre, pos_label):
        return self.loss(pos_pre, pos_label)

# ===================euler loss===================
class EulerLoss(nn.Module):
    def __init__(self, loss_type: str, **kwargs):
        super(EulerLoss, self).__init__()
        self.loss = get_cls_loss(loss_type, **kwargs)
    
    def forward(self, euler_pre, euler_label):
        return self.loss(euler_pre, euler_label)


# ===================ori loss===================
class OriLoss(nn.Module):
    def __init__(self, loss_type: str, **kwargs):
        super(OriLoss, self).__init__()
        self.loss = get_reg_loss(loss_type, **kwargs)
    
    def forward(self, ori_pre, ori_label):
        return self.loss(ori_pre, ori_label)