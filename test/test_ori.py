import torch.nn
import torch
import json

labels = json.load(open("/home/zh/pythonhub/yaolu/datasets/speed/train_label.json"))
ori_error = torch.tensor(0.0)
for key, value in labels.items():
    ori_label = torch.tensor(value["ori"])
    ori_norm = ori_label / torch.norm(ori_label)
    ori_inner_dot = torch.abs(torch.sum(ori_label * ori_norm))
    ori_inner_dot = torch.clamp(ori_inner_dot, max=1)
    ori_error += 2 * torch.arccos(ori_inner_dot)
print(torch.rad2deg(ori_error / len(labels)))

a = torch.tensor([1, 2, 3])
print(torch.sum(a[a>1.5]))