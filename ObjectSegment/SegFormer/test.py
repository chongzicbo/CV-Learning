"""
Descripttion: chengbo's code
version: 1.0.0
Author: chengbo
Date: 2023-10-27 10:59:15
LastEditors: chengbo
LastEditTime: 2023-10-27 10:59:26
"""
import torch.nn.functional as F
import torch

a = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
b = F.interpolate(a, size=(4, 4), mode="bilinear")
print(a)
print(b)
print(a.shape, b.shape)
