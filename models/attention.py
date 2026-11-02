"""已有 SE / CBAM 注意力方法的教学示例；不是 YOLO11，也不是原创算法。

这些模块是参考实现，用来理解如何在网络里加入注意力。
当前 train.py 未接入这些模块，不能据此宣称现有模型有注意力改进或精度提升。
真正的嵌入方式（对普通用户可选）：
  1. 把 SE / CBAM 注册进 ultralytics.nn.modules，并在 ultralytics.nn.tasks 中 import；
  2. 写一个自定义 model.yaml，在 backbone 里引用它们，例如：
       backbone:
         - [-1, 1, Conv, [64, 3, 2]]
         - [-1, 1, C2f, [64, 1, True]]
         - [-1, 1, SE, [64]]
         ...
  注意：CPU 上不建议大改网络结构，先跑通标准训练更重要。
"""
import torch
import torch.nn as nn


class SE(nn.Module):
    """Squeeze-and-Excitation 通道注意力。"""

    def __init__(self, c, r=16):
        super().__init__()
        hidden = max(c // r, 8)
        self.fc = nn.Sequential(
            nn.Linear(c, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(torch.mean(x, dim=(2, 3)).view(b, c))
        return x * w.view(b, c, 1, 1)


class CBAM(nn.Module):
    """Convolutional Block Attention Module：通道注意力 + 空间注意力。"""

    def __init__(self, c, r=16):
        super().__init__()
        hidden = max(c // r, 8)
        self.channel_fc = nn.Sequential(
            nn.Linear(c, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c),
            nn.Sigmoid(),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 通道注意力
        b, c, _, _ = x.shape
        w = self.channel_fc(torch.mean(x, dim=(2, 3)).view(b, c))
        x = x * w.view(b, c, 1, 1)
        # 空间注意力
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True)[0]
        s = self.spatial_conv(torch.cat([avg, mx], dim=1))
        return x * s
