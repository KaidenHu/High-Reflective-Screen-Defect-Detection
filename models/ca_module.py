import torch
import torch.nn as nn
import torch.nn.functional as F

class CA(nn.Module):
    """Coordinate Attention module"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        hidden = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(hidden, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 垂直方向池化
        x_h = self.pool_h(x)  # (n, c, h, 1)
        # 水平方向池化
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (n, c, w, 1)
        
        # 拼接并降维
        y = torch.cat([x_h, x_w], dim=2)  # (n, c, h+w, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # 分割
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # 生成注意力权重
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        out = identity * a_h * a_w
        return out
