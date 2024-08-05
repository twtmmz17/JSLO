import torch
import torch.nn as nn
import torch.nn.functional as F


# 带深度可分离卷积的卷积操作
def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


# 注意力机制
class AttenModule(nn.Module):
    def __init__(self, channel):
        super(AttenModule, self).__init__()
        self.hm_P2 = nn.Sequential(conv_dw(channel,128,1),
                                   conv_dw(128, 64, 1),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=1 // 2, bias=True))  # conv_dw(192,128,1), conv_dw(128,64,1),
        self.senet = ChannelAtten(channel,reduction = 16)
        self.hm_P2[-1].bias.data.fill_(-2.19)

    def forward(self, F3):
        pixel = self.hm_P2(F3)
        h = pixel.shape
        pixel_space = torch.softmax(pixel.view(pixel.shape[0],pixel.shape[1],-1),dim=-1).view(h[0],h[1],h[2],-1)   # 保留测试 softmax
        channel = self.senet(F3)
        return channel*pixel_space, pixel


# 通道注意力机制
class ChannelAtten(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAtten, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局池化，1*1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        #//全局平均池化，batch和channel和原来一样保持不变
        y = self.avg_pool(x).view(b, c)
        #//全连接层+池化
        y = self.fc(y).view(b, c, 1, 1)
        #//和原特征图相乘
        return x * y.expand_as(x)