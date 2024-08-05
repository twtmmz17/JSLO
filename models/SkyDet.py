import torch
from . import skynet
import torch.nn as nn
from .model_parts import AttenModule
from .mobilenet import MobileNet

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


class TDBox(nn.Module):

    def __init__(self, class_number, final_kernel, head_conv, model_type='MRF', isBBStudent=False, inratios=None):
        super(TDBox, self).__init__()
        # head_conv: 过渡卷积操作，卷积核数量

        self.isBBStudent = isBBStudent

        if self.isBBStudent:
            #profile='normal', inratios=None, headconv=256
            self.base_network = MobileNet(headconv=head_conv)
        else:
            self.base_network = MobileNet(profile='customized', inratios=inratios, headconv=head_conv)

        self.roughfeatures = conv_dw(head_conv, head_conv, 1)  #32 to 16#64 to 32# 粗糙的特征
        self.AttentionModify = AttenModule(head_conv)  # 注意力模块，完成像素的筛选和通道筛选
        self.neckfeatures = nn.Sequential(
            nn.Conv2d(head_conv, head_conv, 1, 1, bias=False),
            conv_dw(head_conv, head_conv, 1)
        )
        self.hm_P2 = nn.Sequential(conv_dw(head_conv, head_conv, 1),
                                   conv_dw(head_conv, head_conv, 1),
                                   nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, class_number, kernel_size=final_kernel, stride=1,
                                             padding=final_kernel // 2,
                                             bias=True))  # conv_dw(head_conv, head_conv, 1)
        self.wh_P2 = nn.Sequential(conv_dw(head_conv, head_conv, 1),
                                   conv_dw(head_conv, head_conv, 1),
                                   nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, 8, kernel_size=3, padding=1, bias=True))  # 10
        self.reg_P2 = nn.Sequential(conv_dw(head_conv, head_conv, 1),
                                    conv_dw(head_conv, head_conv, 1),
                                    nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(head_conv, 2, kernel_size=final_kernel, stride=1,
                                              padding=final_kernel // 2, bias=True))

        # 从初始化bias
        self.fill_fc_weights(self.wh_P2)
        self.fill_fc_weights(self.reg_P2)
        self.hm_P2[-1].bias.data.fill_(-2.19)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        roughF = self.roughfeatures(x)
        finefeatures, pixels_mask = self.AttentionModify(roughF)
        features = self.neckfeatures(finefeatures)
        dec_dict = {}
        # 最浅层的检测结果
        dec_dict['wh_P2'] = self.wh_P2(features)
        dec_dict['reg_P2'] = self.reg_P2(features)
        dec_dict['pixel_mask_P2'] = torch.sigmoid(pixels_mask)
        dec_dict['hm_P2'] = torch.sigmoid(self.hm_P2(features))
        return dec_dict