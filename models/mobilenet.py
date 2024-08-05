# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import torch.nn as nn
import math
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['MobileNet', 'mobilenet']
model_urls = {
    'mobilenet_imagenet': 'https://hanlab.mit.edu/projects/amc/external/mobilenet_imagenet.pth.tar',
}


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def get_customized_cfg(cfg, inratios, diff_stride_index):
    """
    Caculate new in_channels based on cfg and inrations, then use diff_stride_index to label it index
    Args:
        cfg: in channel for original prunable layers except the first (3) and the second(32)
        inratios: in ratios for original prunable layers except the first (3) and the second(32)
        diff_stride_index: stride = 2 layers

    Returns:
        stride-labeled cfg
    """
    assert len(inratios) == len(cfg)
    result = []
    for i in range(len(cfg)):
        if i in diff_stride_index:
            result.append((int(inratios[i] * cfg[i] ), 2))
        else:
            result.append(int(inratios[i] * cfg[i]))
    return result


class MobileNet(nn.Module):
    def __init__(self, profile='normal', inratios=None, headconv=256):
        super(MobileNet, self).__init__()
        in_planes = 32
        cfg = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, headconv]
        diff_stride_idx = [1, 3]#, 5, 11]
        if profile == 'normal':
            # original
            in_planes = in_planes
            cfg =[64, (128, 2), 128, (256, 2), 256, 512, 512, 512, 512, 512, 512, 1024, headconv] #[64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
            # customized model from mobilenet
        elif profile == 'customized':
            in_planes = int(inratios[0] * cfg[0])
            # todo : change cfg by ratios to update policy found result

            cfg = get_customized_cfg(cfg, inratios[1:], diff_stride_idx)
        else:
            raise NotImplementedError

        self.conv1 = conv_bn(3, in_planes, stride=2)

        self.features = self._make_layers(in_planes, cfg, conv_dw)


        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        #x = x.mean(3).mean(2)  # global average pooling

        #x = self.classifier(x)
        return x

    def _make_layers(self, in_planes, cfg, layer):
        layers = []
        in_channels = 3
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(layer(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet(pretrained=False, inRatios=None, classes=None, **kwargs):
    """
    Constructs a MobileNet architecture from
    `"Searching for MobileNet" <https://arxiv.org/abs/1905.02244>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = None
    if classes is None:
        if pretrained:
            """
            # for imagenet
            #state_dict = load_state_dict_from_url(model_urls['mobilenet'],progress=progress)
            if 'state_dict' in state_dict:  # a checkpoint but not a state_dict
                sd = state_dict['state_dict']
            sd = {k.replace('module.', ''): v for k, v in sd.items()}

            model.load_state_dict(sd)
            """
            # for voc
            if inRatios == None:
                model = MobileNet(n_class=1000)
                sd = load_state_dict_from_url(model_urls['mobilenet_imagenet'])#
                state_dict = sd['state_dict']
                # for cifar
                model.load_state_dict(state_dict)
            else:
                import torch
                sd = torch.load('./weights/mobilenet_0.028size_export_imagenet.pth.tar')
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
                model = MobileNet(n_class=1000, profile='customized', inratios=inRatios)
                model.load_state_dict(sd)
    else:
        model = MobileNet(n_class=classes)

    return model
