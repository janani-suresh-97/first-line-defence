'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from logzero import logger
import torch.nn.functional as F

cfg = {
    # 'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [256,'M', 64,  128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512,],
    # 'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out  = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        i = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=5, stride=1, padding=1)]
            elif x == 'N':
                layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
            elif i ==0:
                layers += [nn.Conv2d(in_channels, x, kernel_size=15, stride=1,padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
            i+=1
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
