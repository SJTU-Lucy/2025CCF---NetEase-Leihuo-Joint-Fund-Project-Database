import torch
import torch.nn as nn
from model.networks import resnet50, conv1x1


class RetargetNet(nn.Module):
    def __init__(self, use_last_fc=False):
        super(RetargetNet, self).__init__()
        self.use_last_fc = use_last_fc
        self.last_dim = 2048
        self.rig_dim = 174
        self.backbone = resnet50(use_last_fc=use_last_fc)
        self.final_layer = conv1x1(self.last_dim, self.rig_dim, bias=True)
        nn.init.constant_(self.final_layer.weight, 0.)
        nn.init.constant_(self.final_layer.bias, 0.)

    def forward(self, x):
        x = self.backbone(x)
        x = self.final_layer(x)
        x = torch.flatten(x, 1)

        return x