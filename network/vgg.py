import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import vgg

import config


class VGG_Activations(nn.Module):
    """
    This class allows us to execute only a part of a given VGG network and obtain the activations for the specified
    feature blocks.
    """
    def __init__(self, vgg_network, feature_idx):
        super(VGG_Activations, self).__init__()
        features = list(vgg_network.features)
        self.features = nn.ModuleList(features).eval()
        self.idx_list = feature_idx

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.idx_list:
                results.append(x)

        return results