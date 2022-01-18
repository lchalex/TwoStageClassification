import os
import sys
from copy import copy, deepcopy
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import model_urls
from torch.utils import model_zoo


def get_model(arch, num_class, pretrained):
    if arch == 'resnet18':
        net = models.resnet18(pretrained=pretrained)
    elif arch == 'resnet34':
        net = models.resnet34(pretrained=pretrained)
    elif arch == 'resnet50':
        net = models.resnet50(pretrained=pretrained)
    elif arch == 'resnet101':
        net = models.resnet101(pretrained=pretrained)
    elif arch == 'resnet152':
        net = models.resnet152(pretrained=pretrained)
    else:
        raise Exception('Invalid arch : {}'.format(arch))
    
    if num_class != 1000:
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, num_class)

    return net


class MultiheadResNet(ResNet):
    def __init__(self, arch, num_cluster, num_classes, pretrained=False):
        self.arch = arch
        self.num_cluster = num_cluster
        self.num_classes = num_classes
        assert len(num_classes) == num_cluster, "Length of num_classes not equals to num_cluster"

        if arch == 'resnet18':
            params = {
                "block": BasicBlock,
                "layers": [2, 2, 2, 2]
            }
            self.out_channels = (64, 128, 256, 512)

        elif arch == 'resnet34':
            params = {
                "block": BasicBlock,
                "layers": [3, 4, 6, 3]
            }
            self.out_channels = (64, 128, 256, 512)

        elif arch == 'resnet50':
            params = {
                "block": Bottleneck,
                "layers": [3, 4, 6, 3]
            }
            self.out_channels = (256, 512, 1024, 2048)

        elif arch == 'resnet101':
            params = {
                "block": Bottleneck,
                "layers": [3, 4, 23, 3]
            }
            self.out_channels = (256, 512, 1024, 2048)

        elif arch == 'resnet152':
            params = {
                "block": Bottleneck,
                "layers": [3, 8, 36, 3]
            }
            self.out_channels = (256, 512, 1024, 2048)

        else:
            raise Exception('Invalid arch : {}'.format(arch))
        
        super().__init__(**params)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls[arch]))

        self._make_multihead()
        
    def _make_multihead(self): # make multi head for layer3, layer4 and fc
        self.multilayer3 = []
        self.multilayer4 = []
        self.multifc = []
        for num_class in self.num_classes:            
            self.multilayer3.append(deepcopy(self.layer3))
            self.multilayer4.append(deepcopy(self.layer4))
            num_ftrs = self.fc.in_features
            self.multifc.append(nn.Linear(num_ftrs, num_class))

        self.multilayer3 = nn.ModuleList(self.multilayer3)
        self.multilayer4 = nn.ModuleList(self.multilayer4)
        self.multifc = nn.ModuleList(self.multifc)
        del self.layer3
        del self.layer4
        del self.fc
    
    def forward(self, x, cluster_ids):
        assert x.shape[0] == cluster_ids.shape[0], "Input and cluster_ids sizes are not match"

        out = []
        indices = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        for i in range(self.num_cluster):
            index = (cluster_ids == i).nonzero(as_tuple=True)[0]
            if index.shape[0] == 0: # No samples of this cluster
                indices.append(None)
                out.append(None)
                continue
                
            indices.append(index)
            s = x[index]
            s = self.multilayer3[i](s)
            s = self.multilayer4[i](s)

            s = self.avgpool(s)
            s = torch.flatten(s, 1)
            s = self.multifc[i](s)
            out.append(s)
                
        return out, indices


class AuxResNet(ResNet):
    def __init__(self, arch, num_cluster, num_class, pretrained=False):
        self.arch = arch
        self.num_cluster = num_cluster
        self.num_class = num_class

        if arch == 'resnet18':
            params = {
                "block": BasicBlock,
                "layers": [2, 2, 2, 2]
            }
            self.out_channels = (64, 128, 256, 512)

        elif arch == 'resnet34':
            params = {
                "block": BasicBlock,
                "layers": [3, 4, 6, 3]
            }
            self.out_channels = (64, 128, 256, 512)

        elif arch == 'resnet50':
            params = {
                "block": Bottleneck,
                "layers": [3, 4, 6, 3]
            }
            self.out_channels = (256, 512, 1024, 2048)

        elif arch == 'resnet101':
            params = {
                "block": Bottleneck,
                "layers": [3, 4, 23, 3]
            }
            self.out_channels = (256, 512, 1024, 2048)

        elif arch == 'resnet152':
            params = {
                "block": Bottleneck,
                "layers": [3, 8, 36, 3]
            }
            self.out_channels = (256, 512, 1024, 2048)

        else:
            raise Exception('Invalid arch : {}'.format(arch))
        
        super().__init__(**params)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls[arch]))

        self._make_head()
    
    def _make_head(self):
        self.aux_fc = nn.Linear(self.out_channels[1], self.num_cluster)
        self.fc = nn.Linear(self.out_channels[3], self.num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        aux = self.avgpool(x)
        aux = torch.flatten(aux, 1)
        aux = self.aux_fc(aux)
        
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
                
        return x, aux
