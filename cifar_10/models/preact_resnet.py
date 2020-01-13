import torch
import torch.nn.functional as F
import numpy as np


class Parseval_Conv2d(torch.nn.Conv2d):

    def forward(self, input):
        new_weight = self.weight/np.sqrt(
            2*self.kernel_size[0]*self.kernel_size[1]+1)
        return torch.nn.functional.conv2d(
                input, new_weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups)


class PreActParsevalBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActParsevalBlock, self).__init__()
        self.conv1 = Parseval_Conv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.conv2 = Parseval_Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(in_planes)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = torch.nn.Sequential(
                Parseval_Conv2d(
                    in_planes, planes, kernel_size=1,
                    stride=stride, bias=False)
            )

    def forward(self, x):
        out = x
        out = self.bn1(out)
        relu1 = F.relu(out)
        shortcut = self.shortcut(relu1) if hasattr(self, 'shortcut') else x
        out = self.conv1(relu1)
        out = self.bn2(out)
        relu2 = F.relu(out)
        out = self.conv2(relu2)
        out = 0.5*out + 0.5*shortcut
        return relu1, relu2, out


class PreActBlock(torch.nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(in_planes)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes, planes, kernel_size=1,
                    stride=stride, bias=False)
            )

    def forward(self, x):
        out = x
        out = self.bn1(out)
        relu1 = F.relu(out)
        shortcut = self.shortcut(relu1) if hasattr(self, 'shortcut') else x
        out = self.conv1(relu1)
        out = self.bn2(out)
        relu2 = F.relu(out)
        out = self.conv2(relu2)
        out += shortcut
        return relu1, relu2, out


class PreActResNet(torch.nn.Module):
    def __init__(self, block, num_classes=10, initial_planes=64, classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = initial_planes
        self.classes = classes
        self.output_relus = True

        if block == PreActParsevalBlock:
            self.conv1 = Parseval_Conv2d(
                3, self.in_planes, kernel_size=3,
                stride=1, padding=1, bias=False)
        else:
            self.conv1 = torch.nn.Conv2d(
                3, self.in_planes, kernel_size=3,
                stride=1, padding=1, bias=False)

        self.layer1 = block(self.in_planes, self.in_planes, 1)
        self.layer2 = block(self.in_planes, self.in_planes, 1)

        self.layer3 = block(self.in_planes, self.in_planes*2, 2)
        self.layer4 = block(self.in_planes*2, self.in_planes*2, 1)

        self.layer5 = block(self.in_planes*2, self.in_planes*4, 2)
        self.layer6 = block(self.in_planes*4, self.in_planes*4, 1)

        self.layer7 = block(self.in_planes*4, self.in_planes*8, 2)
        self.layer8 = block(self.in_planes*8, self.in_planes*8, 1)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4,
                       self.layer5, self.layer6, self.layer7, self.layer8]

        self.linear = torch.nn.Linear(self.in_planes*8, self.classes)

    def forward(self, x, normalize=True):
        relus = list()
        if normalize:
            mean = torch.cuda.FloatTensor(
                [0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.cuda.FloatTensor(
                [0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
            x = (x-mean)/std
        out = self.conv1(x)
        for block in self.layers:
            relu1, relu2, out = block(out)
            if self.output_relus:
                relus.append(relu1)
                relus.append(relu2)
        out = F.relu(out)
        if self.output_relus:
            relus.append(out)
        out = global_average_pooling(out)
        out = self.linear(out)
        if self.output_relus:
            return relus, out
        else:
            return out


def PreActResNet18(initial_planes=64, classes=10):
    return PreActResNet(
        PreActBlock, initial_planes=initial_planes, classes=classes)


def PreActResNet18Parseval(initial_planes=64, classes=10):
    return PreActResNet(
        PreActParsevalBlock, initial_planes=initial_planes, classes=classes)


def global_average_pooling(inputs):
    reshaped = inputs.view(inputs.size(0), inputs.size(1), -1)
    pooled = torch.mean(reshaped, 2)
    return pooled.view(pooled.size(0), -1)
