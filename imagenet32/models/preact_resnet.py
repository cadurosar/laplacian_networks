'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(planes)
	
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
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


class PreActResNet(nn.Module):
    def __init__(self, block, num_classes=1000,initial_planes=16):
        super(PreActResNet, self).__init__()
        self.in_planes = initial_planes
        self.classes = num_classes
        self.output_relus = True
        wide=5
        
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = block(self.in_planes, self.in_planes*wide, 1)
        self.layer2 = block(self.in_planes*wide, self.in_planes*wide, 1)
        self.layer3 = block(self.in_planes*wide, self.in_planes*wide, 1)
        self.layer4 = block(self.in_planes*wide, self.in_planes*wide, 1)

        self.layer5 = block(self.in_planes*wide, self.in_planes*wide*2, 2)
        self.layer6 = block(self.in_planes*wide*2, self.in_planes*wide*2, 1)
        self.layer7 = block(self.in_planes*wide*2, self.in_planes*wide*2, 1)
        self.layer8 = block(self.in_planes*wide*2, self.in_planes*wide*2, 1)

        self.layer9 = block(self.in_planes*wide*2, self.in_planes*wide*4, 2)
        self.layer10 = block(self.in_planes*wide*4, self.in_planes*wide*4, 1)
        self.layer11 = block(self.in_planes*wide*4, self.in_planes*wide*4, 1)
        self.layer12 = block(self.in_planes*wide*4, self.in_planes*wide*4, 1)
        
        self.linear = nn.Linear(self.in_planes*wide*4*block.expansion, self.classes)

        
    def forward(self, x, save=False, epoch=0,normalize=True):
        relus = list()
        if normalize:
            mean = torch.cuda.FloatTensor([0.481, 0.457, 0.407]).view(1, 3, 1, 1)
            std = torch.cuda.FloatTensor([0.260, 0.253, 0.268 ]).view(1, 3, 1, 1)
            x2 = x-torch.autograd.Variable(mean)
            x3 = x2/torch.autograd.Variable(std)
            out = self.conv1(x3)
        else:
            out = self.conv1(x)   
        out = self.layer1(out)
        relus.extend(out[:2])   
        out = self.layer2(out[2])   
        relus.extend(out[:2])        

        out = self.layer3(out[2])
        relus.extend(out[:2])

        out = self.layer4(out[2])
        relus.extend(out[:2])        

        out = self.layer5(out[2])
        relus.extend(out[:2])        

        out = self.layer6(out[2])
        relus.extend(out[:2])

        out = self.layer7(out[2])
        relus.extend(out[:2])

        out = self.layer8(out[2])
        relus.extend(out[:2])

        out = self.layer9(out[2])
        relus.extend(out[:2])

        out = self.layer10(out[2])
        relus.extend(out[:2])

        out = self.layer11(out[2])
        relus.extend(out[:2])

        out = self.layer12(out[2])
        relus.extend(out[:2])


        final_out = F.relu(out[2])    
        relus.append(final_out)      
        out = final_out
        out = out.view(out.size(0), out.size(1), -1)
        out = torch.mean(out,2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.output_relus:
            return relus,out
        else:
            return out


def PreActResNet18(initial_planes=16,num_classes=1000):  
    return PreActResNet(PreActBlock,num_classes=num_classes,initial_planes=initial_planes)
