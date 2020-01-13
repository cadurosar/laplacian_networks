'''Train CIFAR10 and CIFAR100 with PyTorch.'''
from __future__ import print_function

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable
from gsp import force_smooth_network
from pathlib import Path
import numpy as np
import numpy
from imagenet32x32 import Imagenet32x32

parser = argparse.ArgumentParser(description='Imagenet32x32 Training')
parser.add_argument('-m', default=1, type=int, help='laplacian power')
parser.add_argument('-k', default=0, type=int, help='number of neighbors')
parser.add_argument('--beta', default=0., type=float, help='parseval beta parameter')
parser.add_argument('--gamma', default=0., type=float, help='laplacian weight parameter')
parser.add_argument('--seed', default=0, type=int, help='seed')
args = parser.parse_args()



use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   
    transforms.RandomHorizontalFlip(),      
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = Imagenet32x32(root='data', train=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2,pin_memory=False)

testset = Imagenet32x32(root='data', train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2,pin_memory=False)

if args.beta > 0:
    net = PreActResNet18Parseval()
else:
    net = PreActResNet18()
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
path = "results/{}_{}_{}_{}_{}_{}/".format(args.adversary,args.beta,args.gamma,args.m,args.k,args.seed)
try:
    os.makedirs(path)        
except:
    pass

params = net.parameters()
parseval_parameters = list()
for param in params:
    if len(param.size()) > 1:
        parseval_parameters.append(param)

def do_parseval(parseval_parameters):
    for W in parseval_parameters:
        Ws = W.view(W.size(0),-1)
        W_partial = Ws.data.clone()
        W_partial = (1+args.beta)*W_partial - args.beta*(torch.mm(torch.mm(W_partial,torch.t(W_partial)),W_partial))
        new = W_partial
        new = new.view(W.size())
        W.data.copy_(new)

dataframeStarted = False
dataframeStarted2 = False
dataframe,dataframe2 = None,None
# Training
def train(epoch, optimizer):
    print('\nEpoch: %d' % epoch)
    global dataframeStarted,dataframe
    net.train()
    train_loss2 = 0
    train_loss1 = 0
    train_loss = 0
    correct = 0.
    total = 0.
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), (targets-1).cuda()
        optimizer.zero_grad()
        if args.adversary:
            targets = Variable(targets)
            inputs1, targets1 = Variable(inputs,requires_grad=True), targets
            relus, _ = net(inputs1,normalize=True)
            inputs2 = create_adversary(inputs1,targets1,net,criterion)
            _, outputs = net(inputs2,normalize=True)
            loss1 = criterion(outputs,targets)
        else:
            inputs, targets = Variable(inputs), Variable(targets)
            relus, outputs = net(inputs,normalize=True)
            loss1 = criterion(outputs, targets) 
        if args.gamma > 0:
            if args.k > 0:
                loss2 = force_smooth_network(relus,targets,m=args.m,k=args.k)
            else:
                loss2 = force_smooth_network(relus,targets,m=args.m)                
            value = 1/args.gamma
            loss = loss1 + loss2/(value**args.m)
            train_loss2 += loss2.data.item()
        else:
            loss2 = 0
            train_loss2 += loss2
            loss = loss1
        loss.backward()
        optimizer.step()
        if args.beta > 0:
            do_parseval(parseval_parameters)
        train_loss += loss.item()
        train_loss1 += loss1.item()
        _, predicted = torch.max(outputs.data, 1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets.data).cpu().sum())

        progress_bar(batch_idx, len(trainloader), 'CC: %.3f | SM: %.3f | L: %.3f | A: %.3f%% (%d/%d)'
            % (train_loss1/(batch_idx+1),train_loss2/(batch_idx+1),train_loss/(batch_idx+1),
            100.*correct/total, correct, total))
    f = open(path + 'score_training.txt','a')
    f.write(str(1.*correct/total))
    f.write('\n')
    f.close()
    

def test(epoch):
    print('\nTest... epoch %d' % epoch)
    global dataframe
    net.eval()
    test_loss = 0
    test_loss1 = 0
    test_loss2 = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), (targets-1).cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        relus, outputs = net(inputs,normalize=True)
        loss1 = criterion(outputs, targets) 
        loss2 = 0
        test_loss2 += loss2
        loss = loss1
        test_loss += loss.item()
        test_loss1 += loss1.item()
        _, predicted = torch.max(outputs.data, 1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets.data).cpu().sum())

        progress_bar(batch_idx, len(testloader), 'CC: %.3f | SM: %.3f | L: %.3f | A: %.3f%% (%d/%d)'
            % (test_loss1/(batch_idx+1),test_loss2/(batch_idx+1),test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    f = open(path + 'score.txt','a')
    f.write(str(1.*correct/total))
    f.write('\n')
    f.close()
        
def save(epoch):
    net.forward(examples, True, epoch)

def save_model():
    state = {
        'net': net.module if use_cuda else net,
    }
    torch.save(state, path+'/ckpt.t7')
    
f = open(path + 'score.txt','w')
f.write("0.1\n")
f.close()
f = open(path + 'score_training.txt','w')
f.write("0.1\n")
f.close()


for period in range(3):
    if period == 0:
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    elif period == 1:
        optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(net.parameters(), lr=0.0004, momentum=0.9, weight_decay=5e-4)
                
    for epoch in range(10 * period, 10 * (period + 1)):
        train(epoch, optimizer)
        test(epoch)
        save_model()
#save(epoch)
