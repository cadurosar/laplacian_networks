import os
import argparse
import random
import torch
import torchvision

from models import *
from utils import progress_bar
import gsp


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument(
        '-m', default=1, type=int, help='laplacian power')
    parser.add_argument(
        '-k', default=0, type=int, help='number of neighbors')
    parser.add_argument(
        '--beta', default=0., type=float, help='parseval beta parameter')
    parser.add_argument(
        '--gamma', default=0., type=float, help='laplacian weight parameter')
    parser.add_argument(
        '--seed', default=0, type=int, help='seed')
    args = parser.parse_args()

    uses_parseval = args.beta > 0
    uses_regularizer = args.gamma > 0

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True, num_workers=4, pin_memory=True)

    if uses_parseval:
        net = PreActResNet18Parseval()
    else:
        net = PreActResNet18()
    net.cuda()
    net.output_relus = uses_regularizer
    net = torch.nn.DataParallel(net, device_ids=[0])
    torch.backends.cudnn.benchmark = True

    criterion = torch.nn.CrossEntropyLoss()
    path = "results/{}_{}_{}_{}_{}/".format(
        args.beta, args.gamma, args.m, args.k, args.seed)
    try:
        os.makedirs(path)
    except:
        pass

    if uses_parseval:
        params = net.parameters()
        parseval_parameters = list()
        for param in params:
            if len(param.size()) > 1:
                parseval_parameters.append(param)

        def do_parseval(parseval_parameters):
            for W in parseval_parameters:
                Ws = W.view(W.size(0), -1)
                W_partial = Ws.data.clone()
                a = (1+args.beta)*W_partial
                b = args.beta*(
                    torch.mm(
                            torch.mm(W_partial, torch.t(W_partial)),
                            W_partial)
                    )
                W_partial = a - b
                new = W_partial
                new = new.view(W.size())
                W.data.copy_(new)

    # Training
    def train(epoch, optimizer):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss2 = 0
        train_loss1 = 0
        train_loss = 0
        correct = 0.
        total = 0.

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            if uses_regularizer:
                relus, outputs = net(inputs, normalize=True)
            else:
                outputs = net(inputs, normalize=True)

            loss = criterion(outputs, targets)
            if uses_regularizer:
                _lambda = (args.gamma**args.m)
                loss2 = _lambda * gsp.force_smooth_network(
                    relus, targets, m=args.m)
                loss = loss + loss2
                train_loss2 += loss2.data.item()
            loss.backward()
            optimizer.step()
            if uses_parseval:
                do_parseval(parseval_parameters)
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets.data).cpu().sum())

            progress_bar(
                         batch_idx, len(trainloader),
                         'L: %.3f | SM: %.3f | A: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1),
                            train_loss2/(batch_idx+1),
                            100.*correct/total, correct, total)
                        )

        f = open(path + 'score_training.txt', 'a')
        f.write(str(1.*correct/total))
        f.write('\n')
        f.close()

    def save_model():
        state = dict(net=net.module)
        torch.save(state, path+'/ckpt.t7')

    f = open(path + 'score_training.txt', 'w')
    f.write("0.1\n")
    f.close()

    for period in range(2):
        if period == 0:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        for epoch in range(50 * period, 50 * (period + 1)):
            train(epoch, optimizer)
            save_model()

if __name__ == "__main__":
    main()
