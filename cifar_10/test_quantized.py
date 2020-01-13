import torch
import pandas as pd
import torchvision
import utils
import tqdm
import numpy as np
import models


def maxWeight(net):
    liste_max = []
    index = 0
    for m in net.modules():
        local_max = 0
        if type(m) in [torch.nn.Conv2d, torch.nn.Linear,
                       models.Parseval_Conv2d]:
            w = m.weight
            v = w.view(-1)
            local_max = torch.max(torch.abs(v)).cpu().data.numpy()
            liste_max.append(local_max)
    return liste_max


def bin_list(list_max):
    local_max = []
    for i in range(len(list_max)):
        n = 0
        while list_max[i] < 1:
            list_max[i] = list_max[i]*2
            n += 1
        local_max.append(n-1)
    return local_max


def quantifier(net, n_bit):
    list_max = maxWeight(net)
    local_max = bin_list(list_max)
    j = 0
    for m in net.modules():
        if type(m) in [torch.nn.Conv2d, torch.nn.Linear,
                       models.Parseval_Conv2d]:
            w = m.weight
            a = w.shape
            v = torch.zeros(a).float()
            v = v + pow(2, n_bit-1+local_max[j])
            v = v.float()
            v = v.cuda()
            w.data.copy_(w.data*v)
            w = w.int()
            w = w.float()
            w.data.copy_(w.data/v)
            m.weight.data.copy_(w.data)
            j += 1


def main():

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    batch_size = 100
    dataframeStarted = False
    dataframe = None

    testset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    folders = utils.read_folder_results()
    with torch.no_grad():
        for name_dict, folder in tqdm.tqdm(folders):

            checkpoint = torch.load(folder + "/ckpt.t7")
            net = checkpoint['net']
            net.output_relus = False
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=[0])
            torch.backends.cudnn.benchmark = True
            criterion = torch.nn.CrossEntropyLoss()
            net.eval()
            n_bit = 5
            total = 0
            top1 = 0
            quantifier(net, n_bit)

            with torch.no_grad():
                for (x, y) in testloader:
                    x = x.cuda()
                    y = y.cuda()
                    total += x.size()[0]

                    outputs_clean = net(x, normalize=True)
                    _, predicted = torch.max(outputs_clean, 1)
                    top1 += float((predicted.cpu() == y.cpu()).sum())

                result_dict = dict(
                    name_dict, folder=folder,
                    accuracy_quantized=top1/total*100,
                    n_bit=n_bit)
                if not dataframeStarted:
                    dataframe = pd.DataFrame(result_dict, index=[0])
                    dataframeStarted = True
                else:
                    dataframe = pd.concat(
                        [dataframe, pd.DataFrame(result_dict, index=[0])])
                dataframe.to_pickle("test_results/quantized.pkl")


if __name__ == "__main__":
    main()
