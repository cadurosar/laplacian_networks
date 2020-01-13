import torch
import pandas as pd
import torchvision
import os
import tqdm
import utils
import numpy as np


def compute_fgsm(net, loader, before_norm=True, wanted_snr=33):
    total = 0
    top1 = 0
    top1_under_attack = 0
    snr = list()
    _diff_limit = list()
    criterion = torch.nn.CrossEntropyLoss()

    for (x, y) in loader:
        total += x.size(0)
        x = torch.autograd.Variable(x.cuda(), requires_grad=True)
        y = y.cuda()

        output = net(x, normalize=before_norm)
        net.zero_grad()
        loss = criterion(output, y)
        loss.backward()
        shape = (x.size(0), -1)

        x_norm = torch.norm(x.view(shape), dim=1)
        div_term = 10.0**(wanted_snr/20.0)
        mul_term = (3072.0)**-0.5
        test_term = mul_term/div_term

        diff_limit = x_norm*test_term
        _diff_limit.extend(diff_limit.tolist())
        noise = diff_limit.view(-1, 1, 1, 1) * torch.sign(x.grad)

        noisy_x = x+noise

        if before_norm:
            noisy_x = torch.clamp(noisy_x, 0, 1)

        division = (x_norm/torch.norm(noise.view(shape), dim=1))
        _snr = 20*np.log10(division.detach().cpu().numpy())
        snr.extend(_snr.tolist())

        _, predicted = torch.max(output, 1)
        top1 += float((predicted.cpu() == y.cpu()).sum())

        noisy_output = net(noisy_x, normalize=before_norm)
        _, predicted = torch.max(noisy_output, 1)
        top1_under_attack += float((predicted.cpu() == y.cpu()).sum())
    return _diff_limit, top1/total*100, top1_under_attack/total*100


def main():

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    transform_norm = torchvision.transforms.Compose([

        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = 100
    dataframeStarted = False
    dataframe = None

    testset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform)
    norm_testset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_norm)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    norm_testloader = torch.utils.data.DataLoader(
        norm_testset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    folders = utils.read_folder_results()
    wanted_snr = 33
    for name_dict, folder in tqdm.tqdm(folders):
        checkpoint = torch.load(folder + "/ckpt.t7")
        net = checkpoint['net']
        net.output_relus = False
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        torch.backends.cudnn.benchmark = True
        net.eval()

        epsilons, top1, top1_before_norm = compute_fgsm(
            net, testloader, before_norm=True, wanted_snr=wanted_snr)
        _, top1x, top1_after_norm = compute_fgsm(
            net, norm_testloader, before_norm=False, wanted_snr=wanted_snr)
        assert top1 == top1x

        result_dict = dict(
            name_dict, folder=folder,
            accuracy=top1,
            accuracy_before_norm=top1_before_norm,
            accuracy_after_norm=top1_after_norm,
            snr=wanted_snr, mean_epsilon=np.array(epsilons).mean())
        if dataframe is None:
            dataframe = pd.DataFrame(result_dict, index=[0])
        else:
            dataframe = pd.concat(
                [dataframe, pd.DataFrame(result_dict, index=[0])]
                )

        dataframe.to_pickle("test_results/fgsm.pkl")

if __name__ == "__main__":
    main()
