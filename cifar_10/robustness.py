# -*- coding: utf-8 -*-

import argparse
import os
import time
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import pandas as pd

torch.nn.Module.dump_patches = True
parser = argparse.ArgumentParser(description='Evaluates robustness of various nets on CIFAR',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Architecture
parser.add_argument('--model-name', '-m', type=str,
                    choices=['vanilla', 'parseval', 'ours', 'parseval_ours'],default="vanilla")
# Acceleration
args = parser.parse_args()
print(args)

# /////////////// Model Setup ///////////////
torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)
args.test_bs = 100

if args.model_name == 'vanilla':
    folder = "results/0.0_0.0_1_0_"
elif args.model_name == 'parseval':
    folder = "results/0.0_0.0_1_0_"
elif args.model_name == 'ours':
    folder = "results/0.0_0.0_1_0_"
elif args.model_name == 'parseval_ours':
    folder = "results/0.0_0.0_1_0_"
args.test_bs = 100

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

args.prefetch = 4

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
clean_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch,pin_memory=True)

dataframeStarted = None
dataframe = None

seed_range = 10

for seed in range(seed_range):
    checkpoint = torch.load(folder + "{}/ckpt.t7".format(seed))
        
    net = checkpoint['net']
    net.output_relus = True

    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
    cudnn.benchmark = True


    net.eval()
    cudnn.benchmark = True  # fire on all cylinders

    print('Model Loaded')

    # /////////////// Data Loader ///////////////


    correct = 0
    for batch_idx, (data, target) in enumerate(clean_loader):
         data = data.cuda()

         _, output = net(data,normalize=True)

         pred = output.max(1)[1]
         correct += pred.eq(target.cuda()).sum()

    clean_error = 1 - correct.float() / len(clean_loader.dataset)
    print('Clean dataset error (%): {:.2f}'.format(100 * clean_error))


    # /////////////// Further Setup ///////////////

    def auc(errs):  # area under the distortion-error curve
        area = 0
        for i in range(1, len(errs)):
            area += (errs[i] + errs[i - 1]) / 2
        area /= len(errs) - 1
        return area


    def show_performance(distortion_name):
        with torch.no_grad():
            errs = []
            labels = np.load("data/labels.npy")
            dataset = np.load("data/{}.npy".format(distortion_name))
            dataset = np.transpose(dataset,[0,3,1,2])

            for severity in range(0, 5):
                torch_data = torch.FloatTensor(dataset[10000*severity:10000*(severity+1)])
                torch_labels = torch.LongTensor(labels[10000*severity:10000*(severity+1)])
                test = torch.utils.data.TensorDataset(torch_data, torch_labels)
                distorted_dataset_loader = torch.utils.data.DataLoader(test, batch_size=args.test_bs, shuffle=False,num_workers=args.prefetch,pin_memory=True)



                correct = 0
                for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
                    data = data.cuda()/255

                    _, output = net(data, normalize=True)

                    pred = output.max(1)[1]
                    correct += pred.eq(target.cuda()).sum()
                percentage = correct.float() / 10000
                errs.append( (1 - percentage ).item())

            print('\n=Average', tuple(errs))
            return errs


    # /////////////// End Further Setup ///////////////


    # /////////////// Display Results ///////////////
    import collections

    distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    ]

    error_rates = []
    result_dict = dict()
    for distortion_name in distortions:
        rate = show_performance(distortion_name)
        error_rates.append(np.mean(rate))
        print('Distortion: {:15s}  | Error (%): {:.2f}'.format(distortion_name, 100 * np.mean(rate)))
        for a,b in enumerate(rate):
            result_dict["{}_{}".format(distortion_name,a)] = b
    if not dataframeStarted:
        dataframe = pd.DataFrame(result_dict,index=[seed])
        dataframeStarted = True
    else:
        dataframe = pd.concat([dataframe,pd.DataFrame(result_dict,index=[seed])])
    dataframe.to_csv("{}.csv".format(args.model_name))

    print('Mean Error (%): {:.2f}'.format(100 * np.mean(error_rates)))