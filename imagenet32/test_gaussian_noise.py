import torch
import pandas as pd
import torchvision
import utils
import tqdm
import numpy as np
from imagenet32x32 import Imagenet32x32


def main():

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    batch_size = 200
    dataframeStarted = False
    dataframe = None

    testset = Imagenet32x32(root='data', train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,pin_memory=False)

    folders = utils.read_folder_results()

    for sample in tqdm.tqdm(range(40)):
        noise_global = torch.cuda.FloatTensor(
            np.random.normal(scale=0.5, size=(1, 3, 32, 32)))

        for name_dict, folder in folders:
            print(folder)
            checkpoint = torch.load(folder + "/ckpt.t7")
            net = checkpoint['net']
            net.output_relus = False
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=[0])
            torch.backends.cudnn.benchmark = True
            criterion = torch.nn.CrossEntropyLoss()
            net.eval()
            epsilon = 0.022
            total = 0
            top1 = 0
            top1_under_attack = 0
            snr = list()
            noise = epsilon * noise_global
            with torch.no_grad():
                for (x, y) in testloader:
                    x = x.cuda()
                    y = y.cuda()-1
                    total += x.size()[0]

                    noisy_x = torch.clamp(x + noise, 0, 1)
                    shape = (x.size(0), -1)
                    division = torch.norm(
                        x.view(shape), dim=1)/torch.norm(
                            noise.view(1, -1), dim=1)

                    _snr = 20*np.log10(division.cpu().numpy())
                    snr.extend(_snr.tolist())

                    outputs_clean = net(x, normalize=True)
                    _, predicted = torch.max(outputs_clean, 1)
                    top1 += float((predicted.cpu() == y.cpu()).sum())

                    outputs_noisy = net(noisy_x, normalize=True)
                    _, predicted = torch.max(outputs_noisy, 1)
                    top1_under_attack += float(
                        (predicted.cpu() == y.cpu()).sum())
                result_dict = dict(
                    name_dict, sample=sample, folder=folder,
                    accuracy=top1/total*100,
                    accuracy_gauss=top1_under_attack/total*100,
                    epsilon=epsilon, mean_snr=np.array(snr).mean())
                if not dataframeStarted:
                    dataframe = pd.DataFrame(result_dict, index=[0])
                    dataframeStarted = True
                else:
                    dataframe = pd.concat(
                        [dataframe, pd.DataFrame(result_dict, index=[0])])
                dataframe.to_pickle("test_results/gaussian_noise.pkl")


if __name__ == "__main__":
    main()
