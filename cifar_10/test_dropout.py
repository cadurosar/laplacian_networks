import torch
import pandas as pd
import torchvision
import tqdm
import models
import utils


def main():

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    batch_size = 200
    dataframeStarted = False
    dataframe = None

    testset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    folders = utils.read_folder_results()

    class MyDropout(torch.nn.modules.dropout._DropoutNd):

        def forward(self, input):
            return torch.nn.functional.dropout(
                input, self.p, True, self.inplace)

    def forward2(net, x, dropout=0.25, normalize=True):

        relus = list()
        if normalize:
            mean = torch.cuda.FloatTensor(
                [0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.cuda.FloatTensor(
                [0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
            x = (x-mean)/std
        out = net.conv1(x)
        for block in net.layers:
            relu1, relu2, out = block(out)
            relus.append(relu1)
            relus.append(relu2)
            out = MyDropout(dropout)(out)
        out = torch.nn.functional.relu(out)
        out = models.global_average_pooling(out)
        out = net.linear(out)
        return out

    for a in tqdm.tqdm(range(40)):
        for name_dict, folder in folders:

            checkpoint = torch.load(folder + "/ckpt.t7")
            net = checkpoint['net']
            net.cuda()
            torch.backends.cudnn.benchmark = True
            criterion = torch.nn.CrossEntropyLoss()
            net.eval()
            total = 0
            top1 = 0
            top1_under_attack = 0
            snr = list()
            _diff_limit = list()
            for dropout in [0.25, 0.4]:
                with torch.no_grad():
                    for (x, y) in testloader:
                        total += x.size()[0]
                        x, y = x.cuda(), y.cuda()
                        f2 = forward2(net, x, normalize=True, dropout=dropout)
                        _, predicted = torch.max(f2, 1)
                        top1 += float((predicted.cpu() == y.cpu()).sum())
                    result_dict = dict(name_dict, folder=folder,
                                       accuracy_dropout=top1/total*100,
                                       sample=a, dropout=dropout)
                    if not dataframeStarted:
                        dataframe = pd.DataFrame(result_dict, index=[0])
                        dataframeStarted = True
                    else:
                        dataframe = pd.concat(
                            [dataframe, pd.DataFrame(result_dict, index=[0])])
                    dataframe.to_pickle("test_results/dropout.pkl")

if __name__ == "__main__":
    main()
