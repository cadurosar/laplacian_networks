import torch
import pandas as pd
import torchvision
import tqdm
import models
import utils
from imagenet32x32 import Imagenet32x32

def main():

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    testset = Imagenet32x32(root='data', train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2,pin_memory=False)

    batch_size = 200
    dataframeStarted = False
    dataframe = None


    folders = utils.read_folder_results()

    class MyDropout(torch.nn.modules.dropout._DropoutNd):

        def forward(self, input):
            return torch.nn.functional.dropout(
                input, self.p, True, self.inplace)

    def forward2(net, x, dropout=0.25, normalize=True):

        relus = list()
        if normalize:
            mean = torch.cuda.FloatTensor([0.481, 0.457, 0.407]).view(1, 3, 1, 1)
            std = torch.cuda.FloatTensor([0.260, 0.253, 0.268 ]).view(1, 3, 1, 1)
            x2 = x-mean
            x = x2/std
        out = net.conv1(x)
        relu1, relu2, out = net.layer1(out)
        out = MyDropout(dropout)(out)
        relu1, relu2, out = net.layer2(out)
        out = MyDropout(dropout)(out)
        relu1, relu2, out = net.layer3(out)
        out = MyDropout(dropout)(out)
        relu1, relu2, out = net.layer4(out)
        out = MyDropout(dropout)(out)
        relu1, relu2, out = net.layer5(out)
        out = MyDropout(dropout)(out)
        relu1, relu2, out = net.layer6(out)
        out = MyDropout(dropout)(out)
        relu1, relu2, out = net.layer7(out)
        out = MyDropout(dropout)(out)
        relu1, relu2, out = net.layer8(out)
        out = MyDropout(dropout)(out)
        relu1, relu2, out = net.layer9(out)
        out = MyDropout(dropout)(out)
        relu1, relu2, out = net.layer10(out)
        out = MyDropout(dropout)(out)
        relu1, relu2, out = net.layer11(out)
        out = MyDropout(dropout)(out)
        relu1, relu2, out = net.layer12(out)
        out = MyDropout(dropout)(out)

        out = torch.nn.functional.relu(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = torch.mean(out,2)
        out = out.view(out.size(0), -1)
        out = net.linear(out)
        return out

    for a in tqdm.tqdm(range(2)):
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
            for dropout in [0.15]:
                with torch.no_grad():
                    for (x, y) in testloader:
                        total += x.size()[0]
                        x, y = x.cuda(), y.cuda()-1
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
