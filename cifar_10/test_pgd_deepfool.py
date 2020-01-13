import torch
import pandas as pd
import torchvision
import foolbox
import numpy as np
import tqdm
import utils


def main():

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    dataframeStarted = False
    dataframe = None

    testset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_test)

    folders = utils.read_folder_results()

    PGD = foolbox.attacks.ProjectedGradientDescentAttack
    DeepFool = foolbox.attacks.DeepFoolAttack

    for name_dict, folder in tqdm.tqdm(folders):
        checkpoint = torch.load(folder + "/ckpt.t7")
        net = checkpoint['net']
        net.output_relus = False
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        torch.backends.cudnn.benchmark = True
        net.eval()
        model = foolbox.models.PyTorchModel(
            model=net, bounds=(0, 1), num_classes=10)
        for Attack in [PGD, DeepFool]:
            if Attack == DeepFool:
                testloader = torch.utils.data.DataLoader(
                    testset, batch_size=10, shuffle=False, num_workers=2)
            else:
                testloader = torch.utils.data.DataLoader(
                    testset, batch_size=1, shuffle=False, num_workers=2)

            criterion = foolbox.criteria.Misclassification()
            attack = Attack(model, criterion)

            total = 0
            acc = 0
            acc_adversarial = 0
            l2_distances = []
            for x, y in testloader:
                x = x[0].numpy()
                y = y[0].numpy()
                total += 1

                pred_label = model.batch_predictions(x[None])
                pred_label = np.argmax(np.squeeze(pred_label))

                if y == pred_label:
                    acc += 1
                else:
                    l2_distances.append(0)
                    continue
                if Attack == PGD:
                    adv = foolbox.Adversarial(
                        model, criterion, x, y,
                        distance=foolbox.distances.Linfinity)
                    adversarial = attack(
                        adv, unpack=False,
                        binary_search=False, random_start=False,
                        epsilon=0.01, stepsize=0.002, iterations=20)
                else:
                    adversarial = attack(x, label=y, unpack=False)

                if adversarial.image is None:
                    acc_adversarial += 1
                else:
                    l2_distance = np.sqrt(
                        np.sum(np.square(adversarial.image - x)))
                    l2_distances.append(l2_distance/(3*32*32))
            result_dict = dict(
                name_dict, attack=str(Attack), folder=folder,
                mean_l2=np.mean(np.array(l2_distances)),
                accuracy=acc/total,
                accuracy_adversarial=acc_adversarial/total,
            )
            if not dataframeStarted:
                dataframe = pd.DataFrame(result_dict, index=[0])
                dataframeStarted = True
            else:
                dataframe = pd.concat(
                    [dataframe, pd.DataFrame(result_dict, index=[0])])
            dataframe.to_pickle("test_results/pgd_deepfool.pkl")


if __name__ == "__main__":
    main()
