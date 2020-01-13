import torch
import torchvision
import numpy as np
import tqdm
import matplotlib.pyplot as plt

examples = 1000
trainset = torchvision.datasets.CIFAR10(
    root='data', train=True, download=True, transform=torchvision.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=examples, shuffle=False,
    num_workers=4, pin_memory=True)


def renormalize(noise, inf, sup, distance="linf"):
    if distance == "linf":
        return torch.clamp(noise, inf, sup)
    elif distance == "l2":
        norm = torch.norm(noise.view(noise.size(0),-1),dim=1)
        norm = torch.max(torch.cuda.FloatTensor([1e-12]), norm)
        factor = torch.min(torch.cuda.FloatTensor([1]), sup / norm)
        return noise * factor.view(-1,1,1,1)
    else:
        raise Exception("Wrong distance")

def get_alpha(clean,noisy,noise):
    clean = clean.view(examples,-1)
    noisy = noisy.view(examples,-1)
    difference = noisy-clean
    norm_difference = torch.norm(difference,p=2,dim=1)
    alpha = norm_difference/torch.norm(noise.view(noise.size(0),-1),dim=1)
    return alpha.max().item()

def get_alpha_l2_norm(trainloader,net,iters,l2_norm,best_alpha_zero):

    best_alpha = best_alpha_zero       
    with torch.no_grad():
        for (x, y) in trainloader:
            x = torch.autograd.Variable(x.cuda(),requires_grad=True)
            y = y.cuda()
            output_clean = net(x, normalize=True)
            softmax_clean = torch.nn.Softmax(dim=1)(output_clean)
            for a in range(100):
                noise = torch.cuda.FloatTensor(np.random.uniform(
                        -1, 1, x.shape).astype(
                            np.float32))


                noise = renormalize(noise,l2_norm,l2_norm,distance="l2")

                inputs_test = torch.clamp(x.clone() + noise, 0, 1)
                output = net(inputs_test, normalize=True)
                softmax_noise = torch.nn.Softmax(dim=1)(output)
                alpha = get_alpha(softmax_clean,softmax_noise,inputs_test-x)
                if alpha > best_alpha:
                    best_alpha = alpha
            break
        return best_alpha


networks = [("Vanilla","results/0.0_0.0_1_0_0/ckpt.t7"),
            ("Parseval","results/0.01_0.0_1_0_0/ckpt.t7"),
            ("Regularizer","results/0.01_0.01_2_0_0/ckpt.t7"),
            ("Both","results/0.01_0.01_2_0_0/ckpt.t7") 
           ]
l2_norms = [a/30*1.75 for a in range(1,31)]
dict_results = dict()
for name,folder in networks:
    checkpoint = torch.load(folder)
    net = checkpoint['net']
    net.output_relus = False
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
    torch.backends.cudnn.benchmark = True
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    alphas = list()

    final_alphas = list()
    last_alpha = 0
    for l2_norm in tqdm.tqdm(l2_norms):
        last_alpha = get_alpha_l2_norm(trainloader,net,1,l2_norm,last_alpha)
        alphas.append(last_alpha)
    dict_results[name] = alphas


color = ["black","blue","green","magenta"]
for idx,(name,values) in enumerate(dict_results.items()):
    plt.plot(l2_norms,values,label=name,color=color[idx])
    print(name)
    for a,b in zip(l2_norms,values):
        print(a,b)
plt.legend()
plt.show()


