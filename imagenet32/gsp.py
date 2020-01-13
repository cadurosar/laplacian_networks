import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

def force_smooth_network(relus,targets,m=1,classes=1000,k=None,distance="cosine"):
    targets = targets.clone()
    laplacians = list()
    for i, relu in enumerate(relus):
        result = laplacian(relu,targets,m=m,classes=classes,k=k,distance=distance)
        laplacians.append(result)
    loss = 0
    for i in range(len(laplacians)):
        if i == 0:
            continue
        else:
            result = torch.abs(laplacians[i]-laplacians[i-1])
            loss += result
    return loss/((len(laplacians)-1))

def laplacian(x_values,target,distance="cosine",m=1,classes=1000,k=None,extract=False,reg_l2=False):
    x_values = x_values.clone()
    target = target.clone()

    n_examples = x_values.size(0)
    x_values = x_values.view(n_examples,-1)


    y_true = torch.cuda.FloatTensor(n_examples,classes)
    y_true.zero_()
    y_true.scatter_(1, target.data.view(-1,1), 1)
    y_true = Variable(y_true)
    
    transposed_y_true = torch.t(y_true)
    if k is None:
        neighbours = n_examples
    else:
        neighbours = k        

    if distance == "cosine":
        normalized =  F.normalize(x_values, p=2, dim=1)
        W_tf = torch.mm(normalized,torch.t(normalized))
    elif distance == "l1":    
        normalized =  x_values#F.normalize(y_pred, p=1, dim=1)
        normalized2 =  x_values#F.normalize(y_pred, p=1, dim=1)
        y_pred_1 = normalized.unsqueeze(1)    
        y_pred_2 = normalized2.unsqueeze(0)

        W_tf = torch.abs(y_pred_1 - y_pred_2)
        W_tf_2 = torch.nn.modules.distance.PairwiseDistance()(normalized,normalized)
        print(W_tf_2.shape)
        print(np.allclose(W_tf.detach().numpy(),W_tf_2.detach().numpy()))
        raise Exception("ESSS")
        W_tf = torch.mean(W_tf,dim=2)#/normalized.size()[1]
        W_tf = torch.exp(-W_tf)
    elif distance == "l2":    
        normalized =  x_values#F.normalize(y_pred, p=2, dim=1)
        normalized2 =  x_values#F.normalize(y_pred, p=2, dim=1)
        y_pred_1 = normalized.unsqueeze(1)    
        y_pred_2 = normalized2.unsqueeze(0)
        W_tf = (y_pred_1 - y_pred_2)
        W_tf *= W_tf
        W_tf = torch.mean(W_tf,dim=2)
        W_tf = torch.sqrt(W_tf)
        W_tf = torch.exp(-W_tf)
    
    
    if neighbours != n_examples:
        y, ind = torch.sort(W_tf, 1)
        A = torch.zeros(*y.size()).cuda()
        k_biggest = ind[:,-neighbours:].data
        for index1,value in enumerate(k_biggest):
            A_line = A[index1]
            A_line[value] = 1
        A_final = Variable(torch.min(torch.ones(*y.size()).cuda(),A+torch.t(A)))
        new_W_tf = W_tf*A_final
    else:
        new_W_tf = W_tf
    
    d_tf = torch.sum(new_W_tf,1)
    d_tf = torch.diag(d_tf)
    laplacian_tf = (d_tf - new_W_tf)
    laplacian_after_m = laplacian_tf
    for _ in range(1,m):
        laplacian_after_m = torch.mm(laplacian_after_m,laplacian_tf)
    if reg_l2 and m > 1:
        clone = torch.abs(laplacian_after_m.clone())
        mask = torch.diag(torch.ones_like(clone[0]))
        clone *= (1-mask)
        max_val = torch.max(clone.view(-1))
        laplacian_after_m /= max_val 
    if extract:
        return laplacian_after_m
    else:
        final_laplacian_tf = torch.mm(transposed_y_true, laplacian_after_m)
        final_laplacian_tf = torch.mm(final_laplacian_tf,y_true)
        final_laplacian_tf = torch.trace(final_laplacian_tf)
        return final_laplacian_tf
