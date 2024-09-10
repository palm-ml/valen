import torch 
import torch.nn.functional as F
import numpy as np

def partial_loss(output1, target, true, eps=1e-12):
    output = F.softmax(output1, dim=1)
    l = target * torch.log(output+eps)
    loss = (-torch.sum(l)) / l.size(0)

    revisedY = target.clone()
    revisedY[revisedY > 0]  = 1
    revisedY = revisedY * (output.clone().detach())
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)
    new_target = revisedY

    return loss, new_target

def revised_target(output, target):
    revisedY = target.clone()
    revisedY[revisedY > 0]  = 1
    revisedY = revisedY * (output.clone().detach())
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)
    new_target = revisedY

    return new_target

def threshold_loss(output, target, threshold):
    output1 = F.sigmoid(output)
    output2 = F.softmax(output, dim=1)
    revisedY1 = target.clone()
    label = target.clone()
    label[label>0] = 1
    values, indices = (output1.clone().detach()*label).topk(k=2, dim=1)
    delta_values = values[:, 0] - values[:, 1]
    corrected_labels = torch.zeros_like(output1)
    row_indexes = [i for i in range(0, output1.size(0))]
    col_indexes = indices[:,0]
    corrected_labels[row_indexes, col_indexes] = 1.0
    # 找出小于threshold的样本，将其标签修正
    revisedY1[delta_values>threshold] = corrected_labels[delta_values>threshold] + 0.0
    l = revisedY1 * torch.log(output2)
    loss = (-torch.sum(l)) / l.size(0)
    # loss = F.binary_cross_entropy(output1, revisedY1)
    # l = revisedY1 * torch.log(output1) + label * (1 - revisedY1) * torch.log(1 - output1)
    # loss = (-torch.sum(l)) / l.size(0)

    # output2 = F.softmax(output, dim=1)
    # revisedY = target.clone()
    # revisedY[revisedY > 0]  = 1
    # revisedY = revisedY * output2
    # revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)
    revisedY = target.clone()
    revisedY[revisedY > 0]  = 1
    revisedY = revisedY * (output2.clone().detach())
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)
    # revisedY = F.sigmoid(revisedY)
    # revisedY = revisedY / torch.max(revisedY, dim=1, keepdim=True)
    # 添加修正后的标签
    row_indexes, col_indexes = torch.where(revisedY1 == 1)
    corrected_num = row_indexes.size(0)
    # revisedY[row_indexes] = revisedY[row_indexes] * 0.0
    # revisedY[row_indexes, col_indexes] = 1.0 
    new_target = revisedY
    
    return loss, new_target, corrected_num


def weighted_ce_loss(y_, y, w):
    return -torch.sum(y * torch.log(F.softmax(y_*w, dim=1)))/y.size(0)


def cc_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = - torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss


def rc_loss(outputs, confidence, index):
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * confidence[index, :]
    average_loss = - ((final_outputs).sum(dim=1)).mean()
    return average_loss


def min_loss(output1, target, eps=1e-12):
    output1 = F.softmax(output1, dim=1)
    l =  - target * torch.log(output1 + eps)
    new_labels = torch.zeros_like(output1)
    row_indexes = [ i for i in range(0, output1.size(0))]
    l_clone = l.clone().detach()
    l_clone[l_clone==0] = l_clone.max()
    col_indexes = torch.argmin(l_clone, dim=1)
    new_labels[row_indexes, col_indexes] = 1
    return torch.sum(l * new_labels)/l.size(0)


def gauss_kl_loss(mu, logvar, eps = 1e-12):
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def dirichlet_kl_loss(alpha, prior_alpha):
    KLD = torch.mvlgamma(alpha.sum(1), p=1)-torch.mvlgamma(alpha, p=1).sum(1)-torch.mvlgamma(prior_alpha.sum(1), p=1)+torch.mvlgamma(prior_alpha, p=1).sum(1)+((alpha-prior_alpha)*(torch.digamma(alpha)-torch.digamma(alpha.sum(dim=1, keepdim=True).expand_as(alpha)))).sum(1) 
    return KLD.mean()


# def kl_loss(output, d, method=None):
#     output = F.softmax(output, dim=1)
#     d = F.softmax(d, dim=1)
#     if method == 'right':
#         loss = output*torch.log(output) - output*torch.log(d)
#     if method == 'left':
#         loss = d * torch.log(d) - d * torch.log(output)
#     if method == None:
#         right = output*torch.log(output) - output*torch.log(d)
#         left = d * torch.log(d) - d * torch.log(output)
#         loss = right + left
#     return loss.mean()

def kl_loss(output, d, target, eps=1e-8):
    output = F.softmax(output, dim=1) 
    right_weight = output.clone().detach() * target
    right_weight = right_weight / right_weight.sum(dim=1, keepdim=True)
    right = torch.zeros_like(output)
    left = torch.zeros_like(output)
    right[target == 1] = ( - right_weight.clone().detach() * torch.log(d + eps))[target == 1] 
    left[target == 1] =  ( - d.clone().detach() * torch.log(output+eps))[target == 1]
    print("output",output[0])
    print("d",d[0])
    print("ri-w", right_weight[0])
    print("ri", right[0])
    print("target", target[0])
    print(right.sum().item())
    print(left.sum().item())
    loss = right + left
    loss = loss.sum() / loss.size(0)
    return loss

def label_loss(d, labels, eps=1e-12):
    d = F.softmax(d, dim=1)
    l = labels * torch.log(d)
    loss = (-torch.sum(l)) / l.size(0)
    return loss


def alpha_loss(alpha, prior_alpha):
    KLD = torch.mvlgamma(alpha.sum(1), p=1)-torch.mvlgamma(alpha, p=1).sum(1)-torch.mvlgamma(prior_alpha.sum(1), p=1)+torch.mvlgamma(prior_alpha, p=1).sum(1)+((alpha-prior_alpha)*(torch.digamma(alpha)-torch.digamma(alpha.sum(dim=1, keepdim=True).expand_as(alpha)))).sum(1) 
    return KLD.mean()


def BetaMAP_loss(output, target, alpha, beta):
    L1 = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    L1 = (-torch.sum(L1))/L1.size(0)
    L2 = alpha * torch.log(output) + beta * torch.log(1-output)
    L2 = (-torch.sum(L2))/L2.size(0)
    L = 0.01 * L1 + 0.99 * L2
    print(L1.item(), L2.item(), L.item())
    return L


def MAP_loss(t_o, c_o, r_t, r_c, targets, alpha, beta, gamma):
    # 从候选集合中随机划分真实标签和候选标签
    L1_1 = torch.log(t_o) * r_t 
    # L1_2 = torch.log(c_o) * r_c + torch.log(1 - c_o) * (1 - r_c)
    L1_1 = (- torch.sum(L1_1))/L1_1.size(0) 
    # L1_2 = (- torch.mean(L1_2))
    # L1 = L1_1 + L1_2
    gamma = (gamma - 1) * targets
    gamma = gamma / torch.sum(gamma, keepdim=True, dim=1)
    L2 = gamma  * torch.log(t_o)
    L2 = (- torch.sum(L2))/L2.size(0)
    # print(t_o[0:2])
    # L3 = alpha*torch.log(c_o) + beta*torch.log(1-c_o)
    # L3 = - torch.mean(L3)
    # print(L1_1, L1_2, L2, L3)
    # if torch.isnan(L1):
    #     exit()
    # L = L1 + L2 + L3
    print(gamma[0:2])
    print(L1_1.item(), L2.item())
    L = L2
    return L


