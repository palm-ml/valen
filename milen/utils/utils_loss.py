import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def jin_lossb(outputs, partialY):
    Y = partialY/partialY.sum(dim=1,keepdim=True)
    q = 0.7
    sm_outputs = F.softmax(outputs, dim=1)
    pow_outputs = torch.pow(sm_outputs, q)
    sample_loss = (1-(pow_outputs*Y).sum(dim=1))/q 
    return sample_loss

def jin_lossu(outputs, partialY):
    Y = partialY/partialY.sum(dim=1,keepdim=True)
    logsm = nn.LogSoftmax(dim=1)
    logsm_outputs = logsm(outputs)
    final_outputs = logsm_outputs * Y
    sample_loss = - final_outputs.sum(dim=1)
    return sample_loss

def cour_lossb(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    candidate_outputs = ((sm_outputs*partialY).sum(dim=1))/(partialY.sum(dim=1))
    sig = nn.Sigmoid()
    candidate_loss = sig(candidate_outputs) 
    noncandidate_loss = (sig(-sm_outputs)*(1-partialY)).sum(dim=1) 
    sample_loss = (candidate_loss + noncandidate_loss).mean()
    return sample_loss

def squared_hinge_loss(z):
    hinge = torch.clamp(1-z, min=0)
    return hinge*hinge

def cour_lossu(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    candidate_outputs = ((sm_outputs*partialY).sum(dim=1))/(partialY.sum(dim=1))
    candidate_loss = squared_hinge_loss(candidate_outputs) 
    noncandidate_loss = (squared_hinge_loss(-sm_outputs)*(1-partialY)).sum(dim=1) 
    sample_loss = (candidate_loss + noncandidate_loss).mean()
    return sample_loss

def mae_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = nn.L1Loss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, partialY.float())
    sample_loss = loss_matrix.sum(dim=-1)
    return sample_loss

def mse_loss(outputs, Y):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = nn.MSELoss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, Y.float())
    sample_loss = loss_matrix.sum(dim=-1)
    return sample_loss

def gce_loss(outputs, Y):
    q = 0.7
    sm_outputs = F.softmax(outputs, dim=1)
    pow_outputs = torch.pow(sm_outputs, q)
    sample_loss = (1-(pow_outputs*Y).sum(dim=1))/q # n
    return sample_loss

def phuber_ce_loss(outputs, Y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trunc_point = 0.1
    n = Y.shape[0]
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * Y
    final_confidence = final_outputs.sum(dim=1)
   
    ce_index = (final_confidence > trunc_point)
    sample_loss = torch.zeros(n).to(device)

    if ce_index.sum() > 0:
        ce_outputs = outputs[ce_index,:]
        logsm = nn.LogSoftmax(dim=-1) # because ce_outputs might have only one example
        logsm_outputs = logsm(ce_outputs)
        final_ce_outputs = logsm_outputs * Y[ce_index,:]
        sample_loss[ce_index] = - final_ce_outputs.sum(dim=-1)

    linear_index = (final_confidence <= trunc_point)

    if linear_index.sum() > 0:
        sample_loss[linear_index] = -math.log(trunc_point) + (-1/trunc_point)*final_confidence[linear_index] + 1

    return sample_loss

def cce_loss(outputs, Y):
    logsm = nn.LogSoftmax(dim=1)
    logsm_outputs = logsm(outputs)
    final_outputs = logsm_outputs * Y
    sample_loss = - final_outputs.sum(dim=1)
    return sample_loss

def focal_loss(outputs, Y):
    logsm = nn.LogSoftmax(dim=1)
    logsm_outputs = logsm(outputs)
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = logsm_outputs * Y * (1-sm_outputs) ** 0.5
    sample_loss = - final_outputs.sum(dim=1)
    return sample_loss

def pll_estimator(loss_fn, outputs, partialY, device):
    n, k = partialY.shape[0], partialY.shape[1]
    comp_num = partialY.sum(dim=1)
    temp_loss = torch.zeros(n, k).to(device)

    for i in range(k):
        tempY = torch.zeros(n, k).to(device)
        tempY[:, i] = 1.0 
        temp_loss[:, i] = loss_fn(outputs, tempY)

    coef = 1.0 / comp_num
    total_loss = coef * (temp_loss*partialY).sum(dim=1) 
    total_loss = total_loss.sum()
    return total_loss

# def proden_loss(output1, target, true, eps=1e-12):
#     output = F.softmax(output1, dim=1)
#     l = target * torch.log(output)
#     loss = (-torch.sum(l)) / l.size(0)

#     revisedY = target.clone()
#     revisedY[revisedY > 0] = 1
#     # revisedY = revisedY * (output.clone().detach())
#     revisedY = revisedY * output
#     revisedY = revisedY / (revisedY).sum(dim=1).repeat(revisedY.size(1), 1).transpose(0, 1)
#     new_target = revisedY

class proden_loss:
    def __init__(self, train_p_Y, device):
        self.conf = train_p_Y / train_p_Y.sum(dim=1, keepdim=True)
        self.conf = self.conf.to(device)
        self.device = device
    
    def __call__(self, output1, indexes):
        target = self.conf[indexes].clone().detach()
        output = F.softmax(output1, dim=1)
        l = target * F.log_softmax(output1, dim=1)
        loss = (-torch.sum(l)) / l.size(0)

        return loss

    def update_conf(self, output1, indexes):
        target = self.conf[indexes].clone().detach()
        output = F.softmax(output1, dim=1)
        revisedY = target.clone()
        revisedY[revisedY > 0] = 1
        # revisedY = revisedY * (output.clone().detach())
        revisedY = revisedY * output
        revisedY = revisedY / (revisedY).sum(dim=1).repeat(revisedY.size(1), 1).transpose(0, 1)
        self.conf[indexes,:] = revisedY.clone().detach()

        

def cc_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = - torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss


# def rc_loss(outputs, confidence, index):
#     logsm_outputs = F.log_softmax(outputs, dim=1)
#     final_outputs = logsm_outputs * confidence[index, :]
#     average_loss = - ((final_outputs).sum(dim=1)).mean()
#     return average_loss

class rc_loss:
    def __init__(self, train_p_Y, device):
        self.conf = train_p_Y / train_p_Y.sum(dim=1, keepdim=True)
        self.conf = self.conf.to(device)
        self.device = device
    
    def __call__(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.conf[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss
    
    def update_conf(self, model, batchX, batchY, batch_index):
        confidence = self.conf.clone().detach()
        with torch.no_grad():
            batch_outputs = model(batchX)
            temp_un_conf = F.softmax(batch_outputs, dim=1)
            confidence[batch_index,:] = temp_un_conf * batchY # un_confidence stores the weight of each example
            #weight[batch_index] = 1.0/confidence[batch_index, :].sum(dim=1)
            base_value = confidence.sum(dim=1).unsqueeze(1).repeat(1, confidence.shape[1])
            confidence = confidence/base_value
        self.conf = confidence.clone().detach()

class cavl_loss:
    def __init__(self, train_p_Y, device):
        self.conf = train_p_Y / train_p_Y.sum(dim=1, keepdim=True)
        self.conf = self.conf.to(device)
        self.device = device
    
    def __call__(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.conf[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss
    
    def update_conf(self, model, batchX, batchY, batch_index):
        confidence = self.conf.clone().detach()
        with torch.no_grad():
            batch_outputs = model(batchX)
            cav = (batch_outputs * torch.abs(1 - batch_outputs)) * batchY
            cav_pred = torch.max(cav, dim=1)[1]
            gt_label = F.one_hot(cav_pred, batchY.shape[1])  # label_smoothing() could be used to further improve the performance for some datasets
            confidence[batch_index, :] = gt_label.float()
        self.conf = confidence.clone().detach()

        return confidence

class lws_loss:
    def __init__(self, train_p_Y, device, lw_weight=1, lw_weight0=1, epoch_ratio=None):
        self.conf = train_p_Y / train_p_Y.sum(dim=1, keepdim=True)
        self.conf = self.conf.to(device)
        self.device = device
        self.lw_weight = lw_weight
        self.lw_weight0 = lw_weight0
        self.epoch_ratio=None

    
    def __call__(self, outputs, partialY, index):
        device = self.device
        confidence = self.conf.clone().detach()
        lw_weight = self.lw_weight
        lw_weight0 = self.lw_weight0
        epoch_ratio=self.epoch_ratio

        onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
        onezero[partialY > 0] = 1
        counter_onezero = 1 - onezero
        onezero = onezero.to(device)
        counter_onezero = counter_onezero.to(device)

        sig_loss1 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
        sig_loss1 = sig_loss1.to(device)
        sig_loss1[outputs < 0] = 1 / (1 + torch.exp(outputs[outputs < 0]))
        sig_loss1[outputs > 0] = torch.exp(-outputs[outputs > 0]) / (
            1 + torch.exp(-outputs[outputs > 0]))
        l1 = confidence[index, :] * onezero * sig_loss1
        average_loss1 = torch.sum(l1) / l1.size(0)

        sig_loss2 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
        sig_loss2 = sig_loss2.to(device)
        sig_loss2[outputs > 0] = 1 / (1 + torch.exp(-outputs[outputs > 0]))
        sig_loss2[outputs < 0] = torch.exp(
            outputs[outputs < 0]) / (1 + torch.exp(outputs[outputs < 0]))
        l2 = confidence[index, :] * counter_onezero * sig_loss2
        average_loss2 = torch.sum(l2) / l2.size(0)

        average_loss = lw_weight0 * average_loss1 + lw_weight * average_loss2
        return average_loss#, lw_weight0 * average_loss1, lw_weight * average_loss2
    
    def update_conf(self, model, batchX, batchY, batch_index):
        confidence = self.conf.clone().detach()
        with torch.no_grad():
            device = self.device
            batch_outputs = model(batchX)
            sm_outputs = F.softmax(batch_outputs, dim=1)

            onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1])
            onezero[batchY > 0] = 1
            counter_onezero = 1 - onezero
            onezero = onezero.to(device)
            counter_onezero = counter_onezero.to(device)

            new_weight1 = sm_outputs * onezero
            new_weight1 = new_weight1 / (new_weight1 + 1e-8).sum(dim=1).repeat(
                confidence.shape[1], 1).transpose(0, 1)
            new_weight2 = sm_outputs * counter_onezero
            new_weight2 = new_weight2 / (new_weight2 + 1e-8).sum(dim=1).repeat(
                confidence.shape[1], 1).transpose(0, 1)
            new_weight = new_weight1 + new_weight2

            confidence[batch_index, :] = new_weight
        
        self.conf = confidence.clone().detach()

class plcr_loss:
    def __init__(self, train_p_Y, device, lam=1, is_dynamic_lam=False):
        self.conf = train_p_Y / train_p_Y.sum(dim=1, keepdim=True)
        self.conf = self.conf.to(device)
        self.device = device
        self.consistency_criterion = torch.nn.KLDivLoss(reduction='batchmean').to(device)
        self.lam = lam
        self.is_dynamic_lam = is_dynamic_lam
    
    def __call__(self, y_pred_aug0, y_pred_aug1, y_pred_aug2, targets, indexes, epoch):
        y_pred_aug0_probas_log = torch.log_softmax(y_pred_aug0, dim=-1)
        y_pred_aug1_probas_log = torch.log_softmax(y_pred_aug1, dim=-1)
        y_pred_aug2_probas_log = torch.log_softmax(y_pred_aug2, dim=-1)

        y_pred_aug0_probas = torch.softmax(y_pred_aug0, dim=-1)
        y_pred_aug1_probas = torch.softmax(y_pred_aug1, dim=-1)
        y_pred_aug2_probas = torch.softmax(y_pred_aug2, dim=-1)

        # consist loss
        consist_loss0 = self.consistency_criterion(y_pred_aug0_probas_log,
                                                self.conf[indexes].clone().detach())
        consist_loss1 = self.consistency_criterion(y_pred_aug1_probas_log,
                                                self.conf[indexes].clone().detach())
        consist_loss2 = self.consistency_criterion(y_pred_aug2_probas_log,
                                                self.conf[indexes].clone().detach())
        
        # supervised loss
        super_loss = -torch.mean(
            torch.sum(torch.log(1.0000001 - F.softmax(y_pred_aug0, dim=1)) * (1 - targets), dim=1))
        # dynamic lam
        if self.is_dynamic_lam:
            lam = min((epoch / 100) * self.lam, self.lam)
        else:
            lam = self.lam
        # Unified loss
        final_loss = lam * (consist_loss0 + consist_loss1 + consist_loss2) + super_loss
        # update confidence
        self.confidence_update(y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, targets, indexes)
        return final_loss
    
    def confidence_update(self, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, targets, indexes):
        y_pred_aug0_probas = y_pred_aug0_probas.detach()
        y_pred_aug1_probas = y_pred_aug1_probas.detach()
        y_pred_aug2_probas = y_pred_aug2_probas.detach()

        revisedY0 = targets.clone()

        revisedY0 = revisedY0 * torch.pow(y_pred_aug0_probas, 1 / (2 + 1)) \
                    * torch.pow(y_pred_aug1_probas, 1 / (2 + 1)) \
                    * torch.pow(y_pred_aug2_probas, 1 / (2 + 1))
        revisedY0 = revisedY0 / revisedY0.sum(dim=1).repeat(revisedY0.size(1), 1).transpose(0, 1)

        self.conf[indexes, :] = revisedY0.clone().detach()
        


