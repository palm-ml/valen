import torch 
import torch.nn.functional as F

def squared_hinge_loss(z):
    hinge = torch.clamp(1-z, min=0)
    return hinge*hinge

def clpl_loss(output, partialY):
    outputs = F.softmax(output, dim=1)
    candidate_outputs = ((outputs*partialY).sum(dim=1))/(partialY.sum(dim=1))
    candidate_loss = squared_hinge_loss(candidate_outputs) # n 
    noncandidate_loss = (squared_hinge_loss(-outputs)*(1-partialY)).sum(dim=1) # n
    average_loss = (candidate_loss + noncandidate_loss).mean()
    return average_loss