import torch
from physicsdataset.phy_variables import variables


def sig_loss(output,target):
    signalWeight = variables.expectedSignal/torch.sum(target)
    backgroundWeight = variables.expectedBackground/torch.sum(1-target)
    s = signalWeight*torch.sum(output*target)
    b = backgroundWeight*torch.sum(output*(1-target))
    return -(s*s)/(s+b+0.000001)

