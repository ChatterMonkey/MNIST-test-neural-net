import torch
from physicsdataset.phy_variables import variables
import torch.nn.functional as f

def sig_loss(output,target):
    signalWeight = variables.expectedSignal/torch.sum(target)
    backgroundWeight = variables.expectedBackground/torch.sum(1-target)
    s = signalWeight*torch.sum(output*target)
    b = backgroundWeight*torch.sum(output*(1-target))
    return -(s*s)/(s+b+0.000001)


def find_loss(output, target, loss_function_id):
    if loss_function_id == 0:
        loss = f.mse_loss(output, target)
    elif loss_function_id == 1:
        loss = sig_loss(output,target)
    elif loss_function_id == 2:
        loss = f.binary_cross_entropy(output,target)
    else:
        print("LOSS FUNCTION ID NOT VAID")
        return "LOSS FUNCTION ID NOT VAID"
    return loss
