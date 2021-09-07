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
    elif loss_function_id == 3:
        loss = asimov_loss(output,target)

    else:
        print("LOSS FUNCTION ID NOT VAID")
        return "LOSS FUNCTION ID NOT VAID"
    return loss



#def asimov_significance(output, target):
#    signalWeight=variables.expectedSignal/torch.sum(target)
#    bkgdWeight = variables.expectedBackground/torch.sum(1-target)

#    s = signalWeight*torch.sum(torch.round(output)*target)
#    b = bkgdWeight*torch.sum(torch.round(output)*(1 - target))
#    sigB = variables.systematic*b

 #   return torch.sqrt(2*((s+b)*torch.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB+0.000001)+0.000001)-b*b*torch.log(1+sigB*sigB*s/(b*(b+sigB*sigB)+0.000001))/(sigB*sigB+0.000001)))
def asimov_significance(output, target):
    signalWeight=variables.expectedSignal/torch.sum(target)
    bkgdWeight = variables.expectedBackground/torch.sum(1-target)

    s = signalWeight*torch.sum(output*target)
    b = bkgdWeight*torch.sum(output*(1 - target))
    sigB = variables.systematic*b

    return torch.sqrt(2*((s+b)*torch.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB+0.000001)+0.000001)-b*b*torch.log(1+sigB*sigB*s/(b*(b+sigB*sigB)+0.000001))/(sigB*sigB+0.000001)))

def asimov_loss(output, target):


    significance = asimov_significance(output,target)
    return 1/significance
