import torch
from physicsdataset.phy_variables import variables
import torch.nn.functional as f
import math

def sig_loss(output,target,weights):
    signalWeight = variables.expectedSignal/torch.sum(target)
    backgroundWeight = variables.expectedBackground/torch.sum(1-target)
    s = signalWeight*torch.sum(output*target*weights)
    b = backgroundWeight*torch.sum(output*(1-target)*weights)
    return -(s*s)/(s+b+0.000001)

def sig_invert(output,target,weights):
    signalWeight = variables.expectedSignal/torch.sum(target)
    backgroundWeight = variables.expectedBackground/torch.sum(1-target)
    s = signalWeight*torch.sum(output*target*weights)
    b = backgroundWeight*torch.sum(output*(1-target)*weights)

    return (s+b)/(s*s+0.000001)


def find_loss(output, target, weights, loss_function_id):
    if loss_function_id == 0:
        loss = f.mse_loss(output, target)
    elif loss_function_id == 1:
        loss = sig_loss(output,target,weights)
    elif loss_function_id == 2:
        loss_function = torch.nn.BCELoss(weights)
        loss = loss_function(output,target)
    elif loss_function_id == 3:
        loss = asimov_loss(output,target,weights)
    elif loss_function_id == 4:
        loss = sig_invert(output,target, weights)

    else:
        print("LOSS FUNCTION ID NOT VAID")
        return "LOSS FUNCTION ID NOT VAID"
    return loss

def asimov_significance(output, target, weights):

    signalWeight=variables.expectedSignal/torch.sum(target)
    bkgdWeight = variables.expectedBackground/torch.sum(1-target)

    s = signalWeight*torch.sum(output*target*weights)
    print(output*target)
    print(weights)
    print(output*target*weights)
    b = bkgdWeight*torch.sum(output*(1 - target)*weights)
    sigB = variables.systematic*b

    return torch.sqrt(2*((s+b)*torch.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB+0.000001)+0.000001)-b*b*torch.log(1+sigB*sigB*s/(b*(b+sigB*sigB)+0.000001))/(sigB*sigB+0.000001)))

def asimov_from_tp_fp(s,b,systematic): #DOES NOT WEIGHT S and B
    sigB = b * systematic

    sigB = systematic*b
    lnone = ln_oned(s,b,sigB)
    lntwo = ln_twod(s,b,sigB)
    #print(lnone)
    #print(lntwo)


    asimov_loss = math.sqrt(2*((s+b)*lnone - (b*b)/(sigB*sigB+ 0.000001)*lntwo))
    return asimov_loss


def discrete_asimov_significance(output, target, weights, cutoff, systematic):

    b = 0
    s = 0
    for i in range(len(output)):
        if target[i].item() == 0:
            if (output[i].item() > cutoff) or (output[i].item == cutoff):
                b += weights[i]
        else:
            if (output[i].item() > cutoff) or (output[i].item == cutoff):
                s += weights[i]


    #print("{},{}".format(s,b))
    sigB = systematic*b
    lnone = ln_oned(s,b,sigB)
    lntwo = ln_twod(s,b,sigB)
    #print(lnone)
    #print(lntwo)


    asimov_loss = math.sqrt(2*((s+b)*lnone - (b*b)/(sigB*sigB+ 0.000001)*lntwo))
    return asimov_loss


def asimov_loss(output, target, weights):


    significance = asimov_significance(output,target, weights)
    return -significance


def ln_one(s,b,sigB):

    ln_one = torch.log(((s + b)*(b + sigB*sigB))/( b*b + (s+b)*sigB*sigB+ 0.000001) + 0.00000001)

    return ln_one
def ln_two(s,b,sigB):
    ln_two = torch.log(1 + (sigB*sigB*s)/(b*(b + sigB*sigB)))

    return ln_two



def ln_oned(s,b,sigB):

    ln_one = math.log(((s + b)*(b + sigB*sigB) + 0.000001)/( b*b + (s+b)*sigB*sigB+ 0.000001) )

    return ln_one
def ln_twod(s,b,sigB):
    ln_two = math.log(1 + (sigB*sigB*s)/(b*(b + sigB*sigB)+ 0.000001))

    return ln_two
