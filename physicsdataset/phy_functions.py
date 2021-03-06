import numpy
import torch
from physicsdataset.phy_variables import variables
import torch.nn.functional as f
import math
import pandas
import time

def sig_loss(output,target,weights):
    signalWeight = target.shape[0]*variables.expectedSignal/torch.sum(target)
    backgroundWeight = target.shape[0]*variables.expectedBackground/torch.sum(1-target)
    s = signalWeight*torch.sum(output*target*weights)
    b = backgroundWeight*torch.sum(output*(1-target)*weights)
    return -(s*s)/(s+b+0.000001)

def sig_invert(output,target,weights):
    signalWeight = target.shape[0]*variables.expectedSignal/torch.sum(target)
    backgroundWeight = target.shape[0]*variables.expectedBackground/torch.sum(1-target)
    s = signalWeight*torch.sum(output*target*weights)
    b = backgroundWeight*torch.sum(output*(1-target)*weights)

    return (s+b)/(s*s+0.000001)


def find_loss(output, target, weights, loss_function_id, s_weight_tracker,b_weight_tracker):
    if loss_function_id == 0:
        print("FIX USE OF WEIGHRS!!!!!!!!!!!!")
        loss = f.mse_loss(output, target)
    elif loss_function_id == 1:
        print("FIX USE OF WEIGHRS!!!!!!!!!!!!")
        loss = sig_loss(output,target,weights)
    elif loss_function_id == 2:
        if weights is None:
            loss = torch.nn.functional.binary_cross_entropy(output, target)
        else:
            loss_function = torch.nn.BCELoss(weights)
            loss = loss_function(output,target)
    elif loss_function_id == 3:
        if weights is None:
            loss, s_weight_tracker,b_weight_tracker = asimov_loss_no_weights(output,target,s_weight_tracker,b_weight_tracker)

        else:
            loss = asimov_loss(output,target,weights)


    elif loss_function_id == 4:
        print("FIX USE OF WEIGHRS!!!!!!!!!!!!")
        loss = sig_invert(output,target, weights)

    else:
        print("LOSS FUNCTION ID NOT VAID")
        return "LOSS FUNCTION ID NOT VAID"
    return loss, s_weight_tracker,b_weight_tracker

def asimov_significance(output, target, weights):

    signalWeight=target.shape[0]*variables.expectedSignal/torch.sum(target)
    bkgdWeight = target.shape[0]*variables.expectedBackground/torch.sum(1-target)

    s = signalWeight*torch.sum(output*target*weights)
    #print(output*target)
    #print(weights)
    #print(output*target*weights)
    b = bkgdWeight*torch.sum(output*(1 - target)*weights)
    sigB = variables.systematic*b
    #calculation of asimov loss
    e = 0.0001
    #return torch.sqrt(2*((s+b)*torch.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB+0.000001)+0.000001)-b*b*torch.log(1+sigB*sigB*s/(b*(b+sigB*sigB)+0.000001))/(sigB*sigB+0.000001)))
    #return torch.sqrt(0.1 + 2*((s+b)*torch.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB))-b*b*torch.log(1+sigB*sigB*s/(b*(b+sigB*sigB)))/(sigB*sigB)))
    print(torch.sqrt(2 * ((s + b) * torch.log(
        (s + b) * (b + sigB * sigB) / (b * b + (s + b) * sigB * sigB + e) + e) - b * b * torch.log(
        1 + sigB * sigB * s / (e + b * (b + sigB * sigB))) / (sigB * sigB + e))))
    return torch.sqrt(2*((s+b)*torch.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB + e) + e)-b*b*torch.log(1+sigB*sigB*s/(e + b*(b+sigB*sigB)))/(sigB*sigB + e)))


def asimov_significance_no_weights(output, target, s_weight_tracker,b_weight_tracker):
    #print("calculating loss")


    #print("OUTPUT")
    #print(output)
    #print("TARGET")
    #print(target)

    signalWeight = target.shape[0]*variables.expectedSignal/torch.sum(target)
    bkgdWeight = target.shape[0]*variables.expectedBackground/torch.sum(1-target)
    s_weight_tracker.append(signalWeight.item())
    b_weight_tracker.append(bkgdWeight.item())
    #print("signal weight, background weight")
    #print(signalWeight)
    #print(bkgdWeight)

    s = signalWeight*torch.sum(output*target)
    #print(output*target)
    #print(target)
    #print(1 - target)
    #print(output)
    #print(output*(1-target))
    b = bkgdWeight*torch.sum(output*(1 - target))

    #print("b is {}".format(b))
    #print("s qx is {}".format(s))
    #if b.isnan().item() == True:
    #    print("b is nan")
    #    time.sleep(1000000)



    sigB = variables.systematic*b
    #print("sigb is {} systematic is {}".format(sigB,variables.systematic))
    #calculation of asimov loss
    #time.sleep(100)
    #print(2 * ((s + b) * torch.log(
     #   (s + b) * (b + sigB * sigB) / (b * b + (s + b) * sigB * sigB + 0.000001) + 0.000001) - b * b * torch.log(
      #  1 + sigB * sigB * s / (b * (b + sigB * sigB) + 0.000001)) / (sigB * sigB + 0.000001)))

    #print(torch.sqrt(2*((s+b)*torch.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB+0.000001)+0.000001)-b*b*torch.log(1+sigB*sigB*s/(b*(b+sigB*sigB)+0.000001))/(sigB*sigB+0.000001))))
   #return torch.sqrt( 2*((s+b)*torch.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB+0.000001)+0.0001)-b*b*torch.log(1+sigB*sigB*s/(b*(b+sigB*sigB)+0.000001))/(sigB*sigB+0.000001)))
    #print("with out {}".format(2*((s+b)*torch.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB))-b*b*torch.log(1+sigB*sigB*s/(b*(b+sigB*sigB)))/(sigB*sigB))))
    #epsilon = 1e-10
    #b = b + epsilon
    #sigB = b + epsilon
    #s = s + epsilon
    #print("after correction {}".format(2*((s+b)*torch.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB))-b*b*torch.log(1+sigB*sigB*s/(b*(b+sigB*sigB)))/(sigB*sigB))))
    #print( 2*((s+b)*torch.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB))-b*b*torch.log(1+sigB*sigB*s/(b*(b+sigB*sigB)))/(sigB*sigB)))
    e= 0.001
    asimov_argument = 2*((s+b)*torch.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB + e) + e)-b*b*torch.log(1+sigB*sigB*s/(e + b*(b+sigB*sigB)))/(sigB*sigB + e))
    asimov_loss =  torch.sqrt(2*((s+b)*torch.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB + e) + e)-b*b*torch.log(1+sigB*sigB*s/(e + b*(b+sigB*sigB)))/(sigB*sigB + e)))

    if numpy.isnan(asimov_loss.item()):
        print(s)
        print(b)
        print(asimov_loss)
        print(asimov_loss.item())
        print(asimov_argument)
  #  if asimov_loss. == :
   #     print("asimov is nan")
    #    time.sleep(1000000)
    return asimov_loss,s_weight_tracker,b_weight_tracker
def asimov_from_tp_fp(s,b,systematic): #DOES NOT WEIGHT S and B
    #print("info about asimov stuff")
    #print(s)
    #print(b)
    #print(systematic)
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
def asimov_loss_no_weights(output, target, s_weight_tracker,b_weight_tracker):

    significance,s_weight_tracker,b_weight_tracker = asimov_significance_no_weights(output,target, s_weight_tracker,b_weight_tracker)
    return -significance, s_weight_tracker,b_weight_tracker


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
