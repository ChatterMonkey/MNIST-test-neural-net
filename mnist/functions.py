import torch
from mnist.mnist_variables import variables


def prepare_target(target):
    for i in range(len(target)):
        if target[i].item() == variables.signal: #prepare target
            target[i] = 1
        else:
            target[i] = 0
    target = target.float()
    return target


def subset_data(data,target,background):

    target_counts = torch.unique(target,True,False,True) # 0 returns a list of the unique values in the target and 1 gives their number of occurences
    len_subset = 0
    if target_counts[0].shape[0] != 10:
        print("SUSPICOUS OCURRENCE, not all numbers are present in the data") #possibly becasue the batch size is just small

    for i in range(target_counts[0].shape[0]): #find where the signal and background are
        if target_counts[0][i] == variables.signal or target_counts[0][i] == background:
            len_subset += target_counts[1][i]
    #print(len_subset)

    data_subset = torch.zeros(len_subset,1,28,28)
    target_subset = torch.zeros(len_subset)

    i =0
    for j in range(target.shape[0]):
        if target[j] == variables.signal or target[j] == background:
            data_subset[i] = data[j]
            target_subset[i] = target[j]
            i += 1
    return data_subset,target_subset


def sig_loss(expectedSignal,expectedBackground):
    def sigloss(y_true,y_pred):
        signalWeight = expectedSignal/torch.sum(y_true)    #expected/actual signal numbers
        backgroundWeight = expectedBackground/torch.sum(1-y_true)   #expected/actual background numbers
        y_pred_rearanged = torch.reshape(y_pred,(-1,))
        s = signalWeight*torch.sum(y_pred_rearanged*y_true)
        b = backgroundWeight*torch.sum(y_pred_rearanged*(1-y_true))
        return -(s*s)/(s+b+0.000001)
    return sigloss

def mod_sigloss(expectedSignal,expectedBackground):
    def mod_sigloss(y_true,y_pred):
        signalWeight = expectedSignal/torch.sum(y_true)    #expected/actual signal numbers
        backgroundWeight = expectedBackground/torch.sum(1-y_true)   #expected/actual background numbers
        y_pred_rearanged = torch.reshape(y_pred,(-1,))
        s = signalWeight*torch.sum(y_pred_rearanged*y_true)
        b = backgroundWeight*torch.sum(y_pred_rearanged*(1-y_true))
        return -(s*s)/(s+b+0.000001)
    return mod_sigloss


def significance_loss(target,output,using_full_dataset):

    target = prepare_target(target)

    if using_full_dataset:
        sigloss = sig_loss(len(target)/10,9*len(target)/10)
    else:
        sigloss = sig_loss(len(target)/10,len(target)/10)

    loss = sigloss(target,output)
    return loss


def modified_significance_loss(target,output):
    target = prepare_target(target)

    exp_signal = 1
    exp_bkgd = 900

    sigloss = mod_sigloss(exp_signal,exp_bkgd)
    loss = sigloss(target,output)

    return loss
