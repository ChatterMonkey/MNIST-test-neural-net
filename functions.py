import torch
from variables import signal,train_batch_size,test_batch_size

def prepare_target(target):
    for i in range(len(target)):
        if target[i].item() == signal: #prepare target
            #print("yes")
            target[i] = 1
        else:
            #print("no")
            #print(target[i].item())
            #print(signal)
            target[i] = 0
    target = target.float()
    return target

def sig_loss(expectedSignal,expectedBackground):
    def sigloss(y_true,y_pred):


        signalWeight = expectedSignal/torch.sum(y_true)    #expected/actual signal numbers
        backgroundWeight = expectedBackground/torch.sum(1-y_true)   #expected/actual background numbers



        y_pred_rearanged = torch.reshape(y_pred,(-1,))


        s = torch.sum(y_pred_rearanged*y_true)

        b = torch.sum(y_pred_rearanged*(1-y_true))


        return -(s*s)/(s+b+0.000001)
    return sigloss






def significance_loss(target,output,using_full_dataset):

    target = prepare_target(target)

    if using_full_dataset:
        sigloss = sig_loss(len(target)/10,9*len(target)/10)
    else:
        sigloss = sig_loss(len(target)/10,len(target)/10)

    loss = sigloss(target,output)
    return loss
