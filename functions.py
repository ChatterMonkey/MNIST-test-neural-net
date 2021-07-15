import torch
from variables import signal



def sig_loss(expectedSignal,expectedBackground):
    def sigloss(y_true,y_pred):
        #print(expectedBackground)
        #print(expectedSignal)
        signalWeight = expectedSignal/torch.sum(y_true)    #expected/actual signal numbers
        backgroundWeight = expectedBackground/torch.sum(1-y_true)   #expected/actual background numbers

        s = signalWeight*torch.sum(y_pred*y_true)
        b = backgroundWeight*torch.sum(y_pred*(1-y_true))

        exp = torch.exp(-(s*s)/(s+b+ 0.000000001))
        scaled_exp = torch.exp(-(s*s)/(s+b+ 0.000000001))*10000000
        return scaled_exp
    return sigloss




def significance_loss(target,output,batch_size):
    target_list = torch.zeros(len(target)) # list with 1 for signal and 0 for background
    prediction_list = torch.zeros(len(output)) #list of predictions with 1 for signal and 0 for background

    for i in range(len(target)): #setup the target list
        if target[i] == signal:
            target_list[i] = 1
    for i in range(len(output)): #setup the prediction list
        if output[i].argmax().item() == signal:
            prediction_list[i] = 1

    target_list.requires_grad_(True)
    prediction_list.requires_grad_(True) #nessecary to compute derivatives for backpropagation

    sigloss = sig_loss(batch_size/10,9*batch_size/10)
    loss = sigloss(target_list,prediction_list)
    return loss




