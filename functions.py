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

        #exp = torch.exp(-(s*s)/(s+b+ 0.000000001))
        #scaled_exp = torch.exp(-(s*s)/(s+b+ 0.000000001))*10000000
        return -(s*s)/(s+b+0.000001)
    return sigloss



def sig_loss2(expectedSignal,expectedBackground):
    def sigloss(y_true,y_pred):
        #print(expectedBackground)
        #print(expectedSignal)
        signalWeight = expectedSignal/torch.sum(y_true)/10    #expected/actual signal numbers
        backgroundWeight = expectedBackground/torch.sum(1-y_true)/10  #expected/actual background numbers

        s = signalWeight*torch.sum(y_pred*y_true)/10
        b = backgroundWeight*torch.sum(y_pred*(1-y_true))/10

        #exp = torch.exp(-(s*s)/(s+b+ 0.000000001))
        #scaled_exp = torch.exp(-(s*s)/(s+b+ 0.000000001))*10000000
        return -(s*s)/(s+b+0.000001)
    return sigloss






def significance_loss(target,output,batch_size):
    torch.autograd.set_detect_anomaly(True)
    target_list = torch.zeros(len(target)) # list with 1 for signal and 0 for background
    prediction_list = torch.zeros(len(output)) #list of predictions with 1 for signal and 0 for background
    target_matrix = torch.zeros(len(target),10)
    for i in range(len(target)): #setup the target list
        if target[i] == signal:
            target_list[i] = 1
    for i in range(len(output)): #setup the prediction list
        if output[i].argmax().item() == signal:
            prediction_list[i] =0

    for i in range(len(target_list)): #setup the target list
        if target_list[i] == 1:
            target_matrix[i] = torch.ones(10)
        else:
            target_matrix[i] = torch.zeros(10)
    for i in range(len(output)): #setup the prediction list
        if prediction_list[i] == 1:
            output[i] = torch.tensor(1)
        else:
            output[i] = torch.tensor(0)
    print(output)
    print(target_matrix)


    #target_list.requires_grad_(True)
    #prediction_list.requires_grad_(True) #nessecary to compute derivatives for backpropagation

    sigloss = sig_loss2(batch_size/10,9*batch_size/10)
    loss = sigloss(target_matrix,output)
    return loss




