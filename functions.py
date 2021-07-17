import torch
from variables import signal



def sig_loss(expectedSignal,expectedBackground):
    def sigloss(y_true,y_pred):
        #print(expectedBackground)
        #print(expectedSignal)
        print(expectedSignal)
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
        print(expectedSignal)
        #print(expectedBackground)
        #print(expectedSignal)
        signalWeight = expectedSignal/torch.sum(y_true)    #expected/actual signal numbers
        backgroundWeight = expectedBackground/torch.sum(1-y_true)  #expected/actual background numbers

        s = signalWeight*torch.sum(y_pred*y_true)
        b = backgroundWeight*torch.sum(y_pred*(1-y_true))

        print("s {}".format(s))
        print("b {}".format(b))
        print("s+b {}".format(s+b))

        #exp = torch.exp(-(s*s)/(s+b+ 0.000000001))
        #scaled_exp = torch.exp(-(s*s)/(s+b+ 0.000000001))*10000000
        return -(s*s)/(s+b+0.000001)
    return sigloss






def significance_loss(target,output,batch_size):
    torch.autograd.set_detect_anomaly(True)
    print("batch size is{}".format(batch_size))

    target_matrix = torch.zeros(len(target),10)


    for i in range(len(target)):
        if target[i] == signal:
            target_matrix[i][signal] = 1
    #print(target_matrix)
    #print(output)



    sigloss = sig_loss2(batch_size/10,9*batch_size/10)
    loss = sigloss(target_matrix,output)
    return loss




