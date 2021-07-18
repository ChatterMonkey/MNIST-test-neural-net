import torch
from variables import signal

def prepare_target(target):
    for i in range(len(target)):
        if target[i] == signal: #prepare target
            target[i] = 1
        else:
            target[i] = 0
    target = target.float()
    return target

def sig_loss(expectedSignal,expectedBackground):
    def sigloss(y_true,y_pred):
        print(expectedSignal)
        print(torch.sum(y_true))
        print("Y TRUE!!!!!")
        print(y_true)


        signalWeight = expectedSignal/torch.sum(y_true)    #expected/actual signal numbers
        backgroundWeight = expectedBackground/torch.sum(1-y_true)   #expected/actual background numbers
        print("weights are {},{}".format(signalWeight,backgroundWeight))


        y_pred_rearanged = torch.reshape(y_pred,(-1,))


        s = torch.sum(y_pred_rearanged*y_true)
        print("s")
        print(s)
        #print("s = {}".format(s))
        b = torch.sum(y_pred_rearanged*(1-y_true))
        print(b)


        return (s+b)/(s*s +0.000001)
    return sigloss



def sig_loss2(expectedSignal,expectedBackground):
    def sigloss(y_true,y_pred):
        #print(expectedSignal)
        #print(expectedBackground)
        #print(expectedSignal)
        signalWeight = expectedSignal/torch.sum(y_true)    #expected/actual signal numbers
        backgroundWeight = expectedBackground/torch.sum(1-y_true)  #expected/actual background numbers

        s = signalWeight*torch.sum(y_pred*y_true)
        inverted_target_matrix = torch.zeros(y_true.size()[0],10)
        for i in range(y_true.size()[0]):
            if y_true[i][signal] ==0:
                inverted_target_matrix[i][signal] = 1



        b = backgroundWeight*torch.sum(y_pred*inverted_target_matrix)

        #print("s {}".format(s))
        #print("b {}".format(b))
        #print("s+b {}".format(s+b))

        #exp = torch.exp(-(s*s)/(s+b+ 0.000000001))
        #scaled_exp = torch.exp(-(s*s)/(s+b+ 0.000000001))*10000000
        return -(s*s)/(s+b+0.000001)
    return sigloss






def significance_loss2(target,output,batch_size):
    torch.autograd.set_detect_anomaly(True)

    target_matrix = torch.zeros(len(target),10)
    for i in range(len(target)):
        if target[i] == signal:
            target_matrix[i][signal] = 1

    sigloss = sig_loss2(batch_size/10,9*batch_size/10)
    loss = sigloss(target_matrix,output)
    return loss




def significance_loss(target,output,batch_size):
    #print("significance loss called")
    torch.autograd.set_detect_anomaly(True)
    print("target {}".format(target))
    target = prepare_target(target)
    print("preparedtarget {}".format(target))

    sigloss = sig_loss(batch_size/10,9*batch_size/10)
    loss = sigloss(target,output)
    return loss

