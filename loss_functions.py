import torch



def significance_loss(expectedSignal,expectedBackground):
    def sigloss(y_true,y_pred):
        print(expectedBackground)
        print(expectedSignal)
        signalWeight = expectedSignal/torch.sum(y_true)    #expected/actual signal numbers
        backgroundWeight = expectedBackground/torch.sum(1-y_true)   #expected/actual background numbers

        s = signalWeight*torch.sum(y_pred*y_true)
        b = backgroundWeight*torch.sum(y_pred*(1-y_true))

        return (s+b )/(s*s+ 0.000000001)
    return sigloss



sig_loss = significance_loss(1,10)
sig_loss = sig_loss(torch.tensor([0,1,0,0,0,0,0,0,0,0]),torch.tensor([0,1,0,0,0,0,0,0,0,0]))
print(sig_loss)





