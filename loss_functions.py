import torch



def significance_loss(expectedSignal,expectedBackground):
    def sigloss(y_true,y_pred):
        signalWeight = expectedSignal/torch.sum(y_true)
        backgroundWeight = expectedBackground/torch.sum(1-y_true)

        s = signalWeight*torch.sum(y_pred*y_true)
        b= backgroundWeight*torch.sum(y_pred*(1-y_true))

        return -(s*s)/(s+b + 0.000000001)
    return sigloss




expectedSignal = 1
expectedSignal = 9
y_true = torch.tensor([0,0,1,0,0,0,0,0,0])
y_pred = torch.tensor([0,1,0,0,0,0,0,0,0])


sigloss = significance_loss(1,9)

print(sigloss(y_true,y_pred))




