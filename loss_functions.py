import torch



def significance_loss(expectedSignal,expectedBackground):
    def sigloss(y_true,y_pred):
        signalWeight = expectedSignal/torch.sum(y_true)
        backgroundWeight = expectedBackground/torch.sum(1-y_true)

        s = signalWeight*torch.sum(y_pred*y_true)
        b= backgroundWeight*torch.sum(y_pred*(1-y_true))

        return -(s*s)/(s+b + torch.epsilon())
    return sigloss
