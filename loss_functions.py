import torch



def significance_loss(expectedSignal,expectedBackground):
    def sigloss(y_true,y_pred):
        signalWeight = expectedSignal/torch.sum(y_true)
        backgroundWeight = expectedBackground/torch.sum(1-y_true)

