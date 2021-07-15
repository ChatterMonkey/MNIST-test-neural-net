import torch
import torch.optim as optim

from net import Net
from test import test
from train import train
from variables import *
from train_sigloss import train_sigloss

torch.backends.cudnn.enabled = False



def initilize():
    torch.manual_seed(random_seed)
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    return network, optimizer



def train_and_test():
    torch.manual_seed(random_seed)
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    #network,optimizer = initilize()

    for epoch in range(1, n_epochs + 1):

        print("Epoch number {}".format(epoch))
        train_losses = train_sigloss(network,optimizer)
        print("Training Complete, losses {}".format(train_losses))
        test_losses,total_number_correct,true_positive_count,false_positive_count = test(network,-1)
        print("Testing Complete")
        print("loss: {} percent accuracy: {}, true positives {], false positives".format(test_losses,total_number_correct/10000,true_positive_count,false_positive_count))
        torch.save(network.state_dict(), '/Users/mayabasu/PycharmProjects/MNIST-test-neural-net2/neuralnets/model.pth')




