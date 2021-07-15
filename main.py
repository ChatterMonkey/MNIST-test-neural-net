import torch
import torch.optim as optim
import matplotlib.pyplot as plt
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
    loss_list = []

    for epoch in range(1, n_epochs + 1):

        print("Epoch number {}".format(epoch))
        train_losses = train_sigloss(network,optimizer)

        print("Training Complete, losses {}".format(train_losses))
        test_losses,total_number_correct,true_positive_count,false_positive_count,sample_output = test(network,-1)
        for i in range(len(test_losses)):
            loss_list.append(test_losses[i])
        print("Testing Complete")
        print("loss: {} percent accuracy: {}, true positives {}, false positives {}".format(test_losses,total_number_correct/10000,true_positive_count,false_positive_count))
        torch.save(network.state_dict(), '/Users/mayabasu/PycharmProjects/MNIST-test-neural-net2/neuralnets/model.pth')
    plt.plot(loss_list)
    plt.savefig("/Users/mayabasu/PycharmProjects/MNIST-test-neural-net2/matplotlib_output/loss.png")

train_and_test()
ave_deviations = []
i = 7
deviations = []
torch.manual_seed(i)
net = Net()
test_losses,total_number_correct,true_positive_count,false_positive_count,sample_output = test(net)
for j in range(len(sample_output)):
    average = sum(sample_output[j])/10
    for k in range(10):
        deviation = abs(sample_output[j][k] - average)
        deviations.append(deviation)
print("DEVIATIONS = {}".format(deviations))
print("average deviations = {}".format(sum(deviations)/len(deviations)))
ave_deviations.append(sum(deviations)/len(deviations))
print(ave_deviations)




