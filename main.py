import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from net import Net
from test import test
from train import train
from variables import *
from train_sigloss import train_sigloss
from loaders import test_loader
import math

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

    test_loss_list = []
    train_loss_list = []

    for epoch in range(1, n_epochs + 1):

        print("Epoch number {}".format(epoch))
        train_losses = train(network,optimizer)

        for i in range(len(train_losses)):
            print(train_losses[i])
            train_loss_list.append(train_losses[i])

        print("Training Complete, losses {}".format(train_losses))
        test_losses,total_number_correct,true_positive_count,false_positive_count,sample_output = test(network,-1)
        for i in range(len(test_losses)):
            test_loss_list.append(math.exp(test_losses[i]))

        examples = enumerate(test_loader)
        batch_idx, (example_data, example_targets) = next(examples)

        #print(network(example_data))



        print("Testing Complete")
        print(total_number_correct)
        print("percent accuracy: {}, true positives {}, false positives {}".format(100*total_number_correct/10000,true_positive_count,false_positive_count))
        torch.save(network.state_dict(), '/Users/mayabasu/PycharmProjects/MNIST-test-neural-net2/neuralnets/model.pth')
        #print("LOSSES")
        #print(train_losses)
        #print(train_loss_list)
        #print(test_losses)
        #print(test_loss_list)
    plt.subplot(1,2,1)
    plt.xlabel("Batch # (15 batches per epoch)")
    plt.ylabel("Ln of the Training Loss")
    #plt.ylim(-10,10)

    plt.plot(train_loss_list)
    plt.savefig("/Users/mayabasu/PycharmProjects/MNIST-test-neural-net2/matplotlib_output/lr0.001train.png")

    #plt.subplot(1,2,2)
    #plt.xlabel("Batch # (10 batches per epoch)")
    #plt.ylabel("Ln of the Test Loss")
    #plt.ylim(-10,10)

    #plt.plot(test_loss_list)




    #plt.savefig("/Users/mayabasu/PycharmProjects/MNIST-test-neural-net2/matplotlib_output/lr0.001test_alone.png")

train_and_test()





