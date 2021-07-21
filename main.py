import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from net import Net
from variables import *
from train import train
from test import test
from loaders import test_loader,train_loader
from functions import subset_data

#for batch, (data, target) in enumerate(train_loader):
#    for batch, (data, target) in enumerate(test_loader):
torch.backends.cudnn.enabled = False

using_full_data = True
loss_function_ids = {"Mean Squared Error":0,"Significance Loss":1}



def initilize():
    torch.manual_seed(random_seed)
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    return network, optimizer


def train_and_test(loss_function_id):

    network, optimizer = initilize()
    test_loss_list = []
    train_loss_list = []

    for epoch in range(1, n_epochs + 1):
        print("Epoch number {}".format(epoch))

        for batch, (data, target) in enumerate(train_loader):
            if using_full_data == False:
                data,target = subset_data(data,target,background)
            loss = train(network,optimizer,data,target,loss_function_id)
            train_loss_list.append(loss)
        print("LOSS")
        print(train_loss_list)

        total_number_correct = 0
        total_number = 0
        false_positive_count = 0
        true_positive_count = 0

        for batch, (data, target) in enumerate(test_loader):
            if using_full_data == False:
                data,target = subset_data(data,target,background)
            test_losses,batch_number_correct,batch_true_positive_count,batch_false_positive_count = test(network, data,target,loss_function_id)
            test_loss_list.append(test_losses)
            total_number += target.shape[0]
            total_number_correct += batch_number_correct
            true_positive_count += batch_true_positive_count
            false_positive_count += batch_false_positive_count
        print("LOSS")
        print("{} {} {} {}".format(total_number,total_number_correct,false_positive_count,true_positive_count))



    plt.subplot(1, 2, 1)
    plt.xlabel("Batch # (15 batches per epoch)")
    plt.ylabel("Training Loss")
    # plt.ylim(-10,10)
    plt.plot(train_loss_list)
    #plt.savefig("/Users/mayabasu/PycharmProjects/MNIST-test-neural-net2/matplotlib_output/lr0.001train.png")


    plt.subplot(1, 2, 2)
    plt.xlabel("Batch # (10 batches per epoch)")
    plt.ylabel("Test Loss")
    plt.plot(test_loss_list)
    print(test_loss_list)
    plt.savefig("/Users/mayabasu/PycharmProjects/MNIST-test-neural-net2/matplotlib_output/loss.png")


train_and_test(0)
